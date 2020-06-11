/* Intel TBB device driver layer implementation derived from pthread

   Copyright (c) 2011-2019 pocl developers

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to
   deal in the Software without restriction, including without limitation the
   rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
   sell copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
   IN THE SOFTWARE.
*/

#define _GNU_SOURCE

#ifdef __linux__
#include <sched.h>
#endif

#include <algorithm>
#include <pthread.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <tbb/parallel_for.h>
#include <tbb/partitioner.h>
#include <tbb/task_arena.h>

#include "tbb_scheduler.h"
#include "pocl_cl.h"
#include "tbb.h"
#include "tbb_utils.h"
#include "utlist.h"
#include "pocl_util.h"
#include "common.h"
#include "pocl_mem_management.h"

static void* pocl_tbb_driver_thread (void *p);

typedef struct scheduler_data_
{
  unsigned printf_buf_size;
  void* printf_buf_global_ptr;

  size_t local_mem_size;
  void* local_mem_global_ptr;

  size_t num_tbb_threads;

  _cl_command_node *work_queue __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  POCL_FAST_LOCK_T wq_lock_fast __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  pthread_t meta_thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  pthread_cond_t wake_meta_thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  int meta_thread_shutdown_requested;
} scheduler_data __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

static scheduler_data scheduler;

/* External functions declared in tbb_scheduler.h */

void
tbb_scheduler_init (cl_device_id device)
{
  POCL_FAST_INIT (scheduler.wq_lock_fast);

  pthread_cond_init (&(scheduler.wake_meta_thread), NULL);

  scheduler.printf_buf_size = device->printf_buffer_size;
  assert (device->printf_buffer_size > 0);

  /* safety margin - aligning pointers later (in kernel arg setup)
   * may require more local memory than actual local mem size.
   * TODO fix this */
  scheduler.local_mem_size = device->local_mem_size << 4;

  scheduler.num_tbb_threads = device->max_compute_units;

  /* alloc global memory for all threads
   * TODO memory might not be aligned for all threads, just for the first one */
  scheduler.printf_buf_global_ptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, scheduler.printf_buf_size * scheduler.num_tbb_threads);
  scheduler.local_mem_global_ptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, scheduler.local_mem_size * scheduler.num_tbb_threads);

  /* create one meta thread to serve as an async interface thread */
  pthread_create (&scheduler.meta_thread, NULL, pocl_tbb_driver_thread, NULL);
}

void
tbb_scheduler_uninit ()
{
  unsigned i;

  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  scheduler.meta_thread_shutdown_requested = 1;
  pthread_cond_broadcast (&scheduler.wake_meta_thread);
  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

  pthread_join (scheduler.meta_thread, NULL);

  POCL_FAST_DESTROY (scheduler.wq_lock_fast);
  pthread_cond_destroy (&scheduler.wake_meta_thread);

  scheduler.meta_thread_shutdown_requested = 0;

  pocl_aligned_free (scheduler.printf_buf_global_ptr);
  pocl_aligned_free (scheduler.local_mem_global_ptr);
}

/* push_command and push_kernel MUST use broadcast and wake up all threads,
   because commands can be for subdevices (= not all threads) */
void
tbb_scheduler_push_command (_cl_command_node *cmd)
{
  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.work_queue, cmd);
  pthread_cond_broadcast (&scheduler.wake_meta_thread);
  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
}

/* HINT: continue reading from the bottom of this file up to this point */

inline static void
translate_wg_index_to_3d_index (unsigned index, size_t *index_3d, unsigned xy_slice, unsigned row_size)
{
  index_3d[2] = index / xy_slice;
  index_3d[1] = (index % xy_slice) / row_size;
  index_3d[0] = (index % xy_slice) % row_size;
}

class WorkGroupScheduler {
  _cl_command_node *my_k;
  void **my_arguments;
  
public:
  void operator()( const tbb::blocked_range<size_t>& r ) const {
    _cl_command_node *k = my_k;
    uint8_t *arguments = reinterpret_cast<uint8_t*>(my_arguments);
    
    size_t gids[3];
    unsigned slice_size = k->command.run.pc.num_groups[0] * k->command.run.pc.num_groups[1];
    unsigned row_size = k->command.run.pc.num_groups[0];
    
    for( size_t i=r.begin(); i!=r.end(); ++i ) {
      translate_wg_index_to_3d_index(i, gids, slice_size, row_size);
      ((pocl_workgroup_func) k->command.run.wg) (arguments, (uint8_t *)&k->command.run.pc, gids[0], gids[1], gids[2]);
    }
  }

  WorkGroupScheduler( _cl_command_node *k, void **arguments ) :
      my_k(k), my_arguments(arguments)
  {}
};

void
pocl_tbb_run_basic (_cl_command_node *cmd)
{
  struct pocl_argument *al;
  size_t x, y, z;
  unsigned i;
  cl_kernel kernel = cmd->command.run.kernel;
  pocl_kernel_metadata_t *meta = kernel->meta;
  struct pocl_context *pc = &cmd->command.run.pc;

  void **arguments = (void **)malloc (sizeof (void *) * (meta->num_args + meta->num_locals));

  /* Process the kernel arguments. Convert the opaque buffer
     pointers to real device pointers, allocate dynamic local
     memory buffers, etc. */
  for (i = 0; i < meta->num_args; ++i)
    {
      al = &(cmd->command.run.arguments[i]);
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          if (cmd->device->device_alloca_locals)
            {
              /* Local buffers are allocated in the device side work-group
                 launcher. Let's pass only the sizes of the local args in
                 the arg buffer. */
              assert (sizeof (size_t) == sizeof (void *));
              arguments[i] = (void *)al->size;
            }
          else
            {
              arguments[i] = malloc (sizeof (void *));
              *(void **)(arguments[i]) =
                pocl_aligned_malloc(MAX_EXTENDED_ALIGNMENT, al->size);
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          /* It's legal to pass a NULL pointer to clSetKernelArguments. In
             that case we must pass the same NULL forward to the kernel.
             Otherwise, the user must have created a buffer with per device
             pointers stored in the cl_mem. */
          arguments[i] = malloc (sizeof (void *));
          if (al->value == NULL)
            {
              *(void **)arguments[i] = NULL;
            }
          else
            {
              cl_mem m = (*(cl_mem *)(al->value));
              void *ptr = m->device_ptrs[cmd->device->dev_id].mem_ptr;
              *(void **)arguments[i] = (char *)ptr + al->offset;
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE)
        {
          dev_image_t di;
          fill_dev_image_t (&di, al, cmd->device);

          void *devptr = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT,
                                              sizeof (dev_image_t));
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = devptr;
          memcpy (devptr, &di, sizeof (dev_image_t));
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          dev_sampler_t ds;
          fill_dev_sampler_t(&ds, al);
          arguments[i] = malloc (sizeof (void *));
          *(void **)(arguments[i]) = (void *)ds;
        }
      else
        {
          arguments[i] = al->value;
        }
    }

  if (!cmd->device->device_alloca_locals)
    for (i = 0; i < meta->num_locals; ++i)
      {
        size_t s = meta->local_sizes[i];
        size_t j = meta->num_args + i;
        arguments[j] = malloc (sizeof (void *));
        void *pp = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, s);
        *(void **)(arguments[j]) = pp;
      }

  pc->printf_buffer = reinterpret_cast<uchar*> (pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, cmd->device->printf_buffer_size));
  assert (pc->printf_buffer != NULL);
  pc->printf_buffer_capacity = cmd->device->printf_buffer_size;
  assert (pc->printf_buffer_capacity > 0);
  uint32_t position = 0;
  pc->printf_buffer_position = &position;

  unsigned rm = pocl_save_rm ();
  pocl_set_default_rm ();
  unsigned ftz = pocl_save_ftz ();
  pocl_set_ftz (kernel->program->flush_denorms);

  size_t n = pc->num_groups[0] * pc->num_groups[1] * pc->num_groups[2];

#if defined(POCL_TBB_PARTITIONER_AFFINITY)
  tbb::affinity_partitioner ap;
#endif
  tbb::parallel_for (tbb::blocked_range<size_t>(0, n)
                    , WorkGroupScheduler(cmd, arguments)
#if defined(POCL_TBB_PARTITIONER_AFFINITY)
                    , ap
#elif defined(POCL_TBB_PARTITIONER_SIMPLE)
                    , tbb::simple_partitioner()
#elif defined(POCL_TBB_PARTITIONER_STATIC)
                    , tbb::static_partitioner()
#endif
                    );

  pocl_restore_rm (rm);
  pocl_restore_ftz (ftz);

  if (position > 0)
    {
      write (STDOUT_FILENO, pc->printf_buffer, position);
      position = 0;
    }

  pocl_aligned_free (pc->printf_buffer);

  for (i = 0; i < meta->num_args; ++i)
    {
      if (ARG_IS_LOCAL (meta->arg_info[i]))
        {
          if (!cmd->device->device_alloca_locals)
            {
              POCL_MEM_FREE(*(void **)(arguments[i]));
              POCL_MEM_FREE(arguments[i]);
            }
          else
            {
              /* Device side local space allocation has deallocation via stack
                 unwind. */
            }
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_IMAGE
               || meta->arg_info[i].type == POCL_ARG_TYPE_SAMPLER)
        {
          if (meta->arg_info[i].type != POCL_ARG_TYPE_SAMPLER)
            POCL_MEM_FREE (*(void **)(arguments[i]));
          POCL_MEM_FREE(arguments[i]);
        }
      else if (meta->arg_info[i].type == POCL_ARG_TYPE_POINTER)
        {
          POCL_MEM_FREE(arguments[i]);
        }
    }

  if (!cmd->device->device_alloca_locals)
    for (i = 0; i < meta->num_locals; ++i)
      {
        POCL_MEM_FREE (*(void **)(arguments[meta->num_args + i]));
        POCL_MEM_FREE (arguments[meta->num_args + i]);
      }
  free(arguments);

  pocl_release_dlhandle_cache (cmd);
}

static _cl_command_node *
check_cmd_queue_for_device ()
{
  _cl_command_node *cmd;
  DL_FOREACH (scheduler.work_queue, cmd)
  {
    DL_DELETE (scheduler.work_queue, cmd)
    return cmd; // return first cmd, ToDo: make pretty
  }

  return NULL;
}

static int
tbb_scheduler_get_work ()
{
  _cl_command_node *cmd;

  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  int do_exit = 0;

RETRY:
  do_exit = scheduler.meta_thread_shutdown_requested;

  /* execute a command if available */
  cmd = check_cmd_queue_for_device ();
  if (cmd)
    {
      POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

      assert (pocl_command_is_ready (cmd->event));

      if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
        {
          pocl_tbb_run_basic (cmd);
        }
      else
        {
          pocl_exec_command (cmd);
        }

      POCL_FAST_LOCK (scheduler.wq_lock_fast);
    }

  /* if no command was available, sleep */
  if ((cmd == NULL) && (do_exit == 0))
    {
      pthread_cond_wait (&scheduler.wake_meta_thread, &scheduler.wq_lock_fast);
      goto RETRY;
    }

  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

  return do_exit;
}

static void*
pocl_tbb_driver_thread (void *p)
{
  int do_exit = 0;

  while (1)
    {
      do_exit = tbb_scheduler_get_work ();
      if (do_exit)
        {
          pthread_exit (NULL);
        }
    }
}
