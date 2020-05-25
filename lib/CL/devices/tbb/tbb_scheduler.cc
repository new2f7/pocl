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

#include <tbb/task_group.h>

#include "tbb_scheduler.h"
#include "pocl_cl.h"
#include "tbb.h"
#include "tbb_utils.h"
#include "utlist.h"
#include "pocl_util.h"
#include "common.h"
#include "pocl_mem_management.h"

static void* pocl_tbb_driver_thread (void *p);

typedef struct task_data_
{
  /* per-CU (= per-thread) local memory EDIT: now per task/WG local memory, as we leave a lot of decisions to TBB */
  void *local_mem;
  unsigned current_ftz;
  void *printf_buffer;
} task_data __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

typedef struct scheduler_data_
{
  unsigned printf_buf_size;

  size_t local_mem_size;

  _cl_command_node *work_queue __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  kernel_run_command *kernel_queue;

  POCL_FAST_LOCK_T wq_lock_fast __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

  pthread_t meta_thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  pthread_cond_t wake_meta_thread __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));
  
  int meta_thread_shutdown_requested;
} scheduler_data __attribute__ ((aligned (HOST_CPU_CACHELINE_SIZE)));

static scheduler_data scheduler;

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

  /* create one meta thread to serve as an async interface thread */
  pthread_create (&scheduler.meta_thread, NULL, pocl_tbb_driver_thread, NULL);
}

/* External functions declared in tbb_scheduler.h */

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
}

/* push_command and push_kernel MUST use broadcast and wake up all threads,
   because commands can be for subdevices (= not all threads) */
void tbb_scheduler_push_command (_cl_command_node *cmd)
{
  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.work_queue, cmd);
  pthread_cond_broadcast (&scheduler.wake_meta_thread);
  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
}

static void
tbb_scheduler_push_kernel (kernel_run_command *run_cmd)
{
  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  DL_APPEND (scheduler.kernel_queue, run_cmd);
  pthread_cond_broadcast (&scheduler.wake_meta_thread);
  POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
}

/* HINT: continue reading from the bottom of this file up to this point */

inline static void translate_wg_index_to_3d_index (kernel_run_command *k,
                                                   unsigned index,
                                                   size_t *index_3d,
                                                   unsigned xy_slice,
                                                   unsigned row_size)
{
  index_3d[2] = index / xy_slice;
  index_3d[1] = (index % xy_slice) / row_size;
  index_3d[0] = (index % xy_slice) % row_size;
}

static void
task_thread (kernel_run_command *k, task_data *td, int wg_index)
{
  pocl_kernel_metadata_t *meta = k->kernel->meta;

  void *arguments[meta->num_args + meta->num_locals + 1];
  void *arguments2[meta->num_args + meta->num_locals + 1];
  struct pocl_context pc;

  setup_kernel_arg_array_with_locals (
      (void **)&arguments, (void **)&arguments2, k, reinterpret_cast<char*> (td->local_mem),
      scheduler.local_mem_size);
  memcpy (&pc, &k->pc, sizeof(struct pocl_context));

  // capacity and position already set up
  pc.printf_buffer = reinterpret_cast<uchar*> (td->printf_buffer);
  uint32_t position = 0;
  pc.printf_buffer_position = &position;
  assert (pc.printf_buffer != NULL);
  assert (pc.printf_buffer_capacity > 0);
  assert (pc.printf_buffer_position != NULL);

  /* Flush to zero is only set once at start of kernel (because FTZ is
   * a compilation option), but we need to reset rounding mode after every
   * iteration (since it can be changed during kernel execution). */
  unsigned flush = k->kernel->program->flush_denorms;
  if (td->current_ftz != flush)
    {
      pocl_set_ftz (flush);
      td->current_ftz = flush;
    }

  size_t gids[3];
  unsigned slice_size = k->pc.num_groups[0] * k->pc.num_groups[1];
  unsigned row_size = k->pc.num_groups[0];
  translate_wg_index_to_3d_index(k, wg_index, gids, slice_size, row_size);

  pocl_set_default_rm();
  k->workgroup((uint8_t *) arguments, (uint8_t * ) & pc, gids[0], gids[1], gids[2]);

  if (position > 0)
    {
      write (STDOUT_FILENO, pc.printf_buffer, position);
    }

  free_kernel_arg_array_with_locals ((void **)&arguments, (void **)&arguments2, k);
}

static int
work_group_scheduler (kernel_run_command *k)
{
  size_t pool_size = sizeof (task_data) * k->remaining_wgs;
  task_data *task_pool = reinterpret_cast<task_data*> (
        pocl_aligned_malloc (HOST_CPU_CACHELINE_SIZE, pool_size)
                );
  memset (task_pool, 0, pool_size);

  for (int i = 0; i < k->remaining_wgs; ++i) {
    /* some random value, doesn't matter as long as it's not a valid bool - to force a first FTZ setup */
    task_pool[i].current_ftz = 213;
    task_pool[i].printf_buffer = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, scheduler.printf_buf_size);
    assert (task_pool[i].printf_buffer != NULL);
    assert (scheduler.local_mem_size > 0);
    task_pool[i].local_mem = pocl_aligned_malloc (MAX_EXTENDED_ALIGNMENT, scheduler.local_mem_size);
    assert (task_pool[i].local_mem);
  }

  { // begin parallel region
    tbb::task_group g;

    for (int i = 0; i < k->remaining_wgs; ++i)
      {
        g.run([&] {
            task_thread(k, &task_pool[i], i);
        });
      }

    g.wait();
  } // end parallel region

  for (int i = 0; i < k->remaining_wgs; ++i) {
    pocl_aligned_free (task_pool[i].printf_buffer);
    pocl_aligned_free (task_pool[i].local_mem);
  }

  k->wgs_dealt = k->remaining_wgs;
  k->remaining_wgs = 0;
  pocl_aligned_free (task_pool);
}

static void
finalize_kernel_command (kernel_run_command *k)
{
#ifdef DEBUG_MT
  printf("### kernel %s finished\n", k->cmd->command.run.kernel->name);
#endif

  free_kernel_arg_array (k);

  pocl_release_dlhandle_cache (k->cmd);

  pocl_ndrange_node_cleanup (k->cmd);

  POCL_UPDATE_EVENT_COMPLETE_MSG (k->cmd->event, "NDRange Kernel        ");

  pocl_mem_manager_free_command (k->cmd);
  POCL_FAST_DESTROY (k->lock);
  free_kernel_run_command (k);
}

static void
pocl_tbb_prepare_kernel (void *data, _cl_command_node *cmd)
{
  kernel_run_command *run_cmd;
  cl_kernel kernel = cmd->command.run.kernel;
  struct pocl_context *pc = &cmd->command.run.pc;

  pocl_check_kernel_dlhandle_cache (cmd, 1, 1);

  size_t num_groups = pc->num_groups[0] * pc->num_groups[1] * pc->num_groups[2];

  run_cmd = new_kernel_run_command ();
  run_cmd->data = data;
  run_cmd->kernel = kernel;
  run_cmd->device = cmd->device;
  run_cmd->pc = *pc;
  run_cmd->cmd = cmd;
  run_cmd->pc.printf_buffer = NULL;
  run_cmd->pc.printf_buffer_capacity = scheduler.printf_buf_size;
  run_cmd->pc.printf_buffer_position = NULL;
  run_cmd->remaining_wgs = num_groups;
  run_cmd->wgs_dealt = 0;
  run_cmd->workgroup = reinterpret_cast<pocl_workgroup_func> (cmd->command.run.wg);
  run_cmd->kernel_args = cmd->command.run.arguments;
  run_cmd->next = NULL;
  run_cmd->ref_count = 0;
  POCL_FAST_INIT (run_cmd->lock);

  setup_kernel_arg_array (run_cmd);

  pocl_update_event_running (cmd->event);

  tbb_scheduler_push_kernel (run_cmd);
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

static kernel_run_command *
check_kernel_queue_for_device ()
{
  kernel_run_command *cmd;
  DL_FOREACH (scheduler.kernel_queue, cmd)
  {
    DL_DELETE (scheduler.kernel_queue, cmd)
    return cmd; // return first cmd, ToDo: make pretty
  }

  return NULL;
}

static int
tbb_scheduler_get_work ()
{
  _cl_command_node *cmd;
  kernel_run_command *run_cmd;

  /* execute kernel if available */
  POCL_FAST_LOCK (scheduler.wq_lock_fast);
  int do_exit = 0;

RETRY:
  do_exit = scheduler.meta_thread_shutdown_requested;

  run_cmd = check_kernel_queue_for_device ();
  /* execute kernel if available */
  if (run_cmd)
    {
      ++run_cmd->ref_count;
      POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

      work_group_scheduler (run_cmd);

      POCL_FAST_LOCK (scheduler.wq_lock_fast);
      if ((--run_cmd->ref_count) == 0)
        {
          POCL_FAST_UNLOCK (scheduler.wq_lock_fast);
          finalize_kernel_command (run_cmd);
          POCL_FAST_LOCK (scheduler.wq_lock_fast);
        }
    }

  /* execute a command if available */
  cmd = check_cmd_queue_for_device ();
  if (cmd)
    {
      POCL_FAST_UNLOCK (scheduler.wq_lock_fast);

      assert (pocl_command_is_ready (cmd->event));

      if (cmd->type == CL_COMMAND_NDRANGE_KERNEL)
        {
          pocl_tbb_prepare_kernel (cmd->device->data, cmd);
        }
      else
        {
          pocl_exec_command (cmd);
        }

      POCL_FAST_LOCK (scheduler.wq_lock_fast);
    }

  /* if neither a command nor a kernel was available, sleep */
  if ((cmd == NULL) && (run_cmd == NULL) && (do_exit == 0))
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
