/* pocl_topology.c - retrieving the topology of OpenCL devices

   Copyright (c) 2012,2015 Cyril Roelandt and Pekka Jääskeläinen
   
   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:
   
   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.
   
   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.
*/

#include <stdlib.h>
#include <assert.h>

#include "config.h"

#include <pocl_cl.h>
#include <pocl_file_util.h>

#include "pocl_topology.h"

#ifdef ENABLE_HWLOC

#include <hwloc.h>
#if HWLOC_API_VERSION >= 0x00020000
#define HWLOC_API_2
#else
#undef HWLOC_API_2
#endif

#endif

//#define DEBUG_POCL_TOPOLOGY

/*
 * Sets up:
 *  max_compute_units
 *  global_mem_size
 *  global_mem_cache_type
 *  global_mem_cacheline_size
 *  global_mem_cache_size
 *  local_mem_size
 *  max_constant_buffer_size
 */

#ifdef ENABLE_HWLOC

#ifdef HWLOC_API_2
#define HWLOC_IS_CACHE(type) hwloc_obj_type_is_dcache(type)
#else
#define HWLOC_IS_CACHE(type) (type == HWLOC_OBJ_CACHE)
#endif

/* Returns the highest private cache for a given hwloc_obj_type.
 * Returns the highest/lowest cache if all caches are private/shared. */
hwloc_obj_t
find_highest_private_cache (hwloc_topology_t pocl_topology,
                            hwloc_obj_t highest_cache,
                            hwloc_obj_t lowest_cache,
                            hwloc_obj_type_t type)
{
  /* Look at the first object of the given type. We ignore asymmetric topologies. */
  hwloc_obj_t obj = hwloc_get_next_obj_by_type (pocl_topology, type, NULL);
  if (obj)
    {
      hwloc_obj_t shared_cache = hwloc_get_shared_cache_covering_obj (pocl_topology, obj);
      if (shared_cache)
        {
          if (shared_cache->depth < lowest_cache->depth)
            {
              /* return the highest private cache */
              hwloc_obj_t current = shared_cache;
              while (current->first_child)
                {
                  current = current->first_child;
                  if (HWLOC_IS_CACHE(current->type))
                    return current;
                }
            }
          else /* the lowest cache is shared (shared_cache == lowest_cache) */
            return shared_cache;
        }
      else /* there is no shared cache */
        return highest_cache;
    }
  return NULL;
}

int
pocl_topology_detect_device_info(cl_device_id device)
{
  hwloc_topology_t pocl_topology;
  int ret = 0;

#ifdef HWLOC_API_2
  if (hwloc_get_api_version () < 0x20000)
    POCL_MSG_ERR ("pocl was compiled against libhwloc 2.x but is"
                  "actually running against libhwloc 1.x \n");
#else
  if (hwloc_get_api_version () >= 0x20000)
    POCL_MSG_ERR ("pocl was compiled against libhwloc 1.x but is"
                  "actually running against libhwloc 2.x \n");
#endif

  /*
   * hwloc's OpenCL backend causes problems at the initialization stage
   * because it reloads libpocl.so via the ICD loader.
   *
   * See: https://github.com/pocl/pocl/issues/261
   *
   * The only trick to stop hwloc from initializing the OpenCL plugin
   * I could find is to point the plugin search path to a place where there
   * are no plugins to be found.
   */
  setenv ("HWLOC_PLUGINS_PATH", "/dev/null", 1);

  ret = hwloc_topology_init (&pocl_topology);
  if (ret == -1)
  {
    POCL_MSG_ERR ("Cannot initialize the topology.\n");
    return ret;
  }

#ifdef HWLOC_API_2
  hwloc_topology_set_io_types_filter(pocl_topology, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_SYSTEM, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_GROUP, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_BRIDGE, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_MISC, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_PCI_DEVICE, HWLOC_TYPE_FILTER_KEEP_NONE);
  hwloc_topology_set_type_filter (pocl_topology, HWLOC_OBJ_OS_DEVICE, HWLOC_TYPE_FILTER_KEEP_NONE);
#else
  hwloc_topology_ignore_type (pocl_topology, HWLOC_TOPOLOGY_FLAG_WHOLE_IO);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_SYSTEM);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_GROUP);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_BRIDGE);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_MISC);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_PCI_DEVICE);
  hwloc_topology_ignore_type (pocl_topology, HWLOC_OBJ_OS_DEVICE);
#endif

  ret = hwloc_topology_load (pocl_topology);
  if (ret == -1)
  {
    POCL_MSG_ERR ("Cannot load the topology.\n");
    goto exit_destroy;
  }

#ifdef HWLOC_API_2
  device->global_mem_size =
      hwloc_get_root_obj(pocl_topology)->total_memory;
#else
  device->global_mem_size =
      hwloc_get_root_obj(pocl_topology)->memory.total_memory;
#endif

  /* Get the number of hardware threads from hwloc */
  int depth = hwloc_get_type_depth(pocl_topology, HWLOC_OBJ_PU);
  device->max_compute_units = hwloc_get_nbobjs_by_depth(pocl_topology, depth);

  /* TBD by querying cache information from hwloc */
  size_t global_mem_cache_size = 0, global_mem_cacheline_size = 0, local_mem_size = 0;

  hwloc_obj_t highest_cache = NULL, lowest_cache = NULL;
  /* pointers to the caches actually used at the end */
  hwloc_obj_t global_mem_cache = NULL, cache_as_local_mem = NULL;

  hwloc_obj_t current = hwloc_get_root_obj(pocl_topology);
  while (current->first_child)
    {
      current = current->first_child;
      if (HWLOC_IS_CACHE(current->type))
        {
          if (!highest_cache)
            highest_cache = current;
          lowest_cache = current;
        }
    }

  global_mem_cache = find_highest_private_cache (pocl_topology, highest_cache, lowest_cache, HWLOC_OBJ_CORE);
  if (!global_mem_cache)
    global_mem_cache = find_highest_private_cache (pocl_topology, highest_cache, lowest_cache, HWLOC_OBJ_PU);

  cache_as_local_mem = lowest_cache;

  if ((global_mem_cache) && (global_mem_cache->attr))
    {
      global_mem_cacheline_size = global_mem_cache->attr->cache.linesize;
      global_mem_cache_size = global_mem_cache->attr->cache.size;
      if (global_mem_cacheline_size && global_mem_cache_size)
        {
          device->global_mem_cache_type = 0x2; // CL_READ_WRITE_CACHE, without including all of CL/cl.h
          device->global_mem_cacheline_size = global_mem_cacheline_size;
          device->global_mem_cache_size = global_mem_cache_size;
#ifdef DEBUG_POCL_TOPOLOGY
          printf("detected global_mem_cache_size: %lu\n", device->global_mem_cache_size);
#endif
        }
    }

  if ((cache_as_local_mem) && (cache_as_local_mem->attr))
    {
      local_mem_size = cache_as_local_mem->attr->cache.size;
      if (local_mem_size)
        {
          /* divide by the number of PUs below the selected cache */
          hwloc_const_cpuset_t set = cache_as_local_mem->cpuset;
          unsigned num_pus = hwloc_get_nbobjs_inside_cpuset_by_type (pocl_topology, set, HWLOC_OBJ_PU);
          device->local_mem_size = local_mem_size / num_pus;
          device->max_constant_buffer_size =  local_mem_size / num_pus;
#ifdef DEBUG_POCL_TOPOLOGY
          printf("detected local_mem_size: %lu\n", device->local_mem_size);
#endif
        }
    }

  // Destroy topology object and return
exit_destroy:
  hwloc_topology_destroy (pocl_topology);
  return ret;

}

// #ifdef ENABLE_HWLOC
#elif defined(__linux__) || defined(__ANDROID__)

#define L2_CACHE_SIZE "/sys/devices/system/cpu/cpu0/cache/index2/size"
#define L1_CACHE_SIZE "/sys/devices/system/cpu/cpu0/cache/index1/size"
#define CPUS "/sys/devices/system/cpu/possible"
#define MEMINFO "/proc/meminfo"

int
pocl_topology_detect_device_info (cl_device_id device)
{
  device->global_mem_cacheline_size = HOST_CPU_CACHELINE_SIZE;
  device->global_mem_cache_type
      = 0x2; // CL_READ_WRITE_CACHE, without including all of CL/cl.h

  char *content;
  uint64_t filesize;

  /* global_mem_cache_size and local_mem_size */
  if (pocl_read_file (L2_CACHE_SIZE, &content, &filesize) == 0)
    {
      long val = atol (content);
      device->global_mem_cache_size = val * 1024;
      POCL_MEM_FREE (content);
      if (pocl_read_file (L1_CACHE_SIZE, &content, &filesize) == 0)
        {
          long val = atol (content);
          device->local_mem_size = val * 1024;
          POCL_MEM_FREE (content);
        }
    }
  else
    {
      if (pocl_read_file (L1_CACHE_SIZE, &content, &filesize) == 0)
        {
          long val = atol (content);
          device->global_mem_cache_size = val * 1024;
          device->local_mem_size = val * 1024;
          POCL_MEM_FREE (content);
        }
      else
        {
          POCL_MSG_WARN (
              "Could not figure out CPU cache size, using bogus value\n");
          device->global_mem_cache_size = 1 << 20;
        }
    }

  /* global_mem_size */
  if (pocl_read_file (MEMINFO, &content, &filesize) == 0)
    {
      char *tmp = content;
      unsigned long memsize_kb;
      size_t i;

      while (*tmp && (*tmp != '\n'))
        ++tmp;
      *tmp = 0;
      tmp = content;
      while (*tmp && (*tmp != 0x20))
        ++tmp;
      while (*tmp && (*tmp == 0x20))
        ++tmp;
      int items = sscanf (tmp, "%lu kB", &memsize_kb);

      assert (items == 1);

      device->global_mem_size = memsize_kb * 1024;
      POCL_MEM_FREE (content);
    }
  else
    {
      POCL_MSG_WARN ("Cannot get memory size\n");
      device->global_mem_size = 256 << 20;
    }

  /* max_compute_units */
  if (pocl_read_file (CPUS, &content, &filesize) == 0)
    {
      assert (content);
      assert (filesize > 0);
      unsigned long start, end;
      int items = sscanf (content, "%lu-%lu", &start, &end);
      assert (items == 2);
      device->max_compute_units = (unsigned)end + 1;
      POCL_MEM_FREE (content);
    }
  else
    {
      POCL_MSG_WARN ("Cannot get logical CPU number\n");
      device->max_compute_units = 1;
    }

  return 0;
}

#else

#error Dont know how to get HWLOC-provided values on this system!

#endif
