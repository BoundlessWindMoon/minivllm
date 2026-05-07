/**
 * Stub sm-profiler header — no-op implementation.
 * The real sm-profiler is from mega-qwen's third_party/sm-profiler.
 * We stub it out to avoid the dependency while keeping the kernel code unchanged.
 */
#pragma once
#include <cstdint>

typedef void* sm_profiler_buffer_t;

inline sm_profiler_buffer_t sm_profiler_create_buffer(uint32_t, uint32_t, uint32_t, int) { return nullptr; }
inline void sm_profiler_destroy_buffer(sm_profiler_buffer_t) {}
inline void sm_profiler_init_buffer(sm_profiler_buffer_t) {}
inline void sm_profiler_register_event(sm_profiler_buffer_t, uint32_t, const char*) {}
inline uint64_t* sm_profiler_get_device_ptr(sm_profiler_buffer_t) { return nullptr; }
inline void sm_profiler_export_to_file(sm_profiler_buffer_t, const char*) {}

#define sm_profiler_event_start(buf, id, on) (void)0
#define sm_profiler_event_end(buf, id, on) (void)0
