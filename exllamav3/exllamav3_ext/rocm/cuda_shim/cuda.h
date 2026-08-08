#pragma once
#include <hip/hip_runtime.h>

// ---------------------------------------------------------------------------
// CUDA driver-API spellings
// ---------------------------------------------------------------------------
// cpu/moe_handoff.cu declares function pointers for the driver-API stream
// memory ops (cuStreamWaitValue32 / cuStreamWriteValue32) and resolves them
// with dlopen("libcuda.so.1"). Those declarations need these two spellings,
// which hip_compat's runtime-API aliases do not cover.
//
// Only the *types* are bridged, deliberately. On ROCm the dlopen finds no
// libcuda, MemOps::resolved stays false, and moe_handoff falls back to its
// flag-kernel path -- which upstream keeps as a supported mode (EXL3_MOE_MEMOPS=0
// forces it), so this degrades correctly rather than silently misbehaving.
//
// HIP does have equivalents -- hipStreamWaitValue32 / hipStreamWriteValue32,
// gated on hipDeviceAttributeCanUseStreamWaitValue -- which would recover the
// "no SM occupancy, no per-wait launch cost" property described in
// moe_handoff.cu. The flag-kernel fallback in use here is correct, and pays one
// kernel launch per wait.

#ifndef CUDAAPI
#define CUDAAPI
#endif

typedef unsigned int cuuint32_t;
typedef unsigned long long cuuint64_t;
