// Shim: `#include <cooperative_groups.h>` -> HIP's implementation.
// HIP mirrors the CUDA cooperative-groups API including the
// cooperative_groups:: namespace and grid_group / thread_block types.
#pragma once
#include <hip/hip_cooperative_groups.h>
