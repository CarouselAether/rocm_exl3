"""Row-tiled paged KV-cache append for RDNA (rocm_py patch EXL3_ROCM_KV_UPDATE_ROWS).

Upstream's _paged_kv_update_kernel (attention_fn/triton_paged.py) launches one program per
(token, kv head): 64 threads move one head's K and V (512 B at head_dim 128) with 4-byte
accesses, after two dependent loads (cache_seqlens, block_table). A 2048-token chunk is 16K
such programs, an 8192-token chunk 64K, and the launch overhead dominates: on gfx1151 an
8192-token append at 8 kv heads x 128 took 2.57 ms per layer (26 GB/s).

For one token, all kv heads are one contiguous n_kv_heads * head_dim row in both the source
[bsz, len, n_kv, hd] and the cache [pages, page_size, n_kv, hd], so a program here copies a
BLOCK_T-token x full-row tile with one page lookup per token: 0.35 ms for the same append.
It is a pure copy, so the cache contents are bit-identical to upstream's.

Installed by giving the upstream kernel object a subclass whose launch (`kernel[grid](...)`)
lands here. It stays a JITFunction, so bc_attn still AOT-compiles the original source for the
graphed decode path, and any call whose tensors are not in the contiguous layout this kernel
assumes falls through to the upstream launch.
"""

import torch
import triton
import triton.language as tl

BLOCK_T = 8
BLOCK_W = 1024
NUM_WARPS = 4


@triton.jit
def _kv_update_rows_kernel(
    k,
    v,
    k_cache,
    v_cache,
    block_table,
    cache_seqlens,
    num_pages_per_seq,
    kv_append_len,
    t_blocks,
    ROW: tl.constexpr,
    page_size: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_W: tl.constexpr,
):
    pid = tl.program_id(0)
    batch = pid // t_blocks
    t = (pid - batch * t_blocks) * BLOCK_T + tl.arange(0, BLOCK_T)
    tmask = t < kv_append_len
    logical_t = tl.load(cache_seqlens + batch) + t
    page = logical_t // page_size
    phys = tl.load(block_table + batch * num_pages_per_seq + page, mask = tmask, other = 0)
    dst_row = (phys.to(tl.int64) * page_size + (logical_t - page * page_size)) * ROW
    src_row = (batch * kv_append_len + t).to(tl.int64) * ROW
    for w0 in tl.static_range(0, ROW, BLOCK_W):
        w = w0 + tl.arange(0, BLOCK_W)
        m = tmask[:, None] & (w < ROW)[None, :]
        s = src_row[:, None] + w[None, :]
        d = dst_row[:, None] + w[None, :]
        tl.store(k_cache + d, tl.load(k + s, mask = m), mask = m)
        tl.store(v_cache + d, tl.load(v + s, mask = m), mask = m)


def _launch_rows(grid, k, v, k_cache, v_cache, block_table, cache_seqlens, num_pages_per_seq,
                 kv_append_len, n_kv_heads, page_size, head_dim):
    """Returns False (caller falls back to upstream) unless every tensor is in the flat
    contiguous layout upstream's kernel indexes as well."""
    if not isinstance(grid, tuple) or not isinstance(k, torch.Tensor) or not isinstance(v, torch.Tensor):
        return False
    if not all(t.is_contiguous() for t in (k, v, k_cache, v_cache, block_table)):
        return False
    bsz = grid[0] // kv_append_len if kv_append_len else 0
    row = n_kv_heads * head_dim
    if bsz <= 0 or k.numel() != bsz * kv_append_len * row or v.numel() != k.numel():
        return False
    if k_cache.shape[-1] != head_dim or k_cache.shape[-2] != n_kv_heads:
        return False
    t_blocks = triton.cdiv(kv_append_len, BLOCK_T)
    _kv_update_rows_kernel[(bsz * t_blocks,)](
        k, v, k_cache, v_cache, block_table, cache_seqlens,
        num_pages_per_seq, kv_append_len, t_blocks,
        row, page_size, BLOCK_T, min(BLOCK_W, triton.next_power_of_2(row)),
        num_warps = NUM_WARPS,
    )
    return True


def install(upstream_kernel) -> None:
    base = type(upstream_kernel)

    class _RowTiledLaunch(base):
        def __getitem__(self, grid):
            upstream = base.__getitem__(self, grid)

            def launch(*args, **kwargs):
                if len(args) == 12 and not set(kwargs) - {"num_warps", "num_stages"}:
                    if _launch_rows(grid, *args[:11]):
                        return None
                return upstream(*args, **kwargs)

            return launch

    upstream_kernel.__class__ = _RowTiledLaunch
