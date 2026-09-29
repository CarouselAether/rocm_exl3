"""MQA-specialized DSA prefill kernel for RDNA (gfx11), ROCm-owned.

Drop-in replacement for dsa_triton._dsa_attn_kernel (the one-shot sparse attention kernel
every DeepSeek-V4 prefill chunk runs, R > 8 rows or n_splits == 1): same argument list, same
output layout (sinks, eq. 26 de-rotation, group-major store, H32-rotated packed pools,
non-causal image chunks). Only the work split differs.

Why: the upstream kernel gives every program BLOCK_H = 32 heads and the WHOLE output width,
so it carries a 32 x (512 + 64) fp32 accumulator over 128 lanes (144 VGPRs) next to a
resident 32 x 512 q tile, which RDNA3 WMMA replicates across the two half-waves. It compiles
at the 256-VGPR ceiling with ~2.5K VGPRs spilled and ~5 KB scratch per lane (occupancy 17%,
78 B/VALU of mostly scratch traffic), and was ~33% of DS4 pp2048 GPU time.

Same fix as the decode kernel (dsa_decode_rdna.py): one program owns HP heads (the MMA M
dimension; the single latent KV head is shared by all 64 query heads) and one BD-wide
column block of the output for one query row:

  scores = sum_kc q[HP, kc:kc+KC] . K[BN, kc:kc+KC]^T    RUNTIME loop over D, q re-read
                                                          per chunk (L1 hits)
  acc   += p[HP, BN] . V[BN, BD]                          V = K, this program's columns

The virtual key row is [c (D_c) | r (D_r)] = D wide (a power of two): ring / chunk rows are
contiguous, pool rows are pool_c ++ pool_r. Scores are recomputed per column block, so the
MMA work is (D + BD) / (2 BD) of the minimum (1.5x at BD 256, 1.0x at BD = D).

Production tiling (sweep in RDNA_NOTES "DSA prefill MQA kernel"): HP 64 (all heads), BD 512
(whole row, no score recompute), KC 32, BLOCK_N = BLOCK_W 64, 16 warps -> the 64 x 512 fp32
accumulator is 64 VGPRs over 512 lanes; 5 spills / 20 B scratch (25 / 104 B gathered), vs
~2.9K / 5.2 KB upstream. One program per query row.

Not covered (the proxy falls back to the upstream kernel): Q_SPLIT, OUT_LATENT (GLM-5.2
DSA-on-MLA), non-power-of-two D, H not a multiple of HP, D_r == 0.
"""

import os

import triton
import triton.language as tl

from .dsa_decode_rdna import _pool_tile, _win_tile, _rot_cols, _softmax_step


@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0", "R",
    "win_floor", "ring_beg",
])
def _dsa_prefill_mqa_kernel(
    q, ring, kv_chunk, pool_c, pool_r, block_table, indices, sinks, derot_inv_freq, out,
    k_len, win_len, pool_len, num_pages_per_row, q_pos0, R, win_floor, ring_beg,
    pool_s, h32,
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    D_c_pad: tl.constexpr,         # unused (upstream signature)
    D_r: tl.constexpr,
    K_pad: tl.constexpr,
    compress_rate: tl.constexpr,
    scale: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    HAS_SINKS: tl.constexpr,
    DENSE_POOL: tl.constexpr,
    DEROTATE: tl.constexpr,
    HPG: tl.constexpr,
    BLOCK_H: tl.constexpr,         # unused (upstream signature)
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
    NC_BLOCK: tl.constexpr = 0,
    NC_CHUNK: tl.constexpr = 0,
    NC_HIST: tl.constexpr = 0,
    Q_SPLIT: tl.constexpr = 0,     # must be 0
    OUT_LATENT: tl.constexpr = 0,  # must be 0
    QC: tl.constexpr = 0,
    HP: tl.constexpr = 32,         # heads per program (MMA M)
    BD: tl.constexpr = 256,        # output columns per program
    KC: tl.constexpr = 64,         # score reduction chunk over D (runtime loop)
    KSTAGES: tl.constexpr = 1,
):
    D: tl.constexpr = D_c + D_r
    NHG: tl.constexpr = H // HP
    NDB: tl.constexpr = D // BD
    tl.static_assert(Q_SPLIT == 0 and OUT_LATENT == 0)

    pid = tl.program_id(0)
    row = pid // (NHG * NDB)
    sub = pid % (NHG * NDB)
    hgrp = sub // NDB
    dblk = sub % NDB

    offs_h = hgrp * HP + tl.arange(0, HP)
    c0 = dblk * BD
    vcols = c0 + tl.arange(0, BD)
    q_rows = q + (row * H + offs_h)[:, None] * D
    kcols0 = tl.arange(0, KC)

    if HAS_SINKS:
        m_state = tl.load(sinks + offs_h)
        l = tl.full((HP,), 1.0, tl.float32)
    else:
        m_state = tl.full((HP,), -float("inf"), tl.float32)
        l = tl.zeros((HP,), tl.float32)
    acc = tl.zeros((HP, BD), tl.float32)

    # Phase 1: sliding-window rows by absolute position (same addressing as upstream)
    if HAS_WINDOW:
        q_abs = q_pos0 + row
        if NC_BLOCK or NC_CHUNK:
            top = q_pos0 + R - 1
        else:
            top = q_abs
        for n0 in tl.range(0, win_len, BLOCK_W, num_stages = 1):
            offs_j = n0 + tl.arange(0, BLOCK_W)
            abs_pos = top - offs_j
            in_range = (offs_j < win_len) & (abs_pos >= win_floor)
            if NC_CHUNK:
                in_range = in_range & ((abs_pos >= q_pos0) | (abs_pos > q_abs - NC_HIST))
            mc = in_range & (abs_pos >= q_pos0)
            mr = in_range & (abs_pos < q_pos0)
            idx_c = tl.where(mc, abs_pos - q_pos0, 0)
            if NC_BLOCK:
                ap = tl.where(mr, abs_pos, 0)
                w_phys = tl.load(block_table + row * num_pages_per_row + ap // page_size,
                                 mask = mr, other = 0)
                idx_r = w_phys * page_size + ap % page_size
            else:
                idx_r = tl.where(mr, abs_pos - ring_beg, 0)
            # Score reduction streamed over D in a runtime loop (see module docstring)
            scores = tl.zeros((HP, BLOCK_W), tl.float32)
            for kc in tl.range(0, D, KC, num_stages = KSTAGES):
                kcols = kc + kcols0
                qk = tl.load(q_rows + kcols[None, :])
                k = _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, kcols, D)
                if QC > 0:
                    qk = _rot_cols(qk, h32, kcols, D_c, HP, KC)
                    k = _rot_cols(k, h32, kcols, D_c, BLOCK_W, KC)
                scores = tl.dot(qk, tl.trans(k), acc = scores)
            v = _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, vcols, D)
            if QC > 0:
                v = _rot_cols(v, h32, vcols, D_c, BLOCK_W, BD)
            m_state, l, acc = _softmax_step(scores, in_range, m_state, l, acc, v, scale)

    # Phase 2: pool entries -- gathered by index list, or dense with causal bound
    if DENSE_POOL:
        n_end = tl.minimum((q_pos0 + row + 1) // compress_rate, pool_len)
    else:
        n_end = k_len
    for n0 in tl.range(0, n_end, BLOCK_N, num_stages = 1):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if DENSE_POOL:
            idx = tl.where(offs_n < n_end, offs_n, -1)
        else:
            idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < n_end, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        phys = tl.load(block_table + row * num_pages_per_row + idx_s // page_size,
                       mask = in_range, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_prefill: entry idx >= pool_len")
            tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_prefill: pool page OOB")
        tok = phys * page_size + idx_s % page_size
        scores = tl.zeros((HP, BLOCK_N), tl.float32)
        for kc in tl.range(0, D, KC, num_stages = KSTAGES):
            kcols = kc + kcols0
            qk = tl.load(q_rows + kcols[None, :])
            if QC > 0:
                qk = _rot_cols(qk, h32, kcols, D_c, HP, KC)
            k = _pool_tile(pool_c, pool_r, pool_s, tok, in_range, kcols, kc, D_c, D_r, QC, KC)
            scores = tl.dot(qk, tl.trans(k), acc = scores)
        v = _pool_tile(pool_c, pool_r, pool_s, tok, in_range, vcols, c0, D_c, D_r, QC, BD)
        m_state, l, acc = _softmax_step(scores, in_range, m_state, l, acc, v, scale)

    # Epilogue: normalize, rotate packed-pool columns back, de-rotate the rope pairs, store
    denom = tl.where(l == 0.0, 1.0, l)
    o = acc / denom[:, None]
    if QC > 0:
        # Latent columns were accumulated in the H32 domain (involutory); rope columns pass
        o = _rot_cols(o.to(tl.float16), h32, vcols, D_c, HP, BD).to(tl.float32)
    if DEROTATE:
        # GPT-J pairs (2i, 2i+1) of the rope columns at the query's absolute position;
        # latent pairs get theta 0 (identity)
        pc = c0 + 2 * tl.arange(0, BD // 2)
        is_r = pc >= D_c
        fi = tl.where(is_r, (pc - D_c) // 2, 0)
        theta = tl.load(derot_inv_freq + fi, mask = is_r, other = 0.0) * (q_pos0 + row)
        cos = tl.cos(theta)[None, :]
        sin = tl.sin(theta)[None, :]
        o_e, o_o = tl.split(tl.reshape(o, (HP, BD // 2, 2)))
        o = tl.interleave(o_e * cos - o_o * sin, o_o * cos + o_e * sin)

    if HPG > 0:
        base_h = (offs_h // HPG) * (R * HPG * D) + row * (HPG * D) + (offs_h % HPG) * D
    else:
        base_h = (row * H + offs_h) * D
    tl.store(out + base_h[:, None] + vcols[None, :], o.to(tl.float16))


# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------

def _envi(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


CFG = dict(
    hp = _envi("EXL3_ROCM_DSA_PREFILL_HP", 64),
    bd = _envi("EXL3_ROCM_DSA_PREFILL_BD", 512),
    kc = _envi("EXL3_ROCM_DSA_PREFILL_KC", 32),
    block_n = _envi("EXL3_ROCM_DSA_PREFILL_BLOCK_N", 64),
    block_w = _envi("EXL3_ROCM_DSA_PREFILL_BLOCK_W", 64),
    num_warps = _envi("EXL3_ROCM_DSA_PREFILL_WARPS", 16),
    kstages = _envi("EXL3_ROCM_DSA_PREFILL_KSTAGES", 1),
)


def eligible(kw, cfg = None):
    """(HP, BD) for a _dsa_attn_kernel constexpr set, or None to keep the upstream kernel."""
    cfg = cfg or CFG
    if kw.get("Q_SPLIT", 0) or kw.get("OUT_LATENT", 0):
        return None
    H, D_c, D_r = kw["H"], kw["D_c"], kw["D_r"]
    D = D_c + D_r
    if D_r <= 0 or D & (D - 1) or (kw.get("QC", 0) and D_c % 32):
        return None
    hp = min(cfg["hp"], H)
    bd = min(cfg["bd"], D)
    if H % hp or hp < 16 or D % bd or bd % 32 or D % cfg["kc"]:
        return None
    return hp, bd


def _mqa_kw(kw, t, cfg):
    hp, bd = t
    kw = dict(kw)
    kw.update(HP = hp, BD = bd, KC = cfg["kc"], KSTAGES = cfg["kstages"],
              BLOCK_N = cfg["block_n"], BLOCK_W = cfg["block_w"],
              num_warps = cfg["num_warps"], num_stages = 1)
    return kw


def launch(args, kw, cfg = None):
    """Launch the MQA prefill kernel for an upstream _dsa_attn_kernel call (args / kw exactly
    as dsa_attn passes them); returns the compiled kernel, or None if not eligible."""
    cfg = cfg or CFG
    t = eligible(kw, cfg)
    if t is None:
        return None
    hp, bd = t
    kw = _mqa_kw(kw, t, cfg)
    R = args[0].shape[0]
    D = kw["D_c"] + kw["D_r"]
    grid = (R * (kw["H"] // hp) * (D // bd),)
    return _dsa_prefill_mqa_kernel[grid](*args, **kw)


# ----------------------------------------------------------------------------------------
# Wiring (called from rocm_py.apply(); EXL3_ROCM_DSA_PREFILL=0 skips it)
# ----------------------------------------------------------------------------------------

class _KernelProxy:
    """Stands in for dsa_triton._dsa_attn_kernel (dsa_attn looks it up as a module global at
    call time): eligible launches run the MQA prefill kernel with its own grid, everything
    else the upstream kernel."""

    def __init__(self, orig):
        self.orig = orig
        self.__name__ = orig.__name__

    def __getitem__(self, grid):
        def _launch(*args, **kw):
            ck = launch(args, kw)
            if ck is None:
                return self.orig[grid](*args, **kw)
            return ck
        return _launch

    def run(self, *args, grid, warmup, **kw):
        """v1.5.3: dsa_attn picks (BLOCK_H, BLOCK_N, stages) from a ladder probed by a
        compile-only kernel.run(warmup = True) (attention_fn/smem.py). Report the kernel that
        will launch: for eligible calls the MQA prefill kernel, which takes neither BLOCK_H nor
        the ladder's BLOCK_N / stages, so the stock candidate fits and the launch is unchanged
        from v1.5.0 (without this, the probe would compile the upstream kernel -- the one this
        proxy exists to avoid -- and could step the ladder or raise NoFittingConfig on its
        footprint)."""
        t = eligible(kw)
        if t is None:
            return self.orig.run(*args, grid = grid, warmup = warmup, **kw)
        return _dsa_prefill_mqa_kernel.run(*args, grid = grid, warmup = warmup, **_mqa_kw(kw, t, CFG))

    def __getattr__(self, name):
        return getattr(self.orig, name)


def install():
    from ..modules.attention_fn import dsa_triton as _dt
    if not isinstance(_dt._dsa_attn_kernel, _KernelProxy):
        _dt._dsa_attn_kernel = _KernelProxy(_dt._dsa_attn_kernel)
    return (f"DSA prefill MQA kernel (HP {CFG['hp']}, BD {CFG['bd']}, KC {CFG['kc']}, "
            f"BN {CFG['block_n']}/{CFG['block_w']}, warps {CFG['num_warps']})")
