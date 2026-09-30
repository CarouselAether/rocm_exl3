"""MQA-specialized DSA decode split kernel for RDNA (gfx11), ROCm-owned.

Drop-in replacement for dsa_triton._dsa_attn_split_kernel on the decode path of
DeepSeek-V4 (single latent KV head, 64 query heads, D = 448 latent + 64 rope = 512):
same argument list, same workspace layout, same combine kernel. Only the work split
differs.

Why: the upstream split kernel gives every program BLOCK_H heads and the WHOLE output
width, so its fp32 accumulator is BLOCK_H x (D_c_pad 512 + D_r 64) -- at BLOCK_H 8 over
256 lanes that plus the q tile and the double-oriented K/V tile sits at the 256-VGPR
ceiling with ~700 VGPRs spilled (2.2 KB scratch per lane). Each decode split covers only
~8-40 keys, so the call is all fixed cost: scratch round-trips, 128 programs each
re-reading the same K rows. Measured 160-225 us per call on gfx1151 regardless of
context.

Here one program owns HP heads (the MMA M dimension; the single KV head is shared by
all 64) and one BD-wide column block of the output, over a key split:

  scores = sum_kc q[HP, kc:kc+KC] . K[BN, kc:kc+KC]^T    full-width score, recomputed
                                          per column block (decode key counts are tiny)
  acc   += p[HP, BN] . V[BN, BD]          V = K, only this program's columns

The score reduction over D is a RUNTIME loop of KC-wide chunks, q re-read per chunk
(L1/L2 hits). That is the fix that matters: a loop-invariant q tile gets hoisted and
held as the WMMA A operand, which RDNA3 replicates across the two half-waves -- a
16 x 512 panel per warp is 256 VGPRs on its own, i.e. the upstream kernel's spills.
Production tiling (sweep in RDNA_NOTES): HP 32, BD 256, BLOCK_N/W 32, KC 64, 4 warps,
8 splits -> accumulator 32 x 256 fp32 = 64 VGPRs over 128 lanes, 0 spills (fp16 pool).
The virtual key row is [c (D_c) | r (D_r)] = D wide (a power of two), loaded from the
ring/chunk as one contiguous row or from the paged pool as pool_c ++ pool_r.

Grid contract (unchanged C++ launch, dsv4_attn.cpp): (rows * H / BLOCK_H, n_splits).
BLOCK_H is the COMBINE's head block and defines the workspace layout; this kernel reads
program id % (H / BLOCK_H) as (head group, column block) with
(H / HP) * (D / BD) == H / BLOCK_H. m / l are identical in every column block of a head
group (same scores, same instructions); column block 0 writes them.

Packed pools (QC > 0): same H32-rotated domain as upstream -- q and window tiles are
rotated per 32-group on the latent columns, partials stay rotated, the combine rotates
back. The packed loader here takes an arbitrary 32-group column range.

Not covered (callers fall back to the upstream kernel): Q_SPLIT, OUT_LATENT (GLM-5.2),
non-power-of-two D, H not a multiple of the head/column tiling.
"""

import os

import triton
import triton.language as tl

from ..modules.attention_fn.triton_paged import _rot_h32


@triton.jit
def _qc_plane_cols(qw, row_words, mask_n, g0, n_g, pbase,
                   W: tl.constexpr, BITS: tl.constexpr, NG: tl.constexpr):
    """One bit plane of 32-groups [g0, g0 + NG) of packed rows, (BN, NG * 32) int32;
    groups at or past n_g (relative) read as zero."""
    VPW: tl.constexpr = 32 // W
    garr = tl.arange(0, NG * W)
    grp = garr // W
    cols = (g0 + grp) * BITS + pbase + (garr % W)
    w = tl.load(qw + row_words[:, None] + cols[None, :],
                mask = mask_n[:, None] & (grp < n_g)[None, :], other = 0)
    nib = (w[:, :, None] >> (tl.arange(0, VPW) * W)[None, None, :]) & ((1 << W) - 1)
    return tl.reshape(nib, (w.shape[0], NG * 32))


@triton.jit
def _qc_load_cols(qwords, scales, tok, mask_n, g0, G: tl.constexpr,
                  BITS: tl.constexpr, NG: tl.constexpr):
    """(BN, NG * 32) fp16 tile of columns [g0 * 32, (g0 + NG) * 32) of a packed pool with G
    groups per row (rotated domain, same grid as triton_paged._qc_load_v); zero past G."""
    n_g = G - g0
    row_words = tok * (G * BITS)
    raw = tl.zeros((1, 1), tl.int32)
    pbase = 0
    first = True
    if BITS & 8:
        raw = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 8, BITS, NG)
        pbase += 8
        first = False
    if BITS & 4:
        p = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 4, BITS, NG)
        raw = p if first else (raw << 4) | p
        pbase += 4
        first = False
    if BITS & 2:
        p = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 2, BITS, NG)
        raw = p if first else (raw << 2) | p
        pbase += 2
        first = False
    if BITS & 1:
        p = _qc_plane_cols(qwords, row_words, mask_n, g0, n_g, pbase, 1, BITS, NG)
        raw = p if first else (raw << 1) | p
    garr = tl.arange(0, NG)
    sc = tl.load(scales + tok[:, None] * G + (g0 + garr)[None, :],
                 mask = mask_n[:, None] & (garr < n_g)[None, :], other = 0.0)
    scx = tl.reshape(tl.broadcast_to(sc[:, :, None], (sc.shape[0], NG, 32)), (sc.shape[0], NG * 32))
    mh = (1 << (BITS - 1)) - 0.5
    inv_m = 1.0 / (1 << (BITS - 1))
    return ((raw.to(tl.float32) - mh) * (scx.to(tl.float32) * inv_m)).to(tl.float16)


@triton.jit
def _rot_cols(x, h32, cols, D_c: tl.constexpr, ROWS: tl.constexpr, W: tl.constexpr):
    """H32-rotate the latent columns (cols < D_c) of an fp16 tile; rope columns pass."""
    xr = _rot_h32(x, h32, ROWS, W)
    return tl.where((cols < D_c)[None, :], xr, x)


@triton.jit
def _pool_tile(pool_c, pool_r, pool_s, tok, in_range, cols, c0,
               D_c: tl.constexpr, D_r: tl.constexpr, QC: tl.constexpr, NC: tl.constexpr):
    """(BN, NC) fp16 tile of virtual pool rows [c | r], columns c0 + [0, NC) (cols = c0 +
    arange(NC)); latent part from the fp16 or packed pool, rope part from pool_r."""
    if QC > 0:
        t = _qc_load_cols(pool_c, pool_s, tok, in_range, c0 // 32, D_c // 32, QC, NC // 32)
    else:
        t = tl.load(pool_c + tok[:, None] * D_c + cols[None, :],
                    mask = in_range[:, None] & (cols < D_c)[None, :], other = 0.0)
    t += tl.load(pool_r + tok[:, None] * D_r + (cols - D_c)[None, :],
                 mask = in_range[:, None] & (cols >= D_c)[None, :], other = 0.0)
    return t


@triton.jit
def _win_tile(kv_chunk, ring, idx_c, idx_r, mc, mr, cols, D: tl.constexpr):
    """(BW, len(cols)) fp16 tile of window rows: this step's chunk rows or ring rows."""
    return tl.load(kv_chunk + idx_c[:, None] * D + cols[None, :], mask = mc[:, None], other = 0.0) \
         + tl.load(ring + idx_r[:, None] * D + cols[None, :], mask = mr[:, None], other = 0.0)


@triton.jit
def _softmax_step(scores, in_range, m_state, l, acc, v, scale):
    scores = scores * scale
    scores = tl.where(in_range[None, :], scores, -float("inf"))
    m_new = tl.maximum(m_state, tl.max(scores, axis = 1))
    m_exp = tl.where(m_new == -float("inf"), 0.0, m_new)
    p = tl.exp(scores - m_exp[:, None])
    p = tl.where(in_range[None, :], p, 0.0)
    alpha = tl.where(m_state == -float("inf"), 0.0, tl.exp(m_state - m_exp))
    l = l * alpha + tl.sum(p, axis = 1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), v)
    return m_new, l, acc


@triton.jit(do_not_specialize = [
    "k_len", "win_len", "pool_len", "num_pages_per_row", "q_pos0",
    "win_floor", "ring_beg", "ring_stride",
])
def _dsa_decode_mqa_kernel(
    q, ring, kv_chunk, pool_c, pool_r, block_table, indices, ws_ml, ws_acc,
    k_len, win_len, pool_len, num_pages_per_row, q_pos0, win_floor, ring_beg,
    slot_ids, ring_stride, pool_s, h32,
    H: tl.constexpr,
    page_size: tl.constexpr,
    D_c: tl.constexpr,
    D_c_pad: tl.constexpr,         # unused (upstream signature)
    D_r: tl.constexpr,
    K_pad: tl.constexpr,
    compress_rate: tl.constexpr,
    scale: tl.constexpr,
    HAS_WINDOW: tl.constexpr,
    DENSE_POOL: tl.constexpr,
    BLOCK_H: tl.constexpr,         # combine head block: workspace layout + grid basis
    BLOCK_N: tl.constexpr,
    BLOCK_W: tl.constexpr,
    SEQ: tl.constexpr = 1,
    MULTIROW: tl.constexpr = 0,
    DEBUG_BOUNDS: tl.constexpr = 0,
    DEBUG_PAGES: tl.constexpr = 0,
    Q_SPLIT: tl.constexpr = 0,     # must be 0
    OUT_LATENT: tl.constexpr = 0,  # must be 0
    QC: tl.constexpr = 0,
    HP: tl.constexpr = 32,         # heads per program (MMA M)
    BD: tl.constexpr = 256,        # output columns per program
    KC: tl.constexpr = 64,         # score reduction chunk over D (runtime loop)
    KSTAGES: tl.constexpr = 1,     # software-pipeline depth of the KC loop
):
    D: tl.constexpr = D_c + D_r
    HB_C: tl.constexpr = H // BLOCK_H
    NDB: tl.constexpr = D // BD
    tl.static_assert((H // HP) * NDB == HB_C)
    tl.static_assert(Q_SPLIT == 0 and OUT_LATENT == 0)

    pid = tl.program_id(0)
    split = tl.program_id(1)
    n_splits = tl.num_programs(1)
    row = pid // HB_C
    sub = pid % HB_C
    hgrp = sub // NDB
    dblk = sub % NDB

    if MULTIROW:
        job = row // SEQ
        loc = row % SEQ
        q_pos0 = tl.load(q_pos0 + job)
        win_floor = tl.load(win_floor + job)
        ring_beg = tl.load(ring_beg + job)
        pool_len = tl.load(pool_len + job)
        k_len = tl.load(k_len + job)
        slot = tl.load(slot_ids + job)
        ring = ring + slot.to(tl.int64) * ring_stride
        bt_row = job
        cbase = job * SEQ
    else:
        loc = row
        cbase = 0
        bt_row = row

    offs_h = hgrp * HP + tl.arange(0, HP)
    c0 = dblk * BD
    vcols = c0 + tl.arange(0, BD)

    q_rows = q + (row * H + offs_h)[:, None] * D
    kcols0 = tl.arange(0, KC)

    # This row's virtual key range [window ++ pool] for this split (same split as upstream)
    if DENSE_POOL:
        n_pool = tl.minimum((q_pos0 + loc + 1) // compress_rate, pool_len)
    else:
        n_pool = k_len
    n_win = win_len if HAS_WINDOW else 0
    n_tot = n_win + n_pool
    chunk = (n_tot + n_splits - 1) // n_splits
    j0 = split * chunk
    j1 = tl.minimum(j0 + chunk, n_tot)

    m_state = tl.full((HP,), -float("inf"), tl.float32)
    l = tl.zeros((HP,), tl.float32)
    acc = tl.zeros((HP, BD), tl.float32)

    if HAS_WINDOW:
        q_abs = q_pos0 + loc
        w1 = tl.minimum(j1, n_win)
        for n0 in tl.range(j0, w1, BLOCK_W, num_stages = 1):
            offs_j = n0 + tl.arange(0, BLOCK_W)
            abs_pos = q_abs - offs_j
            in_range = (offs_j < w1) & (abs_pos >= win_floor)
            mc = in_range & (abs_pos >= q_pos0)
            mr = in_range & (abs_pos < q_pos0)
            idx_c = tl.where(mc, cbase + abs_pos - q_pos0, 0)
            idx_r = tl.where(mr, abs_pos - ring_beg, 0)
            # Score reduction streamed over D in a RUNTIME loop: a loop-invariant q would be
            # kept resident as the WMMA A operand (replicated across half-waves on RDNA3:
            # 16 x 512 per warp = 256 VGPRs), which is what spilled the upstream kernel
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

    p0 = tl.maximum(j0 - n_win, 0)
    p1 = j1 - n_win
    for n0 in tl.range(p0, p1, BLOCK_N, num_stages = 1):
        offs_n = n0 + tl.arange(0, BLOCK_N)
        if DENSE_POOL:
            idx = tl.where(offs_n < p1, offs_n, -1)
        else:
            idx = tl.load(indices + row * K_pad + offs_n, mask = offs_n < p1, other = -1)
        in_range = idx >= 0
        idx_s = tl.where(in_range, idx, 0)
        phys = tl.load(block_table + bt_row * num_pages_per_row + idx_s // page_size,
                       mask = in_range, other = 0)
        if DEBUG_BOUNDS:
            tl.device_assert(tl.where(in_range, idx_s < pool_len, True), "dsa_mqa: entry idx >= pool_len")
            tl.device_assert(tl.where(in_range, (phys >= 0) & (phys < DEBUG_PAGES), True), "dsa_mqa: pool page OOB")
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

    # Partials in the combine's layout: ((row * HB_C + h // BLOCK_H) * S + split) * BLOCK_H
    # + h % BLOCK_H
    pidc = row * HB_C + offs_h // BLOCK_H
    base = (pidc * n_splits + split) * BLOCK_H + offs_h % BLOCK_H
    tl.store(ws_ml + base * 2, m_state, mask = dblk == 0)
    tl.store(ws_ml + base * 2 + 1, l, mask = dblk == 0)
    tl.store(ws_acc + base[:, None] * D + vcols[None, :], acc)


# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------

def _envi(name, default):
    v = os.environ.get(name)
    return int(v) if v not in (None, "") else default


# Decode-path tuning (graphed BCDsa; see RDNA_NOTES "DSA decode MQA kernel"). BLOCK_H is
# the combine head block (bc_dsa.BLOCK_H: sets the workspace + grid), HP the heads per
# split program; the column block BD follows from them.
CFG = dict(
    n_splits = _envi("EXL3_ROCM_DSA_DECODE_SPLITS", 8),
    block_h = _envi("EXL3_ROCM_DSA_DECODE_BLOCK_H", 16),
    hp = _envi("EXL3_ROCM_DSA_DECODE_HP", 32),
    block_n = _envi("EXL3_ROCM_DSA_DECODE_BLOCK_N", 32),
    block_w = _envi("EXL3_ROCM_DSA_DECODE_BLOCK_W", 32),
    num_warps = _envi("EXL3_ROCM_DSA_DECODE_WARPS", 4),
    kc = _envi("EXL3_ROCM_DSA_DECODE_KC", 64),
    kstages = _envi("EXL3_ROCM_DSA_DECODE_KSTAGES", 1),
    num_stages = 1,
)


def tiling(H, D, block_h, hp):
    """(HP, BD) for a combine head block, or None when this kernel cannot tile the shape."""
    if D & (D - 1) or H % block_h or hp > H or H % hp:
        return None
    hb_c = H // block_h
    nhg = H // hp
    if hb_c % nhg:
        return None
    ndb = hb_c // nhg
    if D % ndb or (D // ndb) % 32 or D // ndb < 16:
        return None
    return hp, D // ndb


def eligible(constexprs):
    """Whether a split-kernel constexpr set (bc_dsa / dsa_attn) can run on this kernel."""
    c = constexprs
    if c.get("Q_SPLIT", 0) or c.get("OUT_LATENT", 0):
        return None
    H, D_c, D_r = c["H"], c["D_c"], c["D_r"]
    if D_r <= 0 or (c.get("QC", 0) and D_c % 32):
        return None
    t = tiling(H, D_c + D_r, c["BLOCK_H"], min(CFG["hp"], H))
    if t is None:
        # Fall back to fewer heads per program (e.g. H < configured HP)
        for hp in (32, 16):
            if hp <= H:
                t = tiling(H, D_c + D_r, c["BLOCK_H"], hp)
                if t:
                    break
    return t


def bench_config(multirow = False):
    """Runner kwargs for rocm_tools/bench_dsa_decode.py matching the production config."""
    ns = 8 if multirow else CFG["n_splits"]   # the batched path hardcodes 8 splits
    hp, bd = tiling(64, 512, CFG["block_h"], CFG["hp"])
    return dict(n_splits = ns, block_h = CFG["block_h"], block_n = CFG["block_n"],
                block_w = CFG["block_w"], num_warps = CFG["num_warps"],
                num_stages = CFG["num_stages"], extra = dict(HP = hp, BD = bd, KC = CFG["kc"], KSTAGES = CFG["kstages"]))


def sweep_configs():
    """Stage via EXL3_DSA_SWEEP: 'tile' (shape x warps x tiles x KC at 4 splits) or
    'splits' (split count around the production tile)."""
    import itertools
    stage = os.environ.get("EXL3_DSA_SWEEP", "tile")
    out = []
    if stage == "tile":
        for (block_h, hp), nw, bn, kc in itertools.product(
                [(16, 64), (8, 64), (8, 32), (16, 32), (4, 64), (4, 32), (16, 16)],
                (4, 8), (16, 32), (32, 64, 128)):
            t = tiling(64, 512, block_h, hp)
            if t is None:
                continue
            out.append(dict(n_splits = 4, block_h = block_h, block_n = bn, block_w = bn,
                            num_warps = nw, num_stages = 1,
                            extra = dict(HP = t[0], BD = t[1], KC = kc)))
    else:
        for (block_h, hp, nw, kc), ks, ns in itertools.product(
                [(16, 64, 4, 64), (8, 32, 4, 64), (16, 32, 4, 64), (16, 64, 8, 64), (16, 64, 8, 128)],
                (1, 2), (2, 4, 8, 16)):
            t = tiling(64, 512, block_h, hp)
            out.append(dict(n_splits = ns, block_h = block_h, block_n = 32, block_w = 32,
                            num_warps = nw, num_stages = 1,
                            extra = dict(HP = t[0], BD = t[1], KC = kc, KSTAGES = ks)))
    return out


# ----------------------------------------------------------------------------------------
# Wiring (called from rocm_py.apply(); EXL3_ROCM_DSA_DECODE=0 skips it)
# ----------------------------------------------------------------------------------------

class _SplitProxy:
    """Stands in for dsa_triton._dsa_attn_split_kernel on the eager path (dsa_attn looks the
    kernel up as a module global at call time): eligible launches run the MQA kernel with
    the same grid, arguments and workspace, everything else the upstream kernel."""

    def __init__(self, orig):
        self.orig = orig
        self.__name__ = orig.__name__

    @staticmethod
    def _mqa_kw(kw, t):
        kw = dict(kw)
        kw.update(HP = t[0], BD = t[1], KC = CFG["kc"], KSTAGES = CFG["kstages"],
                  BLOCK_N = CFG["block_n"], BLOCK_W = CFG["block_w"],
                  num_warps = CFG["num_warps"], num_stages = CFG["num_stages"])
        return kw

    def __getitem__(self, grid):
        def launch(*args, **kw):
            t = eligible(kw)
            if t is None:
                return self.orig[grid](*args, **kw)
            return _dsa_decode_mqa_kernel[grid](*args, **self._mqa_kw(kw, t))
        return launch

    def run(self, *args, grid, warmup, **kw):
        """v1.5.3: dsa_attn walks a (BLOCK_H, BLOCK_N, stages) ladder against the device's shared
        memory, probing each candidate with a compile-only kernel.run(warmup = True)
        (attention_fn/smem.py shared_bytes). Answer for the kernel that will actually launch:
        the MQA kernel's footprint for eligible calls (it ignores the ladder's BLOCK_N / stages,
        so the stock candidate fits and the pick is the pre-ladder BLOCK_H), the upstream
        kernel's otherwise."""
        t = eligible(kw)
        if t is None:
            return self.orig.run(*args, grid = grid, warmup = warmup, **kw)
        return _dsa_decode_mqa_kernel.run(*args, grid = grid, warmup = warmup, **self._mqa_kw(kw, t))

    def __getattr__(self, name):
        return getattr(self.orig, name)


def install():
    """Route DSA decode split kernels to the MQA kernel. Graphed path (bc_dsa BCDsa /
    BCDsaBatch): wrap _compile_kernel (rebound in every module holding a from-import of it)
    and set bc_dsa.BLOCK_H / N_SPLITS, which size the workspace and the C++ launch grid.
    Eager path (dsa_attn): proxy the module-global kernel. Returns a description."""
    from ..modules.attention_fn import dsa_triton as _dt
    from ..modules.attention_fn import bc_attn as _bca
    from ..modules.attention_fn import bc_dsa as _bcd
    from ..modules.attention_fn import bc_mla as _bcm

    prev = _bcd._compile_kernel      # upstream, or the EXL3_ROCM_DSA_TUNE wrapper

    def _compile_kernel_dsa_decode(device, fn, signature, constexprs, num_warps, num_stages):
        if isinstance(fn, _SplitProxy):
            fn = fn.orig
        if fn.__name__ == "_dsa_attn_split_kernel":
            t = eligible(constexprs)
            if t is not None:
                sig = dict(signature)
                sig.update(HP = "constexpr", BD = "constexpr", KC = "constexpr",
                           KSTAGES = "constexpr")
                c = dict(constexprs)
                c.update(HP = t[0], BD = t[1], KC = CFG["kc"], KSTAGES = CFG["kstages"],
                         BLOCK_N = CFG["block_n"], BLOCK_W = CFG["block_w"])
                return prev(device, _dsa_decode_mqa_kernel, sig, c,
                            CFG["num_warps"], CFG["num_stages"])
        return prev(device, fn, signature, constexprs, num_warps, num_stages)

    _bca._compile_kernel = _compile_kernel_dsa_decode
    _bcd._compile_kernel = _compile_kernel_dsa_decode
    _bcm._compile_kernel = _compile_kernel_dsa_decode
    _bcd.BLOCK_H = CFG["block_h"]
    _bcd.N_SPLITS = CFG["n_splits"]
    if not isinstance(_dt._dsa_attn_split_kernel, _SplitProxy):
        _dt._dsa_attn_split_kernel = _SplitProxy(_dt._dsa_attn_split_kernel)
    return (f"DSA decode MQA split kernel (splits {CFG['n_splits']}, BLOCK_H {CFG['block_h']}, "
            f"HP {CFG['hp']}, BN {CFG['block_n']}/{CFG['block_w']}, KC {CFG['kc']}, "
            f"warps {CFG['num_warps']})")
