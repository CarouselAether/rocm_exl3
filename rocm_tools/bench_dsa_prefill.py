#!/usr/bin/env python3
"""DeepSeek-V4 prefill sparse attention (one-shot _dsa_attn_kernel): microbenchmark and
correctness.

Calls dsa_triton.dsa_attn exactly as dsv4._forward_cached does for a prefill chunk (single
block-table row, sinks, eq. 26 de-rotation, group-major output with 8 groups, packed-pool
staging when -cq), on synthetic caches:

  H 64 query heads, 1 KV head (MQA), D 512 = D_c 448 latent + D_r 64 rope, window 128.
    csa  compress 4,   epp 64: dense pool while ec <= 512, else top-k 512 index lists
    hca  compress 128, epp 2 : dense pool

Chunk of R new rows at absolute position ctx (ctx = 0: first chunk; > 0: the prior window
rows come from the ring, the pool holds the context).

Variants:
  old  upstream dsa_triton._dsa_attn_kernel (default tuning: BLOCK_H 32, N 32, 4 warps, 3 stages)
  new  rocm_py.dsa_prefill_rdna._dsa_prefill_mqa_kernel (CFG, env overrides)

Timing: CUDA events around the whole dsa_attn call (so packed-pool staging counts), best of
reps. Correctness: fp64 torch reference on a subset of rows, max abs error relative to
max |ref|. Resources from the compiled kernel's AMDGCN metadata.

    python rocm_tools/bench_dsa_prefill.py                   # before/after table
    python rocm_tools/bench_dsa_prefill.py --quick
    python rocm_tools/bench_dsa_prefill.py --sweep-old       # old-kernel params (compile + time)
    python rocm_tools/bench_dsa_prefill.py --sweep-new       # new-kernel params
"""

import argparse
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn import dsa_triton as dt
from exllamav3.rocm_py import dsa_prefill_rdna as dp

H, D_C, D_R, WINDOW, TOPK, G_OUT = 64, 448, 64, 128, 512, 8
D = D_C + D_R
DEV = torch.device("cuda:0")


class Case:
    def __init__(self, kind, ctx, R, qc = 0, seed = 0):
        g = torch.Generator(device = "cpu").manual_seed(seed)
        self.kind, self.ctx, self.R, self.qc = kind, ctx, R, qc
        self.m = {"csa": 4, "hca": 128}[kind]
        self.epp = 256 // self.m
        self.pos0 = ctx
        self.win_beg = max(ctx - 200, 0)
        self.floor = ctx - min(WINDOW - 1, ctx - self.win_beg, ctx)
        self.ec = (ctx + R) // self.m
        self.topk = kind == "csa" and self.ec > TOPK

        def rn(*shape, s = 1.0):
            return (torch.randn(shape, generator = g) * s).half().to(DEV)

        self.q = rn(R, H, D)
        self.ring = rn(max(ctx - self.win_beg, 1), D)
        self.kv = rn(R, D)
        self.sinks = (torch.randn(H, generator = g) * 2.0 + 4.0).to(DEV)
        self.inv_freq_neg = (-1.0 / (10000.0 ** (torch.arange(0, D_R, 2).float() / D_R))).to(DEV)

        n_ent = max(self.ec, 1)
        pages = -(-n_ent // self.epp) + 1
        self.num_pages = pages + 3
        rows = self.num_pages * self.epp
        pool_c = rn(rows, D_C)
        self.pool_r = rn(rows, D_R)
        perm = torch.randperm(self.num_pages, generator = g).int()
        self.bt = perm[:pages].view(1, pages).contiguous().to(DEV)
        if qc:
            Gc = D_C // 32
            pq = torch.empty((rows, Gc * qc), dtype = torch.int32, device = DEV)
            ps = torch.empty((rows, Gc), dtype = torch.half, device = DEV)
            ext.quant_cache_cont(pool_c, pq, ps, 0.0)
            deq = torch.empty((rows, D_C), dtype = torch.half, device = DEV)
            ext.dequant_cache_cont(pq, ps, deq, 0.0)
            self.pool_c_ref = deq
            self.pool_c_arg = pq.view(self.num_pages, self.epp, Gc * qc)
            self.qc_arg = (ps, qc)
        else:
            self.pool_c_ref = pool_c
            self.pool_c_arg = pool_c.view(self.num_pages, self.epp, D_C)
            self.qc_arg = None
        self.indices, self.k_len = None, 0
        if self.topk:
            # Per-row causal top-k: random scores, -inf past the row's entry bound
            ar = torch.arange(self.ec, device = DEV)
            bound = (self.pos0 + torch.arange(R, device = DEV) + 1) // self.m
            sc = torch.rand((R, self.ec), generator = torch.Generator(device = DEV).manual_seed(seed),
                            device = DEV)
            sc = sc.masked_fill(ar[None, :] >= bound[:, None], -1.0)
            v, i = sc.topk(TOPK, dim = 1)
            self.indices = torch.where(v >= 0, i, -1).int().contiguous()
            self.k_len = TOPK
        self.out = torch.empty((G_OUT, R, (H // G_OUT) * D), dtype = torch.half, device = DEV)

    def tag(self):
        return f"{self.kind}{'-topk' if self.topk else ''} R{self.R} ctx{self.ctx} qc{self.qc}"

    def run(self):
        return dt.dsa_attn(
            self.q, self.pool_c_arg, self.pool_r, self.bt, sinks = self.sinks,
            ring = self.ring, kv_chunk = self.kv, win_len = WINDOW,
            win_floor = self.floor, ring_beg = self.win_beg,
            indices = self.indices, k_len = self.k_len, pool_len = self.ec, q_pos0 = self.pos0,
            compress_rate = self.m, scale = D ** -0.5,
            derot_inv_freq = self.inv_freq_neg, groups = G_OUT, group_major = True,
            page_size = self.epp, qc = self.qc_arg, out = self.out)

    def ref_rows(self):
        R = self.R
        n = min(R, 40)
        return sorted(set(torch.linspace(0, R - 1, n).round().long().tolist()))

    def reference(self, rows):
        """fp64 reference for the given rows: (len(rows), H, D) (token-major)."""
        pc, pr = self.pool_c_ref.double(), self.pool_r.double()
        scale = D ** -0.5
        out = []
        for r in rows:
            qa = self.pos0 + r
            kvs = []
            for a in range(qa, qa - WINDOW, -1):
                if a < self.floor:
                    break
                kvs.append(self.kv[a - self.pos0] if a >= self.pos0 else self.ring[a - self.win_beg])
            kv = torch.stack(kvs).double()
            if self.topk:
                idx = self.indices[r]
                idx = idx[idx >= 0].long()
            else:
                idx = torch.arange(min((qa + 1) // self.m, self.ec), device = DEV)
            if idx.numel():
                tok = self.bt[0, idx // self.epp].long() * self.epp + idx % self.epp
                kv = torch.cat([kv, torch.cat([pc[tok], pr[tok]], dim = -1)], 0)
            s = (self.q[r].double() @ kv.T) * scale
            s = torch.cat([s, self.sinks.double().unsqueeze(1)], dim = 1)
            p = torch.softmax(s, dim = -1)[:, :-1]
            o = p @ kv
            th = self.inv_freq_neg.double() * qa
            c, sn = th.cos(), th.sin()
            e, od = o[:, D_C::2].clone(), o[:, D_C + 1::2].clone()
            o[:, D_C::2] = e * c - od * sn
            o[:, D_C + 1::2] = od * c + e * sn
            out.append(o)
        return torch.stack(out)


# ----------------------------------------------------------------------------------------

class Recorder:
    """Replaces dsa_triton._dsa_attn_kernel for the bench: routes to old / new, records the
    compiled kernel."""

    def __init__(self, orig):
        self.orig = orig
        self.__name__ = orig.__name__
        self.variant = "old"
        self.old_over = {}
        self.cfg = dict(dp.CFG)
        self.ck = None

    def __getitem__(self, grid):
        def _launch(*args, **kw):
            if self.variant == "new":
                ck = dp.launch(args, kw, self.cfg)
                if ck is None:
                    raise RuntimeError("new kernel declined the shape")
            else:
                kw = dict(kw)
                over = dict(self.old_over)
                if "BLOCK_H" in over:
                    bh = over["BLOCK_H"]
                    R = args[0].shape[0]
                    grid_ = (R * -(-kw["H"] // bh),)
                else:
                    grid_ = grid
                kw.update(over)
                ck = self.orig[grid_](*args, **kw)
            self.ck = ck
            return ck
        return _launch


def _orig_kernel():
    k = dt._dsa_attn_kernel
    while hasattr(k, "orig"):
        k = k.orig
    return k


REC = None


def setup():
    global REC
    REC = Recorder(_orig_kernel())
    dt._dsa_attn_kernel = REC


def kernel_resources(ck):
    if ck is None:
        return {}
    asm = ck.asm.get("amdgcn", "")
    def f(key):
        m = re.search(rf"\.{key}:\s+(\d+)", asm)
        return int(m.group(1)) if m else -1
    vgpr = f("vgpr_count")
    return dict(vgpr = vgpr, spill = f("vgpr_spill_count"), scratch = f("private_segment_fixed_size"),
                lds = f("group_segment_fixed_size"), wmma = len(re.findall(r"\bv_wmma_", asm)))


def time_call(fn, reps = 6, inner = 3):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    best = 1e30
    for _ in range(reps):
        e0.record()
        for _ in range(inner):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) / inner)
    return best * 1000.0   # us


def check(c, ref_cache):
    c.out.zero_()
    out = c.run().float()
    torch.cuda.synchronize()
    rows = c.ref_rows()
    if c.tag() not in ref_cache:
        ref_cache[c.tag()] = c.reference(rows)
    ref = ref_cache[c.tag()]
    got = out.view(G_OUT, c.R, H // G_OUT, D).permute(1, 0, 2, 3).reshape(c.R, H, D)[rows]
    return ((got.double() - ref).abs().max() / ref.abs().max()).item()


def measure(c, variant, ref_cache, do_check = True, cfg = None, old_over = None):
    REC.variant = variant
    REC.cfg = dict(dp.CFG) | (cfg or {})
    REC.old_over = old_over or {}
    REC.ck = None
    err = check(c, ref_cache) if do_check else None
    us = time_call(c.run)
    return dict(err = err, us = us, **kernel_resources(REC.ck))


def fmt(r):
    e = f"{r['err']:.1e}" if r.get("err") is not None else "   -   "
    return (f"{r['us'] / 1000:8.3f} ms  err {e}  vgpr {r.get('vgpr', -1):3d} spill {r.get('spill', -1):5d} "
            f"scr {r.get('scratch', -1):5d} lds {r.get('lds', -1):5d} wmma {r.get('wmma', -1):3d}")


STD = [
    # (kind, ctx, R, qc)
    ("csa", 0, 512, 0), ("csa", 0, 2048, 0), ("hca", 0, 2048, 0),
    ("csa", 3072, 64, 0), ("csa", 3072, 255, 0), ("csa", 3072, 1792, 0),
    ("hca", 3072, 255, 0),
    ("csa", 16384, 64, 0), ("csa", 16384, 255, 0), ("csa", 16384, 2048, 0),
    ("hca", 16384, 64, 0), ("hca", 16384, 2048, 0),
    ("csa", 3072, 255, 4), ("csa", 16384, 2048, 4), ("hca", 16384, 255, 4),
    ("csa", 3072, 40, 4),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action = "store_true")
    ap.add_argument("--only", default = "")
    ap.add_argument("--sweep-old", action = "store_true")
    ap.add_argument("--sweep-new", action = "store_true")
    ap.add_argument("--compile-only", action = "store_true", help = "sweeps: one tiny case, resources only")
    ap.add_argument("--nocheck", action = "store_true")
    args = ap.parse_args()
    torch.cuda.set_device(DEV)
    setup()
    if args.sweep_old:
        return sweep_old(args)
    if args.sweep_new:
        return sweep_new(args)
    cases = STD[:4] if args.quick else STD
    refs = {}
    tot = {"old": 0.0, "new": 0.0}
    for cd in cases:
        c = Case(*cd)
        for v in ("old", "new"):
            if args.only and v != args.only:
                continue
            r = measure(c, v, refs, do_check = not args.nocheck)
            tot[v] += r["us"]
            print(f"  {v} {c.tag():<28} {fmt(r)}", flush = True)
        del c
        torch.cuda.empty_cache()


def sweep_old(args):
    cs = [Case("csa", 0, 2048), Case("csa", 16384, 255)] if not args.compile_only else [Case("csa", 0, 64)]
    refs = {}
    for bh, bn, nw, st in itertools.product([8, 16, 32], [16, 32], [4, 8], [1, 2, 3]):
        over = dict(BLOCK_H = bh, BLOCK_N = bn, num_warps = nw, num_stages = st)
        cells = []
        res = {}
        try:
            for c in cs:
                r = measure(c, "old", refs, do_check = c is cs[0], old_over = over)
                res = res or r
                cells.append(f"{r['us'] / 1000:7.3f}")
        except Exception as e:
            cells.append(f"fail {type(e).__name__}: {str(e)[:80]}")
        print(f"  H{bh:<2} N{bn:<2} w{nw} s{st}  {' | '.join(cells)}  "
              f"vgpr {res.get('vgpr', -1)} spill {res.get('spill', -1)} scr {res.get('scratch', -1)} "
              f"lds {res.get('lds', -1)} err {res.get('err') or 0:.1e}", flush = True)


def sweep_new(args):
    stage = os.environ.get("EXL3_DSA_SWEEP", "tile")
    if args.compile_only:
        cs = [Case("csa", 0, 64)]
    else:
        cs = [Case("csa", 0, 2048), Case("csa", 16384, 255), Case("csa", 16384, 64), Case("hca", 16384, 2048)]
    refs = {}
    if stage == "tile":
        grid = [dict(hp = hp, bd = bd, kc = kc, block_n = bn, block_w = bn, num_warps = nw)
                for hp, bd, kc, bn, nw in itertools.product(
                    (16, 32, 64), (128, 256, 512), (32, 64, 128), (16, 32, 64), (4, 8))]
    else:
        grid = [eval(s) for s in os.environ["EXL3_DSA_SWEEP_LIST"].split(";")]
    for cfg in grid:
        cells, res = [], {}
        try:
            for c in cs:
                r = measure(c, "new", refs, do_check = c is cs[0], cfg = cfg)
                res = res or r
                cells.append(f"{r['us'] / 1000:7.3f}")
        except Exception as e:
            cells.append(f"fail {type(e).__name__}: {str(e)[:80]}")
        desc = " ".join(f"{k}={v}" for k, v in cfg.items())
        print(f"  {desc:<62} {' | '.join(cells)}  vgpr {res.get('vgpr', -1)} spill {res.get('spill', -1)} "
              f"scr {res.get('scratch', -1)} lds {res.get('lds', -1)} err {res.get('err') or 0:.1e}", flush = True)


if __name__ == "__main__":
    main()
