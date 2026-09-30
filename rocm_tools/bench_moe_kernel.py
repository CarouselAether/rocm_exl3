#!/usr/bin/env python3
"""Time ext.exl3_moe alone on synthetic expert weights at a real model's shapes.

    python rocm_tools/bench_moe_kernel.py [-t 256,512,1792] [--shape ds4] [-r 10] [--check]

Why synthetic: bench_moe.py times BlockSparseMLP.forward on the loaded model, which needs
the full checkpoint (77 GB for DS4) and dilutes the kernel with routing, shared experts and
the gather. This builds the expert tables directly -- random trellis bits, suh/svh scaled so
activations stay finite -- and calls the fused kernel exactly as block_sparse_mlp.run_fused
does (FUSED_DET scratch path, num_active known, fused_rows = 128), so only exl3_moe_kernel
is timed. Kernel speed does not depend on the weight values.

The expert stream is ~1.6 GB for DS4 (256 experts x 3 x 2 MB at 2 bits), far past the
32 MiB Infinity Cache, so these are DRAM numbers. Effective GB/s counts the weight bytes of
every expert that received at least one row (what the kernel must stream).

--check runs every shape with EXL3_ROCM_MOE_PIPE=0 (old mainloop) and =1 (new) in the same
process and compares outputs bitwise (the switch is read per call). Half-integer --bits (2.5 etc.)
compare the old half_k mainloop against the half-rate pipelined instances (EXL3_ROCM_HALF_MOE_PIPE);
bit-identical only on the same grid (EXL3_ROCM_MOE_BPS=1). Needs a build with the
switch; on an older build both runs take the same path and the check trivially passes.

Routing is uniform random top-k per token (seeded), so rows per expert ~ T*k/E.
"""

import argparse, os, sys, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3.ext import exllamav3_ext as ext

SHAPES = {
    # name: hidden, intermediate, experts, top_k, bits, mul1
    "ds4":  (4096, 2048, 256, 6, 2, True),
    "mimo": (4096, 2048, 256, 8, 2.5, True),
    "qwen": (2048, 512, 512, 10, 4, True),
}


def build(hidden, inter, E, bits, mul1, dev, seed = 0):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    def trellis(k, n):
        # (E, k/16, n/16, 16*bits) uint16, random bits
        t = torch.randint(0, 65536, (E, k // 16, n // 16, int(16 * bits)), generator = g, dtype = torch.int32)
        return t.to(torch.int16).to(dev)
    def sv(n, scale):
        s = (torch.randint(0, 2, (E, n), generator = g) * 2 - 1).half() * scale
        return s.to(dev)
    w = {}
    w["gt"], w["ut"], w["dt"] = trellis(hidden, inter), trellis(hidden, inter), trellis(inter, hidden)
    w["gsu"], w["usu"], w["dsu"] = sv(hidden, 1.0), sv(hidden, 1.0), sv(inter, 1.0)
    # Keep intermediates O(1): one GEMM over k ~N(0,1) terms grows ~sqrt(k)
    w["gsv"], w["usv"], w["dsv"] = sv(inter, 1.0 / hidden ** 0.5), sv(inter, 1.0 / hidden ** 0.5), sv(hidden, 1.0 / inter ** 0.5)
    def ptrs(t):
        return torch.tensor([t[e].data_ptr() for e in range(E)], dtype = torch.long, device = dev)
    for k in list(w.keys()):
        w["p_" + k] = ptrs(w[k])
    return w


def routing(T, E, k, dev, seed):
    g = torch.Generator(device = "cpu").manual_seed(seed)
    sel = torch.stack([torch.randperm(E, generator = g)[:k] for _ in range(T)]).to(dev)   # (T, k)
    wts = torch.rand((T, k), generator = g).half().to(dev)
    flat = sel.flatten()
    order = torch.argsort(flat, stable = True)
    token_sorted = (order // k).long()
    weight_sorted = wts.flatten()[order].contiguous()
    counts = torch.bincount(flat, minlength = E + 1)
    return sel, counts, token_sorted, weight_sorted


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--tokens", default = "256,512,1792")
    ap.add_argument("--shape", default = "ds4", choices = list(SHAPES))
    ap.add_argument("-r", "--repeats", type = int, default = 10)
    ap.add_argument("--rows", type = int, default = 128, help = "fused_rows (max tokens per expert)")
    ap.add_argument("--check", action = "store_true")
    ap.add_argument("--pipe", default = None, help = "set EXL3_ROCM_MOE_PIPE for the timed runs")
    ap.add_argument("--bits", type = float, default = 0, help = "override the shape's bitrate (1..8, or 1.5 / 2.5 / 3.5 with mul1)")
    ap.add_argument("--mcg", action = "store_true", help = "mcg codebook (cb1) instead of mul1")
    args = ap.parse_args()
    if args.pipe is not None:
        os.environ["EXL3_ROCM_MOE_PIPE"] = args.pipe

    dev = torch.device("cuda:0")
    hidden, inter, E, topk, bits, mul1 = SHAPES[args.shape]
    if args.bits: bits = int(args.bits) if float(args.bits).is_integer() else args.bits
    if args.mcg: mul1 = False
    w = build(hidden, inter, E, bits, mul1, dev)
    C = ext.exl3_moe_max_concurrency(0)
    R = args.rows
    tsg = torch.empty((C, R, hidden), dtype = torch.half, device = dev)
    tsu = torch.empty_like(tsg)
    tig = torch.empty((C, R, inter), dtype = torch.half, device = dev)
    tiu = torch.empty_like(tig)
    mat_bytes = int(hidden * inter * bits // 8)
    print(f" shape={args.shape} hidden={hidden} inter={inter} E={E} top{topk} K={bits} "
          f"concurrency={C} fused_rows={R}  expert bytes={3 * mat_bytes / 2**20:.2f} MiB")

    def run(T, rt):
        sel, counts, token_sorted, weight_sorted = rt
        x = rt_x[T]
        out = torch.zeros((T, hidden), dtype = torch.float, device = dev)
        cl = counts.tolist()
        fused = [c for c in cl[:E] if 0 < c <= R]
        n_act = len(fused)
        base = [0] * (E + 1); n = 0
        for e in range(E):
            if 0 < cl[e] <= R: base[e] = n; n += cl[e]
        fb = torch.tensor(base, dtype = torch.long, device = dev)
        scratch = torch.zeros((max(n, 1), hidden), dtype = torch.float, device = dev)
        def call():
            ext.exl3_moe(x, out, counts, token_sorted, weight_sorted, tsg, tsu, tig, tiu,
                         0, bits, bits, bits,
                         w["p_gt"], w["p_gsu"], w["p_gsv"], w["p_ut"], w["p_usu"], w["p_usv"],
                         w["p_dt"], w["p_dsu"], w["p_dsv"],
                         not mul1, mul1, not mul1, mul1, not mul1, mul1,
                         0.0, n_act, scratch, fb, 1, R, 16)
        return call, scratch, n_act

    toks = [int(t) for t in args.tokens.split(",")]
    rt_x = {}
    for T in toks:
        g = torch.Generator(device = "cpu").manual_seed(100 + T)
        rt_x[T] = torch.randn((T, hidden), generator = g).half().to(dev)

    for T in toks:
        rt = routing(T, E, topk, dev, seed = T)
        call, scratch, n_act = run(T, rt)
        rows = rt[1][:E]
        if args.check:
            res = {}
            for p in ("0", "1"):
                os.environ["EXL3_ROCM_MOE_PIPE"] = p
                scratch.zero_(); call(); torch.cuda.synchronize()
                res[p] = scratch.clone()
            a, b = res["0"], res["1"]
            same = torch.equal(a, b)
            fin = bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all())
            d = (a - b).abs()
            rel = float(d.max() / a.abs().max().clamp_min(1e-9))
            print(f"   check T={T:<5} bitwise={'YES' if same else 'NO '} finite={fin} "
                  f"max_abs={float(d.max()):.3e} max_rel={rel:.3e} rms_ref={float(a.pow(2).mean().sqrt()):.3e}")
            if args.pipe is not None:
                os.environ["EXL3_ROCM_MOE_PIPE"] = args.pipe
            else:
                os.environ.pop("EXL3_ROCM_MOE_PIPE", None)
        # warmup
        for _ in range(2): call()
        torch.cuda.synchronize()
        ts = []
        for _ in range(args.repeats):
            e0 = torch.cuda.Event(enable_timing = True); e1 = torch.cuda.Event(enable_timing = True)
            e0.record(); call(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
        med = statistics.median(ts)
        touched = int(((rows > 0) & (rows <= R)).sum())
        gbs = touched * 3 * mat_bytes / (med * 1e-3) / 1e9
        spread = (max(ts) - min(ts)) / med
        print(f"   T={T:<5} active={n_act:<4} rows/expert~{T * topk / E:5.1f} max={int(rows.max())}  "
              f"{med:8.3f} ms  {gbs:6.1f} GB/s  spread={spread:5.1%}")


if __name__ == "__main__":
    main()
