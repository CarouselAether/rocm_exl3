#!/usr/bin/env python3
"""Check the decode split-K heuristic on RDNA.

triton_paged.py picks split-K for decode as:

    target     = 2 * multi_processor_count
    num_splits = max(1, min(target // programs, cdiv(max_k_len, 4*block_n), 128))

`multi_processor_count` reports **WGPs** on ROCm, not CUs -- gfx1151 returns 20
for a 40-CU part, because RDNA pairs CUs into work-group processors. The same
code on an NVIDIA part with comparable scheduler count would see roughly double,
so the heuristic may under-split here.

Decode is memory-bandwidth-bound, so split-K matters mainly for keeping enough
CUs busy at small batch: with bsz=1 and few kv heads, `programs` is tiny and
almost all parallelism has to come from splitting the kv dimension.

Sweeps num_splits against the auto choice so the heuristic can be checked rather
than assumed.
"""

import argparse
import os
import sys
import time

import torch


def build(repo):
    sys.path.insert(0, repo)
    import types
    for name, sub in [
        ("exllamav3", ""),
        ("exllamav3.modules", "modules"),
        ("exllamav3.modules.attention_fn", "modules/attention_fn"),
    ]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [os.path.join(repo, "exllamav3", sub)]
            sys.modules[name] = stub
    from exllamav3.modules.attention_fn import triton_paged as tp
    return tp


def timed(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    args = ap.parse_args()
    tp = build(os.path.abspath(args.repo))

    dev = torch.device("cuda:0")
    p = torch.cuda.get_device_properties(0)
    print(f"device : {p.name} ({p.gcnArchName})")
    print(f"multi_processor_count = {p.multi_processor_count}  "
          f"(WGPs on ROCm; heuristic target = {2 * p.multi_processor_count})")
    print()

    page = 256
    hd, n_q, n_kv = args.head_dim, args.heads, args.kv_heads
    splits = [None, 1, 2, 4, 8, 16, 32, 64]

    hdr = f"{'bsz':>4} {'ctx':>7} | " + " ".join(f"{('auto' if s is None else s):>7}" for s in splits)
    print(hdr)
    print("-" * len(hdr))

    for bsz, ctx in [(1, 1024), (1, 4096), (1, 16384), (1, 32768), (4, 4096), (8, 4096)]:
        nblk = (ctx + 1 + page - 1) // page
        g = torch.Generator(device=dev).manual_seed(0)
        rnd = lambda *s: (torch.rand(*s, generator=g, device=dev, dtype=torch.float16) - .5) * 2
        q = rnd(bsz, 1, n_q, hd)
        k = rnd(bsz, 1, n_kv, hd)
        v = rnd(bsz, 1, n_kv, hd)
        kc = rnd(bsz * nblk + 4, page, n_kv, hd)
        vc = rnd(bsz * nblk + 4, page, n_kv, hd)
        bt = torch.arange(bsz * nblk, device=dev, dtype=torch.int32).reshape(bsz, nblk).contiguous()
        cs = torch.full((bsz,), ctx, device=dev, dtype=torch.int32)

        row, best, best_s = [], None, None
        for s in splits:
            try:
                kw = {} if s is None else {"num_splits": s}
                f = lambda: tp.paged_attn_triton_decode(
                    q=q, k=k, v=v, k_cache=kc.clone(), v_cache=vc.clone(),
                    block_table=bt, cache_seqlens=cs, causal=True,
                    softmax_scale=hd ** -0.5, **kw)
                t = timed(f) * 1e6      # microseconds
            except Exception:
                row.append("     --")
                continue
            row.append(f"{t:7.1f}")
            if s is not None and (best is None or t < best):
                best, best_s = t, s
        auto = row[0].strip()
        mark = ""
        if best is not None and auto not in ("--", ""):
            try:
                gain = (float(auto) / best - 1.0) * 100
                if gain > 5:
                    mark = f"   <- best={best_s} ({gain:+.0f}% vs auto)"
            except ValueError:
                pass
        print(f"{bsz:>4} {ctx:>7} | " + " ".join(row) + mark)

    print()
    print("microseconds per decode step, lower is better. 'auto' is the shipped heuristic.")
    print("A flagged row means a fixed split beats auto by >5% -- i.e. the WGP-vs-CU")
    print("reporting difference is costing decode throughput at that shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
