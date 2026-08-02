#!/usr/bin/env python3
"""Quantify the Blackwell tile misdetection on gfx1151 prefill.

triton_paged.py:1779 does:

    blackwell = torch.cuda.get_device_capability(q.device)[0] >= 10
    cfg = (128, 32, 8, 2) if blackwell else (128, 64, 8, 2)   # head_dim <= 128

On ROCm, get_device_capability() returns the gfx major, so gfx1151 reports
(11, 5) -> 11 >= 10 -> True, and prefill silently runs the narrow-kv Blackwell
tile (block_n=32) that upstream measured for an RTX 5090.

Prompt processing is compute-bound on Strix Halo, so this is the path where tile
geometry actually costs throughput -- unlike decode, which is bandwidth-bound.

Prints achieved TFLOP/s for both configs so the fix can be justified with a
number rather than an argument.
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


def timed(fn, warmup=3, iters=10):
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
    args = ap.parse_args()
    tp = build(os.path.abspath(args.repo))

    dev = torch.device("cuda:0")
    print(f"device     : {torch.cuda.get_device_properties(0).name} "
          f"({torch.cuda.get_device_properties(0).gcnArchName})")
    print(f"capability : {torch.cuda.get_device_capability(0)}  "
          f"-> upstream picks block_n={'32 (Blackwell)' if torch.cuda.get_device_capability(0)[0] >= 10 else '64'}")
    print()

    page = 256
    head_dim, n_q, n_kv = 128, 32, 8
    print(f"{'q_len':>7} {'ctx':>7} | {'block_n=32 (current)':>22} | {'block_n=64 (intended)':>22} | {'gain':>7}")
    print("-" * 78)

    for q_len, ctx in [(512, 0), (1024, 0), (2048, 0), (4096, 0), (2048, 2048)]:
        total = q_len + ctx
        nblk = (total + page - 1) // page
        g = torch.Generator(device=dev).manual_seed(0)
        rnd = lambda *s: (torch.rand(*s, generator=g, device=dev, dtype=torch.float16) - .5) * 2

        q = rnd(1, q_len, n_q, head_dim)
        k = rnd(1, q_len, n_kv, head_dim)
        v = rnd(1, q_len, n_kv, head_dim)
        kc = rnd(nblk + 2, page, n_kv, head_dim)
        vc = rnd(nblk + 2, page, n_kv, head_dim)
        bt = torch.arange(nblk, device=dev, dtype=torch.int32).unsqueeze(0).contiguous()
        cs = torch.full((1,), ctx, device=dev, dtype=torch.int32)

        def run(bn):
            return lambda: tp.paged_attn_triton_prefill(
                q=q, k=k, v=v, k_cache=kc.clone(), v_cache=vc.clone(),
                block_table=bt, cache_seqlens=cs, causal=True,
                softmax_scale=head_dim ** -0.5, block_n=bn,
            )

        try:
            t32 = timed(run(32))
            t64 = timed(run(64))
        except Exception as e:
            print(f"{q_len:>7} {ctx:>7} | error: {type(e).__name__}: {str(e)[:40]}")
            continue

        # causal attention FLOPs: 2 * 2 * q_len * avg_kv * heads * dim
        avg_kv = ctx + q_len / 2
        flops = 4.0 * q_len * avg_kv * n_q * head_dim
        tf32, tf64 = flops / t32 / 1e12, flops / t64 / 1e12
        gain = (t32 / t64 - 1.0) * 100
        print(f"{q_len:>7} {ctx:>7} | {t32*1e3:8.3f} ms {tf32:7.2f} TF/s | "
              f"{t64*1e3:8.3f} ms {tf64:7.2f} TF/s | {gain:+6.1f}%")

    print()
    print("gain = speedup from using the intended non-Blackwell tile (block_n=64).")
    print("Positive means the capability misdetection is costing prefill throughput.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
