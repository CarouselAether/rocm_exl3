#!/usr/bin/env python3
"""
Phase 2 go/no-go: does upstream exllamav3's in-tree Triton paged attention produce
correct output on gfx1151?

This is the gate for the whole ROCm overhaul. The ROCm fork's garbage-token bug was
flash-attn's *own* Triton fallback; upstream's triton_paged.py is a different
implementation, so the bug does not automatically carry over -- but it is unproven on
RDNA 3.5 and everything downstream assumes it works.

Deliberately does NOT import the compiled exllamav3_ext: this tests attention only, so
it runs before any kernel porting work.

Usage:
    rocm_tools/attn_check.py [--repo PATH]
"""

import argparse
import math
import os
import sys

import torch


def build_reference(q, k, v, k_cache, v_cache, block_table, cache_seqlens, sm_scale, causal):
    """Independent fp32 reference. Gathers the paged cache, appends new K/V, full softmax.

    Intentionally naive -- this is the oracle, so clarity beats speed. Runs in fp32 to
    keep reference error well below the fp16 tolerance we compare against.
    """
    bsz, q_len, n_q_heads, dim = q.shape
    page_size = k_cache.shape[1]
    n_kv_heads = k_cache.shape[2]
    group = n_q_heads // n_kv_heads

    outputs = []
    for b in range(bsz):
        ctx = int(cache_seqlens[b].item())
        total = ctx + q_len

        n_blocks = (total + page_size - 1) // page_size
        phys = block_table[b, :n_blocks]
        k_buf = k_cache[phys].reshape(-1, n_kv_heads, dim).float().clone()
        v_buf = v_cache[phys].reshape(-1, n_kv_heads, dim).float().clone()
        # New tokens occupy [ctx, total)
        k_buf[ctx:total] = k[b].float()
        v_buf[ctx:total] = v[b].float()
        k_buf = k_buf[:total]
        v_buf = v_buf[:total]

        out_b = torch.empty(q_len, n_q_heads, dim, dtype=torch.float32, device=q.device)
        for h in range(n_q_heads):
            kv_h = h // group
            qh = q[b, :, h, :].float()                    # (q_len, dim)
            kh = k_buf[:, kv_h, :]                        # (total, dim)
            vh = v_buf[:, kv_h, :]
            scores = (qh @ kh.T) * sm_scale                # (q_len, total)
            if causal:
                # Lower-right alignment: query i sits at absolute position ctx + i
                pos = torch.arange(q_len, device=q.device).unsqueeze(1) + ctx
                key = torch.arange(total, device=q.device).unsqueeze(0)
                scores = scores.masked_fill(key > pos, float("-inf"))
            out_b[:, h, :] = torch.softmax(scores, dim=-1) @ vh
        outputs.append(out_b)

    return torch.stack(outputs)


def make_case(bsz, q_len, ctx_len, n_q_heads, n_kv_heads, dim, device, page_size=256, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *s: (torch.rand(*s, generator=g, device=device, dtype=torch.float16) - 0.5) * 2

    total_max = ctx_len + q_len
    blocks_per_seq = (total_max + page_size - 1) // page_size
    num_blocks = bsz * blocks_per_seq + 4

    k_cache = rand(num_blocks, page_size, n_kv_heads, dim)
    v_cache = rand(num_blocks, page_size, n_kv_heads, dim)
    # Non-identity block mapping: a contiguous table would hide page-indexing bugs
    block_table = torch.arange(
        bsz * blocks_per_seq, device=device, dtype=torch.int32
    ).flip(0).reshape(bsz, blocks_per_seq).contiguous()

    return dict(
        q=rand(bsz, q_len, n_q_heads, dim),
        k=rand(bsz, q_len, n_kv_heads, dim),
        v=rand(bsz, q_len, n_kv_heads, dim),
        k_cache=k_cache,
        v_cache=v_cache,
        block_table=block_table,
        cache_seqlens=torch.full((bsz,), ctx_len, device=device, dtype=torch.int32),
        sm_scale=dim ** -0.5,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--tol", type=float, default=2e-2, help="max abs diff vs fp32 reference")
    args_cli = ap.parse_args()

    repo = os.path.abspath(args_cli.repo)
    sys.path.insert(0, repo)

    # Importing exllamav3.* normally runs exllamav3/__init__.py, which JIT-builds the whole
    # C++ extension -- and that build currently fails on ROCm. Attention is pure Python +
    # Triton, so register stub parent packages carrying only __path__: relative imports
    # (`from .common import ...`) still resolve, but no package __init__ ever executes.
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

    if not torch.cuda.is_available():
        print("FAIL: no GPU visible to torch")
        return 1

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    cap = torch.cuda.get_device_capability(0)
    print(f"device      : {props.name}")
    print(f"gcnArchName : {getattr(props, 'gcnArchName', '?')}")
    print(f"torch       : {torch.__version__}  hip={torch.version.hip}")
    print(f"capability  : {cap}")

    # triton_paged.py:1779 keys the Blackwell tile config off capability[0] >= 10.
    # On ROCm this reports the gfx major, so gfx11xx trips it and silently selects
    # narrow-kv Blackwell tiles.
    if cap[0] >= 10:
        print(f"  !! WARNING: capability major {cap[0]} >= 10 -> upstream selects Blackwell "
              f"tile configs (triton_paged.py:1779). This is the misdetection in PLAN.md §0.")
    print()

    from exllamav3.modules.attention_fn.common import AttnArgs
    from exllamav3.modules.attention_fn import triton_paged as tp

    print(f"has_triton  : {tp.has_triton}")
    if not tp.has_triton:
        print("FAIL: triton unavailable -- upstream would fall back to SDPA, not a valid Phase 2 result")
        return 1
    print()

    cases = [
        # name,                bsz q_len ctx   qh  kvh dim
        ("decode mha",           1,   1,  512,  8,  8, 128),
        ("decode gqa",           1,   1,  512, 32,  8, 128),
        ("decode gqa batched",   4,   1, 1000, 32,  8, 128),
        ("decode multi-token",   1,   4,  777, 32,  8, 128),
        ("prefill short",        1,  64,    0, 32,  8, 128),
        ("prefill w/ context",   1, 128,  512, 32,  8, 128),
        ("prefill long",         1, 512,  256, 32,  8, 128),
        ("head_dim 64",          1,   1,  512, 32,  8,  64),
        ("head_dim 256",         1,   1,  384,  8,  8, 256),
    ]

    backends = [
        ("decode", tp.fn_triton_paged_attn_decode),
        ("prefill", tp.fn_triton_paged_attn_prefill),
        ("paged", tp.fn_triton_paged_attn),
        ("longq", tp.fn_triton_paged_attn_longq),
    ]

    failures, ran = [], 0
    for name, bsz, q_len, ctx, qh, kvh, dim in cases:
        c = make_case(bsz, q_len, ctx, qh, kvh, dim, dev)
        ref = build_reference(
            c["q"], c["k"], c["v"], c["k_cache"], c["v_cache"],
            c["block_table"], c["cache_seqlens"], c["sm_scale"], causal=True,
        )

        for bname, fn in backends:
            # Each backend gets a pristine cache: kernels may write new K/V back in place
            kc, vc = c["k_cache"].clone(), c["v_cache"].clone()
            a = AttnArgs(
                bsz=bsz, q_len=q_len, num_q_heads=qh, dim=dim,
                kv_len=q_len, num_kv_heads=kvh,
                q=c["q"], k=c["k"], v=c["v"],
                k_cache=kc, v_cache=vc,
                causal=True, sm_scale=c["sm_scale"],
                cu_seqlens=None, max_seqlen=None,
                window_size=None, softcap=0.0,
                block_table=c["block_table"], cache_seqlens=c["cache_seqlens"],
            )
            try:
                out = fn(a)
            except Exception as e:
                print(f"  {name:22s} {bname:8s} ERROR   {type(e).__name__}: {e}")
                failures.append((name, bname, f"{type(e).__name__}: {e}"))
                continue

            if out is None:
                continue  # backend declined these args -- expected for most combos

            ran += 1
            diff = (out.float() - ref).abs()
            mad, mx = diff.mean().item(), diff.max().item()
            nan = bool(torch.isnan(out).any())
            ok = (mx <= args_cli.tol) and not nan
            print(f"  {name:22s} {bname:8s} {'ok  ' if ok else 'FAIL'} "
                  f"max={mx:.4e} mean={mad:.4e}{'  NaN!' if nan else ''}")
            if not ok:
                failures.append((name, bname, f"max={mx:.4e} nan={nan}"))

    print()
    print(f"{ran} backend/shape combinations executed, {len(failures)} failed")
    if failures:
        print("\nPHASE 2: NO-GO -- upstream Triton paged attention is not correct here.")
        for n, b, d in failures:
            print(f"  {n} / {b}: {d}")
        return 1

    if ran == 0:
        print("\nPHASE 2: INCONCLUSIVE -- every backend declined every shape.")
        return 1

    print("\nPHASE 2: GO -- Triton paged attention matches the fp32 reference on this GPU.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
