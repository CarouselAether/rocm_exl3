#!/usr/bin/env python3
"""Parity check for the Triton paged-attention kernels at KV lengths straddling 8192,
where paged_attn_triton_prefill first activates its split+combine path (bound_kv >= 8192)
and the decode split count grows. Based on rocm_tools/attn_check.py (same stub-import
trick, same fp32 oracle), extended with longer contexts and sliding-window cases.
"""

import os
import sys
import types

import torch

REPO = "/home/carousel/Desktop/exlproject/rocm_exl3"
TOL = 2e-2


def build_reference(q, k, v, k_cache, v_cache, block_table, cache_seqlens, sm_scale,
                    causal, window_left=-1):
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
        k_buf[ctx:total] = k[b].float()
        v_buf[ctx:total] = v[b].float()
        k_buf = k_buf[:total]
        v_buf = v_buf[:total]

        out_b = torch.empty(q_len, n_q_heads, dim, dtype=torch.float32, device=q.device)
        pos = torch.arange(q_len, device=q.device).unsqueeze(1) + ctx
        key = torch.arange(total, device=q.device).unsqueeze(0)
        for h in range(n_q_heads):
            kv_h = h // group
            qh = q[b, :, h, :].float()
            kh = k_buf[:, kv_h, :]
            vh = v_buf[:, kv_h, :]
            scores = (qh @ kh.T) * sm_scale
            if causal:
                scores = scores.masked_fill(key > pos, float("-inf"))
            if window_left >= 0:
                scores = scores.masked_fill(key < pos - window_left, float("-inf"))
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
    sys.path.insert(0, REPO)
    for name, sub in [
        ("exllamav3", ""),
        ("exllamav3.modules", "modules"),
        ("exllamav3.modules.attention_fn", "modules/attention_fn"),
    ]:
        if name not in sys.modules:
            stub = types.ModuleType(name)
            stub.__path__ = [os.path.join(REPO, "exllamav3", sub)]
            sys.modules[name] = stub

    dev = torch.device("cuda:0")
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name} / {getattr(props, 'gcnArchName', '?')}  "
          f"torch {torch.__version__} hip={torch.version.hip}")

    from exllamav3.modules.attention_fn.common import AttnArgs
    from exllamav3.modules.attention_fn import triton_paged as tp
    assert tp.has_triton

    # name, bsz, q_len, ctx, qh, kvh, dim, window
    cases = [
        # decode across the boundary
        ("decode 4k",        1,    1,  4096, 32, 8, 128, None),
        ("decode 7.9k",      1,    1,  8000, 32, 8, 128, None),
        ("decode 8.5k",      1,    1,  8500, 32, 8, 128, None),
        ("decode 12k",       1,    1, 12288, 32, 8, 128, None),
        ("decode 16k",       1,    1, 16384, 32, 8, 128, None),
        # short continuation chunks over a long cache (prefill kernel, q_len > 16)
        ("chunk32 8.5k",     1,   32,  8500, 32, 8, 128, None),
        ("chunk128 12k",     1,  128, 12288, 32, 8, 128, None),
        # long prefill chunks: total crosses 8192 -> IS_SPLIT activates
        ("prefill 6k+2k",    1, 2048,  6144, 32, 8, 128, None),
        ("prefill 8k+2k",    1, 2048,  8192, 32, 8, 128, None),
        ("prefill 14k+2k",   1, 2048, 14336, 32, 8, 128, None),
        ("prefill 0+9k",     1, 9216,     0, 32, 8, 128, None),
        # sliding window over long context
        ("swa1k decode 12k", 1,    1, 12288, 32, 8, 128, 1024),
        ("swa1k chunk 12k",  1,  128, 12288, 32, 8, 128, 1024),
        ("swa4k prefill",    1, 2048,  8192, 32, 8, 128, 4096),
        # batched decode long
        ("decode b4 9k",     4,    1,  9000, 32, 8, 128, None),
    ]

    backends = [
        ("decode", tp.fn_triton_paged_attn_decode),
        ("prefill", tp.fn_triton_paged_attn_prefill),
        ("paged", tp.fn_triton_paged_attn),
        ("longq", tp.fn_triton_paged_attn_longq),
    ]

    failures, ran = [], 0
    for name, bsz, q_len, ctx, qh, kvh, dim, window in cases:
        c = make_case(bsz, q_len, ctx, qh, kvh, dim, dev)
        wl = window if window is not None else -1
        ref = build_reference(
            c["q"], c["k"], c["v"], c["k_cache"], c["v_cache"],
            c["block_table"], c["cache_seqlens"], c["sm_scale"],
            causal=True, window_left=wl,
        )

        for bname, fn in backends:
            kc, vc = c["k_cache"].clone(), c["v_cache"].clone()
            a = AttnArgs(
                bsz=bsz, q_len=q_len, num_q_heads=qh, dim=dim,
                kv_len=q_len, num_kv_heads=kvh,
                q=c["q"], k=c["k"], v=c["v"],
                k_cache=kc, v_cache=vc,
                causal=True, sm_scale=c["sm_scale"],
                cu_seqlens=None, max_seqlen=None,
                window_size=window, softcap=0.0,
                block_table=c["block_table"], cache_seqlens=c["cache_seqlens"],
            )
            try:
                out = fn(a)
            except Exception as e:
                print(f"  {name:18s} {bname:8s} ERROR   {type(e).__name__}: {e}")
                failures.append((name, bname, f"{type(e).__name__}: {e}"))
                continue
            if out is None:
                continue
            ran += 1
            diff = (out.float() - ref).abs()
            mad, mx = diff.mean().item(), diff.max().item()
            nan = bool(torch.isnan(out).any())
            ok = (mx <= TOL) and not nan
            print(f"  {name:18s} {bname:8s} {'ok  ' if ok else 'FAIL'} "
                  f"max={mx:.4e} mean={mad:.4e}{'  NaN!' if nan else ''}")
            if not ok:
                failures.append((name, bname, f"max={mx:.4e} nan={nan}"))
        del c, ref
        torch.cuda.empty_cache()

    print(f"\n{ran} combos executed, {len(failures)} failed")
    for n, b, d in failures:
        print(f"  {n} / {b}: {d}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
