#!/usr/bin/env python3
"""DeepSeek-V4 decode sparse attention (DSA split + combine): microbenchmark and correctness.

Calls the decode-path kernels with the shapes and arguments the graphed decode path
(bc_dsa.BCDsa / BCDsaBatch -> dsv4_attn.cpp) passes them, on synthetic caches:

  H 64 query heads, 1 KV head (MQA), D 512 = D_c 448 latent + D_r 64 rope, window 128,
  sinks, eq. 26 de-rotation, group-major output (8 groups). Pool geometry per layer kind:
    csa  compress 4,   epp 64: dense pool while ec <= 512 (ctx <= 2K), else top-k 512
    hca  compress 128, epp 2 : dense pool
    win  no compressor (ratio-0 layers): window only

Variants:
  old  dsa_triton._dsa_attn_split_kernel + _dsa_attn_combine_kernel (what bc_dsa compiles)
  new  rocm_py.dsa_decode_rdna._dsa_decode_mqa_kernel + the same combine

Timing: split+combine captured 20x into a CUDA graph and replayed (launch overhead out of
the picture, like the decode graphs); us per (split + combine) call. Correctness: fp32/fp64
torch reference of the same attention (window ++ pool, sinks, de-rotation, QC pools read
through the reference dequantizer), max abs error relative to max |ref|.

Resource usage (VGPR, spills, scratch, LDS, WMMA count) from the compiled kernel's AMDGCN
metadata.

    python rocm_tools/bench_dsa_decode.py                      # before/after table
    python rocm_tools/bench_dsa_decode.py --sweep-old          # old-kernel parameter sweep
    python rocm_tools/bench_dsa_decode.py --sweep-new          # new-kernel parameter sweep
"""

import argparse
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import triton

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.attention_fn import dsa_triton as dt
from exllamav3.modules.attention_fn.triton_paged import _get_h32

H, D_C, D_R, WINDOW, TOPK, G_OUT = 64, 448, 64, 128, 512, 8
D = D_C + D_R
DEV = torch.device("cuda:0")


# ----------------------------------------------------------------------------------------
# Cases
# ----------------------------------------------------------------------------------------

class Case:
    """One decode attention call: `seq` query rows per job, `B` jobs (MULTIROW when B > 0)."""

    def __init__(self, kind, ctx, seq = 1, qc = 0, B = 0, seed = 0):
        g = torch.Generator(device = "cpu").manual_seed(seed)
        self.kind, self.ctx, self.seq, self.qc, self.B = kind, ctx, seq, qc, B
        nj = max(B, 1)
        self.R = R = nj * seq
        self.m = {"csa": 4, "hca": 128, "win": 1}[kind]
        self.has_comp = kind != "win"
        self.epp = 256 // self.m if self.has_comp else 256
        # Per-job positions: pos = ctx (+ a small per-job skew under MULTIROW)
        self.pos = [ctx + 3 * j for j in range(nj)]
        self.win_beg = [p - 200 for p in self.pos]
        self.ec = [(p + seq) // self.m if self.has_comp else 0 for p in self.pos]
        self.topk = self.has_comp and kind == "csa" and max(self.ec) > TOPK
        self.floor = [p - min(WINDOW - 1, p - wb, p) for p, wb in zip(self.pos, self.win_beg)]

        def rn(*shape, s = 1.0):
            return (torch.randn(shape, generator = g) * s).half().to(DEV)

        # q: realistic scale so the softmax is neither flat nor one-hot
        self.q = rn(R, H, D, s = 1.0)
        ring_rows = 512
        self.ring = rn(nj, ring_rows, D)          # rows at abs - win_beg
        self.kv = rn(R, D)                        # this step's rows at abs - pos
        self.sinks = (torch.randn(H, generator = g) * 2.0 + 4.0).to(DEV)
        self.inv_freq_neg = (-1.0 / (10000.0 ** (torch.arange(0, D_R, 2).float() / D_R))).to(DEV)

        # Paged pools: enough pages for the max entry count, pages shuffled per job
        n_ent = max(max(self.ec), 1)
        pages = -(-n_ent // self.epp) + 1
        self.num_pages = pages * nj + 2
        rows = self.num_pages * self.epp
        pool_c = rn(rows, D_C)
        self.pool_r = rn(rows, D_R)
        perm = torch.randperm(self.num_pages, generator = g).int()
        self.bt = perm[:pages * nj].view(nj, pages).contiguous().to(DEV)
        self.npr = pages
        self.h32 = _get_h32(DEV)
        if qc:
            Gc = D_C // 32
            pq = torch.empty((rows, Gc * qc), dtype = torch.int32, device = DEV)
            ps = torch.empty((rows, Gc), dtype = torch.half, device = DEV)
            ext.quant_cache_cont(pool_c, pq, ps, 0.0)
            deq = torch.empty((rows, D_C), dtype = torch.half, device = DEV)
            ext.dequant_cache_cont(pq, ps, deq, 0.0)
            self.pool_c_ref = deq
            self.pool_c_arg = pq
            self.pool_s = ps
        else:
            self.pool_c_ref = pool_c
            self.pool_c_arg = pool_c
            self.pool_s = self.q      # dummy fp16 pointer
        # Top-k indices (topk regime): K_pad = 512, distinct entries in [0, ec)
        self.K_pad = 512 if kind == "csa" else 32
        self.indices = torch.full((R, self.K_pad), -1, dtype = torch.int32)
        if self.topk:
            for r in range(R):
                ec = self.ec[r // seq]
                self.indices[r, :TOPK] = torch.randperm(ec, generator = g)[:TOPK].int()
        self.indices = self.indices.to(DEV)
        self.k_len = TOPK if self.topk else 0
        # MULTIROW per-job arrays
        if B:
            i32 = lambda v: torch.tensor(v, dtype = torch.int32, device = DEV)
            self.a_pos, self.a_floor, self.a_beg = i32(self.pos), i32(self.floor), i32(self.win_beg)
            self.a_ec, self.a_klen = i32(self.ec), i32([self.K_pad] * nj)
            self.a_slots = i32(list(range(nj)))
        self.out = torch.empty((G_OUT, R, (H // G_OUT) * D), dtype = torch.half, device = DEV)

    def tag(self):
        s = f"{self.kind}{'-topk' if self.topk else ''} ctx{self.ctx} seq{self.seq} qc{self.qc}"
        return s + (f" B{self.B}" if self.B else "")

    def n_keys(self, r):
        j = r // self.seq
        loc = r % self.seq
        qa = self.pos[j] + loc
        nw = qa - max(self.floor[j], qa - WINDOW + 1) + 1
        if not self.has_comp:
            return nw
        if self.topk:
            return nw + TOPK
        return nw + min((self.pos[j] + loc + 1) // self.m, self.ec[j])

    def reference(self):
        """fp64 reference, group-major (G, R, hpg * D) fp32."""
        out = torch.zeros((self.R, H, D), dtype = torch.float64, device = DEV)
        pc = self.pool_c_ref.double()
        pr = self.pool_r.double()
        scale = D ** -0.5
        for r in range(self.R):
            j, loc = r // self.seq, r % self.seq
            q_pos0 = self.pos[j]
            qa = q_pos0 + loc
            rows = []
            for a in range(qa, qa - WINDOW, -1):
                if a < self.floor[j]:
                    break
                if a >= q_pos0:
                    rows.append(self.kv[j * self.seq + a - q_pos0])
                else:
                    rows.append(self.ring[j, a - self.win_beg[j]])
            kv = torch.stack(rows).double()
            if self.has_comp:
                if self.topk:
                    idx = self.indices[r]
                    idx = idx[idx >= 0].long()
                else:
                    idx = torch.arange(min((qa + 1) // self.m, self.ec[j]), device = DEV)
                if idx.numel():
                    phys = self.bt[j, idx // self.epp].long()
                    tok = phys * self.epp + idx % self.epp
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
            out[r] = o
        return out.view(self.R, G_OUT, (H // G_OUT) * D).transpose(0, 1).float()


# ----------------------------------------------------------------------------------------
# Launchers
# ----------------------------------------------------------------------------------------

def _ws(c, n_splits, block_h):
    hb = H // block_h
    ws_ml = torch.empty((c.R * hb * n_splits * block_h * 2,), dtype = torch.float, device = DEV)
    ws_acc = torch.empty((c.R * hb * n_splits * block_h * D,), dtype = torch.float, device = DEV)
    return ws_ml, ws_acc


def _split_args(c, ws_ml, ws_acc):
    pool_c = c.pool_c_arg if c.has_comp else c.kv
    pool_r = c.pool_r if c.has_comp else c.kv
    idx = c.indices if c.topk else c.bt
    if c.B:
        return (c.q, c.ring, c.kv, pool_c, pool_r, c.bt, idx, ws_ml, ws_acc,
                c.a_klen, WINDOW, c.a_ec, c.npr, c.a_pos, c.a_floor, c.a_beg,
                c.a_slots, c.ring.shape[1] * D, c.pool_s if c.qc else c.sinks,
                c.h32 if c.qc else c.sinks)
    return (c.q, c.ring[0], c.kv, pool_c, pool_r, c.bt, idx, ws_ml, ws_acc,
            c.k_len, WINDOW, c.ec[0], 0, c.pos[0], c.floor[0], c.win_beg[0],
            0, 0, c.pool_s if c.qc else c.sinks, c.h32 if c.qc else c.sinks)


def _consts(c, block_h, block_n = 32, block_w = 16):
    return dict(
        H = H, page_size = c.epp, D_c = D_C, D_c_pad = 512, D_r = D_R,
        K_pad = c.K_pad, compress_rate = c.m, scale = D ** -0.5,
        HAS_WINDOW = True, DENSE_POOL = not c.topk,
        BLOCK_H = block_h, BLOCK_N = block_n, BLOCK_W = block_w,
        SEQ = c.seq if c.B else 1, MULTIROW = 1 if c.B else 0,
        DEBUG_BOUNDS = 0, DEBUG_PAGES = 0, Q_SPLIT = 0, OUT_LATENT = 0, QC = c.qc,
    )


class Runner:
    """Holds workspaces and launches split + combine for one (case, variant)."""

    def __init__(self, c, variant, n_splits = 16, block_h = 8, block_n = 32, block_w = 16,
                 num_warps = 8, num_stages = 2, extra = None):
        self.c, self.variant, self.n_splits, self.block_h = c, variant, n_splits, block_h
        self.block_n, self.block_w, self.num_warps, self.num_stages = block_n, block_w, num_warps, num_stages
        self.extra = extra or {}
        self.ws_ml, self.ws_acc = _ws(c, n_splits, block_h)
        self.ck = None
        if variant == "old":
            k = dt._dsa_attn_split_kernel
            self.kernel = getattr(k, "orig", k)      # unwrap the rocm_py launch proxy
        else:
            from exllamav3.rocm_py import dsa_decode_rdna as dd
            self.kernel = dd._dsa_decode_mqa_kernel

    def split(self):
        c = self.c
        hb = H // self.block_h
        consts = _consts(c, self.block_h, self.block_n, self.block_w) | self.extra
        ck = self.kernel[(c.R * hb, self.n_splits)](
            *_split_args(c, self.ws_ml, self.ws_acc), **consts,
            num_warps = self.num_warps, num_stages = self.num_stages)
        if self.ck is None:
            self.ck = ck

    def combine(self):
        c = self.c
        hb = H // self.block_h
        dt._dsa_attn_combine_kernel[(c.R * hb, triton.cdiv(D, 128))](
            self.ws_ml, self.ws_acc, c.sinks, c.inv_freq_neg, c.out,
            c.a_pos if c.B else c.pos[0], c.R, self.n_splits, c.h32 if c.qc else c.sinks,
            H = H, D_c = D_C, D_r = D_R, HAS_SINKS = True, DEROTATE = True,
            HPG = H // G_OUT, BLOCK_H = self.block_h, BLOCK_D = 128,
            SEQ = c.seq if c.B else 1, MULTIROW = 1 if c.B else 0, OUT_LATENT = 0, QC = c.qc,
            num_warps = 4, num_stages = 2)

    def run(self):
        self.split()
        self.combine()
        return self.c.out


def kernel_resources(ck):
    """VGPR / spill / scratch / LDS / WMMA-count from a CompiledKernel's AMDGCN."""
    if ck is None:
        return {}
    asm = ck.asm.get("amdgcn", "")
    def f(key):
        m = re.search(rf"\.{key}:\s+(\d+)", asm)
        return int(m.group(1)) if m else -1
    vgpr = f("vgpr_count")
    return dict(
        vgpr = vgpr, spill = f("vgpr_spill_count"), scratch = f("private_segment_fixed_size"),
        lds = f("group_segment_fixed_size"), wmma = len(re.findall(r"\bv_wmma_", asm)),
        waves = min(16, 1536 // max(vgpr, 1)) if vgpr > 0 else -1,
    )


def time_graph(fns, iters = 40, reps = 8):
    """Capture `iters` rounds of fns into one graph; best-of-reps us per round."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(2):
            for f in fns:
                f()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(iters):
            for f in fns:
                f()
    g.replay()
    torch.cuda.synchronize()
    best = 1e30
    e0, e1 = torch.cuda.Event(enable_timing = True), torch.cuda.Event(enable_timing = True)
    for _ in range(reps):
        e0.record()
        g.replay()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000.0 / iters)
    return best


def check(runner, ref = None):
    c = runner.c
    c.out.zero_()
    out = runner.run().float()
    torch.cuda.synchronize()
    if ref is None:
        ref = c.reference()
    err = (out - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    return err, ref


def measure(runner, ref = None, do_check = True):
    err = None
    if do_check:
        err, ref = check(runner, ref)
    t_all = time_graph([runner.split, runner.combine])
    t_split = time_graph([runner.split])
    res = kernel_resources(runner.ck)
    return dict(err = err, us = t_all, us_split = t_split, **res), ref


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------

OLD_DEFAULT = dict(n_splits = 16, block_h = 8, block_n = 32, block_w = 16, num_warps = 8, num_stages = 2)


def std_cases(quick = False):
    cases = []
    for ctx in ([512] if quick else [512, 16384]):
        for kind in ["csa", "hca", "win"] if not quick else ["csa"]:
            for seq in [1, 3]:
                for qc in ([0, 4] if kind != "win" else [0]):
                    cases.append(dict(kind = kind, ctx = ctx, seq = seq, qc = qc))
    return cases


def fmt_row(tag, r):
    e = f"{r['err']:.2e}" if r.get("err") is not None else "-"
    return (f"  {tag:<34} {r['us']:8.1f} {r['us_split']:8.1f}  err {e}  "
            f"vgpr {r.get('vgpr', -1):3d} spill {r.get('spill', -1):4d} scratch {r.get('scratch', -1):5d} "
            f"lds {r.get('lds', -1):5d} wmma {r.get('wmma', -1):3d} waves {r.get('waves', -1)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-old", action = "store_true")
    ap.add_argument("--sweep-new", action = "store_true")
    ap.add_argument("--quick", action = "store_true")
    ap.add_argument("--only", default = "", help = "old|new")
    ap.add_argument("--mr", action = "store_true", help = "also MULTIROW (B=2 jobs) cases")
    args = ap.parse_args()
    torch.cuda.set_device(DEV)

    if args.sweep_old:
        sweep_old()
        return
    if args.sweep_new:
        sweep_new()
        return

    cases = std_cases(args.quick)
    if args.mr:
        cases += [dict(kind = "csa", ctx = 512, seq = 1, qc = 0, B = 2),
                  dict(kind = "csa", ctx = 16384, seq = 3, qc = 4, B = 2),
                  dict(kind = "hca", ctx = 16384, seq = 2, qc = 0, B = 3)]
    print(f"  {'case':<34} {'us/call':>8} {'split':>8}")
    for cd in cases:
        c = Case(**cd)
        ref = None
        if args.only in ("", "old"):
            rk = dict(OLD_DEFAULT)
            if c.B:
                rk["n_splits"] = 8
            r, ref = measure(Runner(c, "old", **rk), ref)
            print(fmt_row("old " + c.tag(), r), flush = True)
        if args.only in ("", "new"):
            from exllamav3.rocm_py import dsa_decode_rdna as dd
            rk = dd.bench_config(c.B > 0)
            r, ref = measure(Runner(c, "new", **rk), ref)
            print(fmt_row("new " + c.tag(), r), flush = True)


def sweep_old():
    c512 = Case("csa", 512)
    c16k = Case("csa", 16384)
    ref512 = c512.reference()
    print("  n_splits sweep at the current tuning (H8 N32 w8 s2)")
    for ns in [1, 2, 4, 8, 16]:
        for c, ref in [(c512, ref512), (c16k, None)]:
            r, _ = measure(Runner(c, "old", **(OLD_DEFAULT | dict(n_splits = ns))), ref, do_check = c is c512)
            print(fmt_row(f"ns{ns} {c.tag()}", r), flush = True)
    print("  BLOCK_H x warps x stages sweep (N32, ns16)")
    for bh, nw, st in itertools.product([8, 16, 32, 64], [4, 8, 16], [1, 2]):
        try:
            r, _ = measure(Runner(c512, "old", **(OLD_DEFAULT | dict(block_h = bh, num_warps = nw, num_stages = st))),
                           ref512)
            print(fmt_row(f"H{bh} w{nw} s{st} {c512.tag()}", r), flush = True)
        except Exception as e:
            print(f"  H{bh} w{nw} s{st}: {type(e).__name__}: {str(e)[:100]}", flush = True)
    print("  BLOCK_N / BLOCK_W sweep (H8 w8 s2 ns16)")
    for bn, bw in itertools.product([16, 32, 64], [16, 32]):
        try:
            r, _ = measure(Runner(c512, "old", **(OLD_DEFAULT | dict(block_n = bn, block_w = bw))), ref512)
            print(fmt_row(f"N{bn} W{bw} {c512.tag()}", r), flush = True)
        except Exception as e:
            print(f"  N{bn} W{bw}: {type(e).__name__}: {str(e)[:100]}", flush = True)


def sweep_new():
    from exllamav3.rocm_py import dsa_decode_rdna as dd
    cs = [Case("csa", 512), Case("csa", 16384), Case("csa", 512, seq = 3), Case("csa", 512, qc = 4),
          Case("win", 512)]
    refs = [c.reference() for c in cs]
    for cfg in dd.sweep_configs():
        rk = dict(cfg)
        row = []
        for c, ref in zip(cs, refs):
            try:
                r, _ = measure(Runner(c, "new", **rk), ref)
                row.append(r)
            except Exception as e:
                row.append(None)
                print(f"  {cfg} {c.tag()}: {type(e).__name__}: {str(e)[:200]}", flush = True)
        desc = " ".join(f"{k}={v}" for k, v in cfg.items() if k != "extra") + " " + \
               " ".join(f"{k}={v}" for k, v in cfg.get("extra", {}).items())
        cells = " | ".join(
            (f"{r['us']:6.1f} e{r['err']:.0e}" if r else "  fail") for r in row)
        res = row[0] or {}
        print(f"  {desc:<60} {cells}   vgpr {res.get('vgpr', -1)} spill {res.get('spill', -1)} "
              f"wmma {res.get('wmma', -1)}", flush = True)


if __name__ == "__main__":
    main()
