#!/usr/bin/env python3
"""Correctness check for the WMMA GEMM backend of ext.hgemm (rocm/wmma_gemm_rdna.hip).

    rocm_tools/wmma_gemm_check.py            # full matrix (DS4 shapes, edges, fallbacks)
    rocm_tools/wmma_gemm_check.py --quick    # a subset

For every case, ext.hgemm runs twice in this process: with the WMMA path forced on for
every eligible call (EXL3_ROCM_WMMA_GEMM=2, bypassing the measured routing policy so every
shape exercises the kernels) and with EXL3_ROCM_WMMA_GEMM=0 (hipBLAS). Both are compared with a float64 reference computed from
the same fp16 inputs. The environment switches are read per call, so toggling os.environ
here is enough.

Pass criteria:
  - fp32 out: elementwise |c - ref| <= K * 2^-24 * (|A| @ |B|), the textbook worst-case bound
    for an fp32-accumulated dot product of length K (gamma_K). Both paths are checked against
    it and both max|err| are reported. They differ in kind: hipBLAS's fp32-output kernel is
    VALU FMA (round-to-nearest per step, error grows ~sqrt(K)); the WMMA accumulator on
    RDNA3 rounds each 16-deep step toward zero, so its error is biased and grows ~K/16 --
    measured ~5-10x hipBLAS's max|err| at K = 2-8K, ~1e-5 relative at K = 4096. hipBLAS's own
    fp16-output kernels use the same WMMA instruction, so this is the matrix-core contract.
  - fp16 out: elementwise |c - ref| <= 1 ulp_fp16(ref) + the fp32 error actually measured
    (faithful rounding; hipBLAS's own fp16 output is not always the nearest fp16 either,
    measured up to ~0.51 ulp at K = 8192), and the WMMA fp16 output must be bitwise the
    round-to-nearest of the WMMA fp32 output (same accumulators, fp16 epilogue)
  - routing: a case the backend should take must differ bitwise from hipBLAS in fp32 output
    (different accumulation), a case it must decline (m <= 8, strided C, misaligned pointer,
    N % 8, K % block_k) must be bitwise identical to hipBLAS in both outputs. That proves
    which path ran. (fp16 output alone cannot: at tiny M the two often round identically.)
  - M edges (1, 7, ...) are also forced through the kernel with EXL3_ROCM_WMMA_GEMM_MIN_M=1,
    exercising its partial-tile loads and bounds-checked stores.
  - a strided-C write leaves the neighbouring columns untouched.
  - HIP graph capture + replay of a routed call reproduces the eager result bitwise.
"""

import argparse
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3.ext import exllamav3_ext as ext

MS = [1, 7, 9, 64, 160, 255, 256, 1023, 1792]
NK = [(4096, 8192), (4096, 2048), (1024, 4096), (2048, 4096), (512, 4096), (256, 4096), (32768, 1024)]


def set_env(on = True, f16 = True, min_m = None, mode = None):
    os.environ["EXL3_ROCM_WMMA_GEMM"] = str(mode) if mode is not None else ("2" if on else "0")
    os.environ["EXL3_ROCM_WMMA_GEMM_F16"] = "1" if f16 else "0"
    if min_m is None:
        os.environ.pop("EXL3_ROCM_WMMA_GEMM_MIN_M", None)
    else:
        os.environ["EXL3_ROCM_WMMA_GEMM_MIN_M"] = str(min_m)


def run(a, b, dtype, out = None):
    c = out if out is not None else torch.empty(a.shape[:-1] + (b.shape[1],), dtype = dtype, device = a.device)
    c.fill_(float("nan")) if out is None else None
    ext.hgemm(a, b, c)
    return c


def ulp16(x):
    # spacing of fp16 at |x| (normal range; subnormal spacing below 2^-14)
    ax = x.abs().clamp(min = 2.0 ** -14)
    e = torch.floor(torch.log2(ax))
    return torch.pow(2.0, e - 10)


class Result:
    def __init__(self):
        self.fail = 0
        self.n = 0

    def check(self, ok, msg):
        self.n += 1
        if not ok:
            self.fail += 1
        print(("  ok    " if ok else "  FAIL  ") + msg, flush = True)


def case(res, m, n, k, expect_route, label = "", min_m = None, a = None, b = None):
    dev = "cuda"
    if a is None:
        a = (torch.randn((m, k), device = dev) * 0.5).half()
    if b is None:
        b = (torch.randn((k, n), device = dev) * (1.0 / k ** 0.5)).half()
    ref = (a.double().reshape(-1, k) @ b.double()).reshape(a.shape[:-1] + (n,))
    absref = (a.double().abs().reshape(-1, k) @ b.double().abs()).reshape(ref.shape)
    gamma = k * 2.0 ** -24 * absref + 1e-30

    set_env(on = False)
    h32 = run(a, b, torch.float)
    h16 = run(a, b, torch.half)
    set_env(on = True, min_m = min_m)
    w32 = run(a, b, torch.float)
    w16 = run(a, b, torch.half)
    set_env()

    e_h32 = (h32.double() - ref).abs().max().item()
    e_w32 = (w32.double() - ref).abs().max().item()
    e_h16 = (h16.double() - ref).abs().max().item()
    e_w16 = (w16.double() - ref).abs().max().item()
    g_w32 = ((w32.double() - ref).abs() / gamma).max().item()
    g_h32 = ((h32.double() - ref).abs() / gamma).max().item()
    ok32 = g_w32 <= 1.0 and g_h32 <= 1.0 and torch.isfinite(w32).all().item()
    tol16 = ulp16(ref) + (w32.double() - ref).abs() + 1e-12
    ok16 = bool(((w16.double() - ref).abs() <= tol16).all().item()) and torch.isfinite(w16).all().item()
    routed32 = not torch.equal(w32, h32)
    routed16 = not torch.equal(w16, h16)
    route_ok = (routed32 == expect_route) and (expect_route or not routed16)
    # the WMMA fp16 epilogue rounds the same fp32 accumulators: it must equal fp32-out rounded
    rne16 = torch.equal(w16, w32.half())
    if routed32:
        ok16 = ok16 and rne16
    else:
        # a declined call is hipBLAS's result (bitwise, checked by route_ok); hipBLAS's fp16
        # output is not held to the faithful-rounding bound (it misses it by a hair at times)
        ok16 = torch.isfinite(w16).all().item()

    tag = f"m={m:<5} n={n:<5} k={k:<5} {label}"
    res.check(ok32 and ok16 and route_ok,
              f"{tag:<44} fp32 err wmma {e_w32:.2e} hipblas {e_h32:.2e} (of bound {g_w32:.3f} {g_h32:.3f}) "
              f"| fp16 err wmma {e_w16:.2e} hipblas {e_h16:.2e}{'' if (rne16 or not routed32) else ' NOT-RNE'} | routed {'Y' if routed32 else 'n'} "
              f"(expect {'Y' if expect_route else 'n'})")
    return e_w32, e_h32, e_w16, e_h16, g_w32


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action = "store_true")
    args = ap.parse_args()
    torch.manual_seed(0)
    prop = torch.cuda.get_device_properties(0)
    arch = getattr(prop, "gcnArchName", "?").split(":")[0]
    has_table = arch in ("gfx1151", "gfx1100", "gfx1101")
    print(f"device {prop.name} arch {arch}  torch {torch.__version__}  table: {'yes' if has_table else 'no (all fallback)'}")
    res = Result()

    ms = [1, 9, 255, 1792] if args.quick else MS
    nks = NK[:3] if args.quick else NK

    print("\n-- DS4 shapes (default routing: m > 8)")
    worst = [0.0, 0.0]
    for (n, k) in nks:
        for m in ms:
            r = case(res, m, n, k, expect_route = has_table and m > 8)
            if m > 8:
                worst[0] = max(worst[0], r[0] / max(r[1], 1e-30))
                worst[1] = max(worst[1], r[4])

    if has_table:
        print("\n-- every compiled config for this arch, pinned with EXL3_ROCM_WMMA_GEMM_CFG")
        src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "exllamav3", "exllamav3_ext", "rocm", "wmma_gemm_table_rdna.hip.h")).read()
        cfgs = [tuple(int(x) for x in mm.group(1).split(","))
                for mm in re.finditer(r"\{ ([\d, ]+), exl3_wmma_launch_c\d+ \}", src)]
        bit = 1 if arch == "gfx1151" else 2
        for i, c in enumerate(cfgs):
            if not (c[8] & bit):
                continue
            bm, bn = c[0] * c[2] * 16, c[1] * c[3] * 16
            # one partial-tile shape (M, N off the tile grid) and one exact multiple of the tile
            for (m, n, k) in [(255, 1000, 4096), (bm * 2, bn * 3, 1024)]:
                os.environ["EXL3_ROCM_WMMA_GEMM_CFG"] = str(i)
                case(res, m, n, k, expect_route = True, label = f"cfg {i} {bm}x{bn}x{c[4] * 16}")
            os.environ.pop("EXL3_ROCM_WMMA_GEMM_CFG", None)

    print("\n-- small M forced through the kernel (EXL3_ROCM_WMMA_GEMM_MIN_M=1)")
    for m in [1, 2, 7, 8, 17]:
        for (n, k) in [(4096, 2048), (1024, 4096)]:
            case(res, m, n, k, expect_route = has_table, label = "forced", min_m = 1)

    print("\n-- odd shapes")
    # admitted: N % 8 == 0 but not a multiple of the tile; K a multiple of 64
    for (m, n, k) in [(255, 1000, 4096), (100, 4104, 2048), (1023, 200, 1024), (33, 4096, 64 * 7)]:
        case(res, m, n, k, expect_route = has_table, label = "edge tiles")
    # declined: N % 8 != 0, K % 16 != 0
    for (m, n, k) in [(255, 4100, 4096), (255, 1001, 1024), (255, 4096, 1000), (64, 4096, 4104)]:
        case(res, m, n, k, expect_route = False, label = "fallback (N%8 / K%16)")

    print("\n-- batched leading dims (a is [2, 128, K])")
    a = (torch.randn((2, 128, 4096), device = "cuda") * 0.5).half()
    case(res, 256, 2048, 4096, expect_route = has_table, label = "3-D a", a = a)

    print("\n-- misaligned A pointer (2-byte offset view) -> fallback")
    buf = (torch.randn((255 * 4096 + 1,), device = "cuda") * 0.5).half()
    a = buf[1:].view(255, 4096)
    case(res, 255, 2048, 4096, expect_route = False, label = "misaligned A", a = a)

    print("\n-- strided C (column slice, like exl3.py's reconstruct slices) -> fallback")
    for dtype in (torch.float, torch.half):
        m, k, n_full, n0, n1 = 255, 2048, 4096, 1024, 3072
        a = (torch.randn((m, k), device = "cuda") * 0.5).half()
        b = (torch.randn((k, n1 - n0), device = "cuda") * (1.0 / k ** 0.5)).half()
        ref = a.double() @ b.double()
        outs = []
        for on in (True, False):
            set_env(on = on)
            big = torch.full((m, n_full), 7.0, dtype = dtype, device = "cuda")
            ext.hgemm(a, b, big[:, n0:n1])
            outs.append(big)
        set_env()
        err = (outs[0][:, n0:n1].double() - ref).abs().max().item()
        untouched = bool((outs[0][:, :n0] == 7).all().item() and (outs[0][:, n1:] == 7).all().item())
        same = torch.equal(outs[0], outs[1])
        res.check(untouched and same and err < 1e-2,
                  f"strided C {str(dtype):<14} err {err:.2e}  neighbours untouched {untouched}  == hipBLAS {same}")

    print("\n-- HIP graph capture of a routed call")
    for dtype in (torch.float, torch.half):
        m, n, k = 1792, 4096, 2048
        a = (torch.randn((m, k), device = "cuda") * 0.5).half()
        b = (torch.randn((k, n), device = "cuda") * (1.0 / k ** 0.5)).half()
        c = torch.empty((m, n), dtype = dtype, device = "cuda")
        set_env(on = True)
        ext.hgemm(a, b, c)
        eager = c.clone()
        c.zero_()
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.cuda.graph(g, stream = s):
                ext.hgemm(a, b, c)
        torch.cuda.current_stream().wait_stream(s)
        c.zero_()
        g.replay()
        torch.cuda.synchronize()
        res.check(torch.equal(c, eager), f"graph replay {str(dtype):<14} bitwise == eager")
        set_env()

    print("\n-- default policy (EXL3_ROCM_WMMA_GEMM=1): which DS4 shapes route (from EXL3_ROCM_WMMA_GEMM_TRACE)")
    import tempfile
    tf = tempfile.TemporaryFile(mode = "w+")
    torch.cuda.synchronize()
    sys.stderr.flush()
    saved = os.dup(2)
    os.dup2(tf.fileno(), 2)
    try:
        set_env(mode = 1)
        os.environ["EXL3_ROCM_WMMA_GEMM_TRACE"] = "1"
        for (n, k) in nks:
            for m in ms:
                a = (torch.randn((m, k), device = "cuda") * 0.5).half()
                b = (torch.randn((k, n), device = "cuda") * (1.0 / k ** 0.5)).half()
                for dt in (torch.float, torch.half):
                    run(a, b, dt)
        torch.cuda.synchronize()
    finally:
        os.environ.pop("EXL3_ROCM_WMMA_GEMM_TRACE", None)
        set_env()
        os.dup2(saved, 2)
        os.close(saved)
    tf.seek(0)
    dec = {}
    for line in tf:
        mm = re.match(r"\[wmma_gemm\] m=(\d+) n=(\d+) k=(\d+) out=(\w+) -> (\w+)", line)
        if mm:
            dec[(int(mm.group(1)), int(mm.group(2)), int(mm.group(3)), mm.group(4))] = "W" if mm.group(5) == "wmma" else "h"
    for (n, k) in nks:
        line = [f"{m}:{dec.get((m, n, k, 'fp32'), '-')}{dec.get((m, n, k, 'fp16'), '-')}" for m in ms]
        print(f"  n={n:<5} k={k:<5} " + " ".join(line))
    print("  (per M: fp32 then fp16 output; W = WMMA, h = hipBLAS, - = m <= 8, not traced)")

    print(f"\nrouted DS4 shapes: worst fp32 max|err| ratio wmma/hipBLAS {worst[0]:.2f}, "
          f"worst fraction of the gamma_K bound {worst[1]:.3f}")
    print(f"=== {'PASS' if res.fail == 0 else 'FAIL'}: {res.n - res.fail}/{res.n} ===")
    return 1 if res.fail else 0


if __name__ == "__main__":
    sys.exit(main())
