#!/usr/bin/env python3
"""Time the fused exl3_moe path, so the MoE tile-K choice can be justified with a number.

    python rocm_tools/bench_moe.py -m /path/to/model [-b 256,1024] [-r 7]

RDNA forces MOE_TILESIZE_K = 16 (rocm/quant/exl3_moe_shape_rdna.hip.h) because
upstream's 32 activates the split-K path in exl3_gemm_kernel_inner, which is
numerically broken on this hardware. That was a correctness decision taken without
measuring the cost. This measures it.

To A/B, rebuild with the override put back to upstream's value and re-run:

    python setup.py build_ext --inplace                       # K = 16 (default)
    EXL3_RDNA_MOE_TILESIZE_K=32 ... rebuild ... -> see below

The build flag has to reach hipcc, so either add
-DEXL3_RDNA_MOE_TILESIZE_K=32 to setup.py's define list, or edit the default in
exl3_moe_shape_rdna.hip.h. Note the K=32 build is KNOWN NUMERICALLY WRONG -- its
timing is "what correctness cost us", not a usable configuration.

Method: times only BlockSparseMLP.forward, synchronised per call, so attention and
the rest of the block do not dilute the signal. Reports the median of -r repeats
with the first discarded.

Per gfx1151-hardware-facts the benchmark noise floor here is ~4.7% spread (1.6%
stdev) and the first run reads high, so the median-of-repeats and the discarded
warmup are both load-bearing. Do not believe a difference under ~5%.
"""

import argparse, os, sys, time, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required = True)
    ap.add_argument("-b", "--batches", default = "256,1024",
                    help = "prompt lengths to time (MoE only runs fused above bsz 8)")
    ap.add_argument("-r", "--repeats", type = int, default = 7)
    ap.add_argument("-cs", "--cache_size", type = int, default = 32768)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    print(f" -- loading {args.model_dir}")
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache_size)
    model.load(progressbar = True)
    tok = Tokenizer.from_config(config)

    try:
        from exllamav3.rocm_py import describe
        print("ROCm patches:"); print(describe())
    except Exception:
        pass

    acc = {"t": 0.0, "n": 0}
    orig_forward = BlockSparseMLP.forward

    def forward_timed(self, x, params, out_dtype = None):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        y = orig_forward(self, x, params, out_dtype)
        torch.cuda.synchronize()
        acc["t"] += time.perf_counter() - t0
        acc["n"] += 1
        return y

    BlockSparseMLP.forward = forward_timed

    g = torch.Generator().manual_seed(1234)
    vocab = int(getattr(config, "vocab_size", 0) or tok.actual_vocab_size or 32000)
    full = torch.randint(0, max(vocab - 1, 1), (1, max(batches)), generator = g)

    print(f"\n MoE-layer time, median of {args.repeats - 1} timed runs (first discarded)\n")
    for b in batches:
        samples = []
        calls = 0
        for r in range(args.repeats):
            acc["t"] = 0.0
            acc["n"] = 0
            with torch.inference_mode():
                model.prefill(input_ids = full[:, :b], params = {"attn_mode": "flash_attn_nc"})
            if r > 0:
                samples.append(acc["t"])
                calls = acc["n"]
        med = statistics.median(samples)
        lo, hi = min(samples), max(samples)
        spread = (hi - lo) / med if med else 0.0
        per_call = med / calls if calls else 0.0
        print(f"   prompt={b:<6} moe_total={med * 1e3:8.2f} ms   "
              f"per_layer={per_call * 1e3:6.3f} ms   layers={calls:<4} "
              f"spread={spread:5.1%}   tok/s_moe={b / med:9.1f}")
        if spread > 0.05:
            print(f"      note: spread exceeds the ~4.7% noise floor -- rerun before trusting this")

    print("\n Compare against a rebuild with EXL3_RDNA_MOE_TILESIZE_K=32.")
    print(" That build is numerically wrong; the delta is the price of correctness,")
    print(" and only justifies fixing split-K if it is large.")


if __name__ == "__main__":
    main()
