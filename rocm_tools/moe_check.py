"""Compare the fused exl3_moe path against the per-expert torch path, in situ.

    python rocm_tools/moe_check.py -m /path/to/model [-b 1,2,4,8,16,64]

Runs one real BlockSparseMLP layer twice on IDENTICAL input -- once as the model
would run it, once with fused_mode_buffers cleared so every expert falls through
to the torch path (self.ups[i].forward, i.e. the per-expert exl3 Linear, which a
dense model exercises correctly) -- and reports the elementwise error.

Both calls go through the module's own forward, so routing, gathering, the
shared-expert branch and the output accumulation are identical between them and
cancel out of the diff. What remains is the fused kernel vs the reference.

Both runs also clear bc / support_quant_paths, which is load-bearing rather than
incidental -- without it the tool silently measures the wrong thing at every batch
size. See the comment on run() for the two specific ways that happens.

Why in situ rather than a synthetic harness: exl3_moe takes ~30 arguments of
expert pointer tables, and a harness that builds them by hand tests the harness.
The rule this follows is "verify kernels through the path the model actually
calls" -- a C++ harness on an inner kernel has already passed here once while the
shipped launch was wrong.

Interpreting the output: fp16 accumulation in a different order gives small
errors that do NOT grow with batch size. A path that is actually wrong shows a
large max error, or an error that changes character with batch size. The whole
point of sweeping -b is that exl3_moe is suspected of being correct at
multi-token and wrong at bsz == 1, so a single batch size cannot show it.
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-b", "--batches", default="1,2,4,8,16,64",
                    help="comma-separated token counts to sweep")
    ap.add_argument("-cs", "--cache_size", type=int, default=32768)
    ap.add_argument("-l", "--layers", type=int, default=3,
                    help="report the first N MoE layers per batch size")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    print(f" -- loading {args.model_dir}")
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache_size)
    model.load(progressbar=True)
    tok = Tokenizer.from_config(config)

    try:
        from exllamav3.rocm_py import describe
        print("ROCm patches:"); print(describe())
    except Exception:
        pass

    state = {"n": 0, "bsz": 0, "worst": 0.0}
    orig_forward = BlockSparseMLP.forward

    def forward_cmp(self, x, params, out_dtype=None):
        # Only compare layers that actually have a fused path to test.
        if getattr(self, "fused_mode_buffers", None) is None or state["n"] >= args.layers:
            return orig_forward(self, x, params, out_dtype)

        # params carries the residual for alt_residual_channel layers and is
        # mutated downstream, so both runs get their own shallow copy and their
        # own clone of the residual.
        # Both runs must be forced into branch 1057 (the per-expert / fused branch).
        # Two traps, both hit on the first version of this tool:
        #   - bszn_eligible is true for bsz <= MAX_BSZN (8), which sends BOTH runs to
        #     branch 1222 -> bc.run_bszN -> mgemm. The diff is then mgemm vs itself
        #     and reads as a perfect 0.
        #   - inside branch 1057, the per-expert loop only reaches the torch path when
        #     `bc is None and not support_quant_paths`; otherwise it takes
        #     bc.run_single_expert, so the "reference" is another kernel.
        # Clearing bc and support_quant_paths for both runs fixes both: the fused run
        # gets branch 1057 with the fused kernel, the reference gets the torch path.
        def run(fused):
            p = dict(params)
            if "residual" in p and isinstance(p["residual"], torch.Tensor):
                p["residual"] = p["residual"].clone()
            s_buf, s_bc, s_sqp = self.fused_mode_buffers, self.bc, self.support_quant_paths
            self.bc = None
            self.support_quant_paths = False
            if not fused:
                self.fused_mode_buffers = None
            try:
                return orig_forward(self, x.clone(), p, out_dtype)
            finally:
                self.fused_mode_buffers, self.bc, self.support_quant_paths = s_buf, s_bc, s_sqp

        out_fused = run(True).float()
        out_ref = run(False).float()

        d = (out_fused - out_ref).abs()
        denom = out_ref.abs().max().clamp_min(1e-6)
        maxabs = float(d.max())
        rel = maxabs / float(denom)
        nf_f = int((~torch.isfinite(out_fused)).sum())
        nf_r = int((~torch.isfinite(out_ref)).sum())

        # maxabs alone cannot tell a uniformly-wrong tensor from a handful of bad
        # elements, and those imply very different bugs. `bad` is the share of
        # elements off by more than 1% of the tensor scale; `col_span` is how many
        # distinct hidden-dim columns those land in -- a defect confined to one tile
        # or one expert shows up as a narrow span, arithmetic noise as a wide one.
        flat = d.view(-1, d.shape[-1])
        thresh = 0.01 * float(denom)
        badmask = flat > thresh
        nbad = int(badmask.sum())
        frac = nbad / max(flat.numel(), 1)
        meanabs = float(d.mean())
        cols = torch.nonzero(badmask.any(dim=0)).flatten()
        col_span = f"{cols.numel()}/{flat.shape[-1]}"
        cols_rng = f"[{int(cols.min())}..{int(cols.max())}]" if cols.numel() else "[-]"

        flag = ""
        if nf_f or nf_r:
            flag = f"  <== NONFINITE fused={nf_f} ref={nf_r}"
        elif rel > 0.05:
            flag = "  <== LARGE"
        state["worst"] = max(state["worst"], rel)

        print(f"   bsz={state['bsz']:<4} {self.key:44} "
              f"max={maxabs:8.4g} rel={rel:7.2%} mean={meanabs:8.3g} "
              f"bad={frac:6.2%} cols={col_span:>9} {cols_rng}{flag}")
        state["n"] += 1
        return out_fused

    BlockSparseMLP.forward = forward_cmp

    # A fixed pseudo-random token sequence: real ids so the embedding is
    # meaningful, but independent of tokenizer quirks at short lengths.
    g = torch.Generator().manual_seed(1234)
    vocab = int(getattr(config, "vocab_size", 0) or tok.actual_vocab_size or 32000)
    full = torch.randint(0, max(vocab - 1, 1), (1, max(batches)), generator=g)

    print("\n fused exl3_moe vs per-expert torch, same input\n")
    for b in batches:
        state["n"] = 0
        state["bsz"] = b
        ids = full[:, :b]
        with torch.inference_mode():
            model.prefill(input_ids=ids, params={"attn_mode": "flash_attn_nc"})
        print()

    print(f" worst relative error across sweep: {state['worst']:.3%}")
    print(" small and batch-independent => fp16 accumulation order (benign)")
    print(" large, or growing/appearing at one batch size => real defect")


if __name__ == "__main__":
    main()
