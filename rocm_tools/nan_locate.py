"""Find the first module whose output goes NaN/Inf during a forward pass.

    python rocm_tools/nan_locate.py -m /path/to/model [-p "Hello"]

Wraps every module's forward and reports the first one whose output is not
finite, plus the input range going in. A NaN has a single point of origin, so
this replaces feature-by-feature bisecting with one run that names the module.

Prints a few finite modules before the failure too, so you can see whether the
values were already drifting (overflow) or whether the module produced NaN from
clean input (uninitialised memory, bad rsqrt, out-of-bounds read).
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from exllamav3 import Config, Model, Cache, Tokenizer


def stat(t):
    if not isinstance(t, torch.Tensor):
        return "(not a tensor)"
    f = t.float()
    finite = torch.isfinite(f)
    n_nan = int((~finite).sum())
    if n_nan:
        return f"shape={tuple(t.shape)} NONFINITE {n_nan}/{f.numel()}"
    return (f"shape={tuple(t.shape)} min={f.min():+.4g} max={f.max():+.4g} "
            f"absmax={f.abs().max():.4g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-p", "--prompt", default="Hello")
    ap.add_argument("-cs", "--cache_size", type=int, default=32768,
                    help="must be >= the loader chunk size, not just the prompt")
    ap.add_argument("-n", "--show", type=int, default=6, help="finite modules to show before the fault")
    args = ap.parse_args()

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

    ids = tok.encode(args.prompt, encode_special_tokens=True)
    print(f" -- prompt {args.prompt!r} -> {ids.shape[-1]} tokens")

    history = []
    first_bad = {"hit": False}

    def wrap(mod):
        orig = mod.forward
        def fwd(x, params, *a, **kw):
            # Clone: modules mutate the residual stream in place, so holding a
            # reference and stat-ing it after the call reports the POST-call
            # state -- which made every input look already-NaN.
            xin = x.detach().clone() if isinstance(x, torch.Tensor) else None
            y = orig(x, params, *a, **kw)
            t = y[0] if isinstance(y, tuple) else y
            if isinstance(t, torch.Tensor) and not first_bad["hit"]:
                if not torch.isfinite(t.float()).all():
                    first_bad["hit"] = True
                    print("\n" + "=" * 78)
                    print(f"FIRST NON-FINITE OUTPUT: {mod.__class__.__name__}  key={getattr(mod,'key','?')}")
                    print(f"  input : {stat(xin)}")
                    print(f"  output: {stat(t)}")
                    print("=" * 78)
                    print(f"\nlast {args.show} finite modules before it:")
                    for line in history[-args.show:]:
                        print("   " + line)
                else:
                    history.append(f"{mod.__class__.__name__:24} {getattr(mod,'key','?'):40} {stat(t)}")
            return y
        mod.forward = fwd

    def wrap_rec(m, depth=0):
        wrap(m)
        for c in getattr(m, "modules", []) or []:
            wrap_rec(c, depth + 1)

    for m in model.modules:
        wrap_rec(m)

    with torch.inference_mode():
        # flash_attn_nc is the cacheless mode -- passing "cache" trips an assert.
        # A cacheless forward is what we want here anyway: it exercises every
        # module once with no paged-cache state to go wrong.
        model.prefill(input_ids=ids, params={"attn_mode": "flash_attn_nc"})

    if not first_bad["hit"]:
        print("\nno non-finite module output during prefill.")
        print(f"last {args.show} modules:")
        for line in history[-args.show:]:
            print("   " + line)
        print("\n-> prefill is clean; the fault is in decode (or in the logits head).")


if __name__ == "__main__":
    main()
