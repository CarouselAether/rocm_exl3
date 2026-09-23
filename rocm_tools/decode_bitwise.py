#!/usr/bin/env python3
"""Bitwise A/B of a whole-model greedy decode across a code change.

The kernel-level A/B (mgemv_bitwise.py) covers the multi-matrix path, but the
single-matrix graph GEMV path only runs inside the BC modules, which no Python
binding reaches directly. This records every decode step's logits on a fixed
prompt BEFORE a change and compares them bit for bit AFTER it, through the
exact route the model takes at bsz 1 -- the acceptance bar for a change whose
arithmetic is claimed to be identical (the launch-count fusions of 2026-09).

    python rocm_tools/decode_bitwise.py -m /path/to/model --save    ref.pt   # old build
    python rocm_tools/decode_bitwise.py -m /path/to/model --compare ref.pt   # new build

Greedy (ArgmaxSampler), so the token sequence is a function of the logits
alone; the prompt is prefilled (GEMM path) and the new tokens decode one at a
time (GEMV / mgemv paths). Any element differing in any step fails; the report
names the first differing step and the max difference, so a genuine ulp
change is distinguishable from a wrong result. Exits nonzero on a difference.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

PROMPT = (
    "Q: Briefly explain why the sky is blue, then name three primary colors.\nA:"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-n", "--new_tokens", type=int, default=48)
    ap.add_argument("--save", metavar="FILE")
    ap.add_argument("--compare", metavar="FILE")
    args = ap.parse_args()
    if bool(args.save) == bool(args.compare):
        ap.error("exactly one of --save / --compare")

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    ids = tokenizer.encode(PROMPT, add_bos=True)
    job = Job(input_ids=ids, max_new_tokens=args.new_tokens,
              sampler=ArgmaxSampler(), return_logits=True)
    generator.enqueue(job)
    logits, tokens = [], []
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming":
                if "logits" in r:
                    logits.append(r["logits"].reshape(-1, r["logits"].shape[-1]).cpu())
                if "token_ids" in r:
                    tokens.append(r["token_ids"].reshape(-1).cpu())
    L = torch.cat(logits, dim=0)
    T = torch.cat(tokens, dim=0)
    text = tokenizer.decode(T.view(1, -1))[0]
    print(f"  {L.shape[0]} steps, logits {tuple(L.shape)} {L.dtype}")
    print(f"  text: {text[:200]!r}")

    ok = True
    if args.save:
        torch.save({"logits": L, "tokens": T}, args.save)
        print(f"reference written to {args.save}")
    else:
        ref = torch.load(args.compare)
        R, RT = ref["logits"], ref["tokens"]
        n = min(R.shape[0], L.shape[0])
        if R.shape != L.shape or R.dtype != L.dtype:
            print(f"  shape/dtype differ: ref {tuple(R.shape)} {R.dtype} vs {tuple(L.shape)} {L.dtype}")
            ok = False
        it = torch.int32 if L.dtype == torch.float else torch.int16
        same = torch.equal(R[:n].view(it), L[:n].view(it))
        if same and ok:
            print(f"  ok   all {n} steps bit-identical; tokens identical: "
                  f"{torch.equal(RT[:n], T[:n])}")
        else:
            ok = False
            diff = (R[:n].float() - L[:n].float()).abs()
            steps = (diff.view(n, -1).max(dim=1).values > 0).nonzero().view(-1)
            first = int(steps[0].item()) if steps.numel() else -1
            print(f"  FAIL {steps.numel()} of {n} steps differ; first at step {first}; "
                  f"max abs diff {diff.max().item():.3e}; tokens identical: "
                  f"{torch.equal(RT[:n], T[:n])}")
        print("BITWISE " + ("PASS" if ok else "FAIL"))
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
