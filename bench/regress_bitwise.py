#!/usr/bin/env python3
"""Regression-suite wrapper around rocm_tools/decode_bitwise.py's method: one model load, several prompts.

Same prompt, same route as decode_bitwise.py (Config/Model/Cache(4096)/Generator, ArgmaxSampler,
return_logits, 48 new tokens), so references are interchangeable with that tool's. Variants:
  short   decode_bitwise's prompt once (~21 tokens: prefill on the small-row paths)
  long    the prompt x40 (~800 tokens: prefill also on the large-row R > 32 paths)

    regress_bitwise.py -m MODEL --variants short long --save    DIR --tag ds4 --json out.json
    regress_bitwise.py -m MODEL --variants short long --compare DIR --tag ds4 --json out.json

--save writes DIR/<tag>_<variant>.pt; --compare reads them. The JSON holds, per variant:
status identical | differs | missing, steps, first differing step, max abs diff, tokens identical,
and the first 120 chars of the decoded text. Exit 0 if every variant is identical (or saved).
Exits via os._exit() (native teardown segfault after model load, see RDNA_NOTES).
"""

import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "rocm_tools"))

import torch  # noqa: E402
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job  # noqa: E402
from exllamav3.generator.sampler import ArgmaxSampler  # noqa: E402
from decode_bitwise import PROMPT  # noqa: E402  (the one prompt definition)

REPEAT = {"short": 1, "long": 40}


def decode(generator, tokenizer, ids, n):
    job = Job(input_ids=ids, max_new_tokens=n, sampler=ArgmaxSampler(), return_logits=True)
    generator.enqueue(job)
    logits, tokens = [], []
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming":
                if "logits" in r:
                    logits.append(r["logits"].reshape(-1, r["logits"].shape[-1]).cpu())
                if "token_ids" in r:
                    tokens.append(r["token_ids"].reshape(-1).cpu())
    return torch.cat(logits, dim=0), torch.cat(tokens, dim=0)


def compare(R, RT, L, T):
    n = min(R.shape[0], L.shape[0])
    out = {"steps": int(L.shape[0]), "ref_steps": int(R.shape[0]),
           "tokens_identical": bool(RT.shape == T.shape and torch.equal(RT, T))}
    if R.shape != L.shape or R.dtype != L.dtype:
        out.update(status="differs", reason=f"shape/dtype ref {tuple(R.shape)} {R.dtype} vs {tuple(L.shape)} {L.dtype}")
        if R.shape[1:] != L.shape[1:] or R.dtype != L.dtype:
            return out
    it = torch.int32 if L.dtype == torch.float else torch.int16
    if torch.equal(R[:n].view(it), L[:n].view(it)) and "status" not in out:
        out["status"] = "identical"
        return out
    out["status"] = "differs"
    eq = R[:n] == L[:n]   # equal elements count as 0 (masked vocab rows hold -inf)
    diff = torch.where(eq, torch.zeros_like(R[:n].float()), (R[:n].float() - L[:n].float()).abs())
    diff = torch.nan_to_num(diff, nan=float("inf"))
    steps = (diff.view(n, -1).max(dim=1).values > 0).nonzero().view(-1)
    out.update(n_steps_differ=int(steps.numel()), first_step=int(steps[0]) if steps.numel() else -1,
               max_abs_diff=float(diff.max().item()))
    return out


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("--variants", nargs="+", default=["short"], choices=sorted(REPEAT))
    ap.add_argument("-n", "--new_tokens", type=int, default=48)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--save", metavar="DIR")
    ap.add_argument("--compare", metavar="DIR")
    ap.add_argument("--json", required=True)
    args = ap.parse_args()
    if bool(args.save) == bool(args.compare):
        ap.error("exactly one of --save / --compare")

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens=4096)
    model.load(progressbar=False)
    generator = Generator(model=model, cache=cache, tokenizer=tokenizer)

    res, ok = {}, True
    for v in args.variants:
        ids = tokenizer.encode(PROMPT * REPEAT[v], add_bos=True)
        L, T = decode(generator, tokenizer, ids, args.new_tokens)
        text = tokenizer.decode(T.view(1, -1))[0]
        r = {"prompt_tokens": int(ids.shape[-1]), "text": text[:120]}
        if args.save:
            os.makedirs(args.save, exist_ok=True)
            p = os.path.join(args.save, f"{args.tag}_{v}.pt")
            torch.save({"logits": L, "tokens": T, "prompt_tokens": ids.shape[-1]}, p)
            r.update(status="saved", file=p, steps=int(L.shape[0]))
        else:
            p = os.path.join(args.compare, f"{args.tag}_{v}.pt")
            if not os.path.exists(p):
                r.update(status="missing", file=p)
            else:
                ref = torch.load(p)
                r.update(compare(ref["logits"], ref["tokens"], L, T))
        ok &= r["status"] in ("saved", "identical")
        res[v] = r
        print(f"  {args.tag}/{v}: {r['status']}  prompt {r['prompt_tokens']} tok  "
              + (f"first step {r.get('first_step')} max|d| {r.get('max_abs_diff'):.3e} tokens same {r.get('tokens_identical')}"
                 if r["status"] == "differs" else "") + f"  {text[:80]!r}", flush=True)
    with open(args.json, "w") as f:
        json.dump(res, f, indent=1)
    print("BITWISE " + ("PASS" if ok else "FAIL"), flush=True)
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
