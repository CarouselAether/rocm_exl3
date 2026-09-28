#!/usr/bin/env python3
"""Greedy-decode agreement A/B across a numerics-changing kernel swap.

decode_bitwise.py demands bit-identical logits; a kernel that changes accumulation order
(e.g. the DSA decode split kernel, EXL3_ROCM_DSA_DECODE) cannot pass that bar, so this
reports how close the two builds stay instead. Per fixed prompt: greedy-decode N tokens
through the model's normal bsz-1 route (prefill, then graphed decode), recording every
step's logits and token.

    EXL3_ROCM_DSA_DECODE=0 python rocm_tools/decode_agree.py -m MODEL --save ref.pt
    python rocm_tools/decode_agree.py -m MODEL --compare ref.pt

Report per prompt:
  prefix   tokens identical up to the first divergence (greedy runs cannot be compared
           past it: the contexts differ from then on)
  agree    fraction of identical tokens over all N (informational only)
  maxdiff  max |logit_a - logit_b| over the steps whose context is identical (steps
           0..first divergence inclusive), and that over max |logit|
  top10    max |logit diff| over the reference's top-10 tokens; KL(ref || new) of the
           next-token distributions (mean / max over the common-context steps)
  top2gap  at the divergence step, the reference's top-1 minus top-2 logit: a near-tie
           flipping is expected noise, a large gap flipping is a real difference

--cq N runs with the quantized cache (packed DSA pools, QC > 0); --batch decodes all
prompts concurrently (BCDsaBatch, MULTIROW kernels). The third prompt is long (~3K tokens) so compressed-attention layers decode in the
indexer top-k regime, not just the dense-pool one.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, CacheLayer_quant, Tokenizer, Generator, Job
from exllamav3.generator.sampler import ArgmaxSampler

PROMPTS = [
    "Q: Briefly explain why the sky is blue, then name three primary colors.\nA:",
    "Write a short story about a lighthouse keeper who discovers a message in a bottle. "
    "The story should have a clear beginning, middle and end.\n\n",
]


def long_prompt(tokenizer, n_tokens = 3000):
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    text = ""
    for f in ["README.md", "doc/exl3.md", "doc/convert.md", "PROFILE.md"]:
        p = os.path.join(here, f)
        if os.path.exists(p):
            text += open(p).read() + "\n\n"
    ids = tokenizer.encode(text)[0, :n_tokens]
    body = tokenizer.decode(ids.view(1, -1))[0]
    return body + "\n\nSummarize the text above in five bullet points:\n"


def run(generator, tokenizer, prompts, n):
    """Greedy-decode every prompt; all enqueued at once (batched decode) when several."""
    plens = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt, add_bos = True)
        plens.append(ids.shape[-1])
        generator.enqueue(Job(input_ids = ids, max_new_tokens = n, sampler = ArgmaxSampler(),
                              return_logits = True, stop_conditions = [], identifier = i))
    logits = [[] for _ in prompts]
    tokens = [[] for _ in prompts]
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming":
                i = r["identifier"]
                if "logits" in r:
                    logits[i].append(r["logits"].reshape(-1, r["logits"].shape[-1]).float().cpu())
                if "token_ids" in r:
                    tokens[i].append(r["token_ids"].reshape(-1).cpu())
    return [(torch.cat(l, 0), torch.cat(t, 0), pl) for l, t, pl in zip(logits, tokens, plens)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required = True)
    ap.add_argument("-n", "--new_tokens", type = int, default = 256)
    ap.add_argument("--save", metavar = "FILE")
    ap.add_argument("--compare", metavar = "FILE")
    ap.add_argument("--cq", type = int, default = 0, help = "quantized cache bits (server -cq)")
    ap.add_argument("--batch", action = "store_true",
                    help = "enqueue all prompts at once (batched MULTIROW decode graphs)")
    args = ap.parse_args()
    if bool(args.save) == bool(args.compare):
        ap.error("exactly one of --save / --compare")

    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cap = 16384 if args.batch else 8192      # three concurrent jobs need ~10K tokens
    if args.cq:
        cache = Cache(model, max_num_tokens = cap, layer_type = CacheLayer_quant,
                      k_bits = args.cq, v_bits = args.cq)
    else:
        cache = Cache(model, max_num_tokens = cap)
    model.load(progressbar = False)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)

    prompts = PROMPTS + [long_prompt(tokenizer)]
    results = []
    if args.batch:
        outs = run(generator, tokenizer, prompts, args.new_tokens)
    else:
        outs = [run(generator, tokenizer, [p], args.new_tokens)[0] for p in prompts]
    for i, (L, T, plen) in enumerate(outs):
        text = tokenizer.decode(T.view(1, -1))[0]
        print(f"  prompt {i}: {plen} prompt tokens, {T.shape[0]} new; text: {text[:120]!r}")
        results.append({"logits": L, "tokens": T})

    ok = True
    if args.save:
        torch.save(results, args.save)
        print(f"reference written to {args.save}")
    else:
        ref = torch.load(args.compare)
        for i, (a, b) in enumerate(zip(ref, results)):
            RT, T, R, L = a["tokens"], b["tokens"], a["logits"], b["logits"]
            n = min(RT.shape[0], T.shape[0])
            neq = (RT[:n] != T[:n]).nonzero().view(-1)
            first = int(neq[0]) if neq.numel() else n
            k = min(first + 1, n, R.shape[0], L.shape[0])
            d = (R[:k] - L[:k]).abs()
            maxdiff = d.max().item()
            rel = maxdiff / max(R[:k].abs().max().item(), 1e-6)
            gap = ""
            if first < n:
                t2 = R[first].topk(2).values
                gap = f"; divergence top2gap(ref) {float(t2[0] - t2[1]):.3f}"
            agree = (RT[:n] == T[:n]).float().mean().item()
            # Distribution-level view: the all-vocab max is dominated by far-tail logits
            lp_r = torch.log_softmax(R[:k], -1)
            lp_n = torch.log_softmax(L[:k], -1)
            kl = (lp_r.exp() * (lp_r - lp_n)).sum(-1)
            top = R[:k].topk(10, dim = -1).indices
            dtop = (R[:k].gather(1, top) - L[:k].gather(1, top)).abs().max().item()
            print(f"  prompt {i}: prefix {first}/{n}  agree {agree:.3f}  maxdiff {maxdiff:.3e} "
                  f"(rel {rel:.2e}) top10 maxdiff {dtop:.3e}  KL mean {kl.mean().item():.2e} "
                  f"max {kl.max().item():.2e}  over {k} common-context steps{gap}")
    sys.stdout.flush()
    os._exit(0 if ok else 1)


if __name__ == "__main__":
    main()
