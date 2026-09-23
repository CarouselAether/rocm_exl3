#!/usr/bin/env python3
"""Decode throughput and coherence with and without MTP drafting, one model load.

MTP (the model's own next-token head as draft) turns decode into draft + verify:
a verify step runs the target model on num_draft_tokens + 1 rows, so this is
also the m > 1 exercise of the m == 1-only GEMV fast paths' fallbacks -- the
cooperative GEMM and the bszN MoE route at m <= 8. Reported per num_draft_tokens
(-ndt): tok/s, acceptance, and the generated text for eyeballing.

    rocm_tools/bench_mtp.py -m /path/to/model                 # plain, then -ndt 2
    rocm_tools/bench_mtp.py -m /path/to/model -ndt 3 2 1 -n 256

Plain figures follow bench_model.py (random 512-token prompt, 128 new tokens,
median of repeats) for continuity with recorded numbers, plus a natural-prompt
greedy run so the MTP comparison is like for like (acceptance depends on the
text). Exits via os._exit() like the other tools (native teardown segfault).
"""

import argparse
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import DefaultSampler, ArgmaxSampler

PROMPTS = [
    "Q: Briefly explain why the sky is blue, then name three primary colors.\nA:",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.\n\n",
    "Explain, step by step, how to compute the greatest common divisor of two integers, with an example.\n\n",
]


STOP_IDS = set()


def run_job(generator, ids, n, sampler, seed=1234):
    # Stop on every EOS id the model declares (config.eos_token_id_list), not just the
    # tokenizer's one -- a probe that stops only on the latter runs past the answer
    job = Job(input_ids=ids, max_new_tokens=n, sampler=sampler, seed=seed,
              stop_conditions=list(STOP_IDS) or None)
    generator.enqueue(job)
    text, final = [], None
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming":
                text.append(r.get("text", ""))
                if r.get("eos", False):
                    final = r
    return "".join(text), final


def rate(final):
    return final["new_tokens"] / final["time_generate"] if final["time_generate"] else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required=True)
    ap.add_argument("-ndt", "--num_draft_tokens", type=int, nargs="+", default=[2])
    ap.add_argument("-n", "--new_tokens", type=int, default=256)
    ap.add_argument("-r", "--repeats", type=int, default=3)
    ap.add_argument("--no_plain", action="store_true")
    args = ap.parse_args()

    label = os.path.basename(args.model_dir.rstrip("/"))
    print(f" -- loading {label} (+ MTP head)", flush=True)
    t0 = time.perf_counter()
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    draft_model = Model.from_config(config, component="mtp")
    tokenizer = Tokenizer.from_config(config)
    # max_history mirrors model_init: recurrent-state models (gated delta net) keep
    # num_draft_tokens + 1 snapshots so a rejected draft can roll back
    max_history = max(args.num_draft_tokens + [4])
    cache = Cache(model, max_num_tokens=4096, max_history=max_history)
    draft_cache = Cache(draft_model, max_num_tokens=4096)
    model.load(progressbar=False)
    draft_model.load(progressbar=False)
    print(f" -- loaded in {time.perf_counter() - t0:.1f}s; draft caps: "
          f"mtp_draft={draft_model.caps.get('mtp_draft')} default_draft_size={draft_model.caps.get('default_draft_size')}",
          flush=True)

    global STOP_IDS
    STOP_IDS = {tokenizer.eos_token_id} | {t for t in (config.eos_token_id_list or []) if t is not None}
    STOP_IDS.discard(None)
    rng = torch.Generator().manual_seed(7)
    vocab = tokenizer.actual_vocab_size

    def rand_ids(n):
        return torch.randint(int(vocab * .05), int(vocab * .95), (1, n), dtype=torch.long, generator=rng)

    if not args.no_plain:
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer)
        text, _ = run_job(gen, tokenizer.encode(PROMPTS[0], add_bos=True), 150, DefaultSampler())
        print(f"\n=== plain: sampled coherence ===\n{text.strip()[:600]}\n", flush=True)
        rates = []
        for i in range(args.repeats + 1):
            _, f = run_job(gen, rand_ids(512), 128, DefaultSampler())
            if i: rates.append(rate(f))
        print(f"=== plain: decode128 after random-512 prefill: median {statistics.median(rates):.2f} tok/s "
              f"(spread {(max(rates) - min(rates)) / statistics.median(rates):.1%})", flush=True)
        rates = []
        for p in PROMPTS:
            _, f = run_job(gen, tokenizer.encode(p, add_bos=True), args.new_tokens, ArgmaxSampler())
            rates.append(rate(f))
        print(f"=== plain: greedy {args.new_tokens} tokens, natural prompts: "
              f"median {statistics.median(rates):.2f} tok/s ({', '.join(f'{r:.1f}' for r in rates)})", flush=True)
        del gen

    for ndt in args.num_draft_tokens:
        # Caches must exist before model.load(), so every generator reuses the one cache
        gen = Generator(model=model, cache=cache, tokenizer=tokenizer,
                        draft_model=draft_model, draft_cache=draft_cache, num_draft_tokens=ndt)
        route = "mtp_draft" if gen.mtp_draft else ("dflash" if gen.dflash_draft else "draft-model (MTP head as a separate draft model)")
        print(f"\n -- ndt={ndt}: draft route = {route}", flush=True)
        rates, accs = [], []
        first_text = None
        for p in PROMPTS:
            text, f = run_job(gen, tokenizer.encode(p, add_bos=True), args.new_tokens, ArgmaxSampler())
            a, rj = f.get("accepted_draft_tokens", 0), f.get("rejected_draft_tokens", 0)
            rates.append(rate(f))
            accs.append(a / (a + rj) if a + rj else 0.0)
            if first_text is None: first_text = text
        print(f"\n=== MTP ndt={ndt} (verify rows m={ndt + 1}): greedy {args.new_tokens} tokens: "
              f"median {statistics.median(rates):.2f} tok/s ({', '.join(f'{r:.1f}' for r in rates)}), "
              f"acceptance {statistics.median(accs):.1%} ({', '.join(f'{a:.0%}' for a in accs)})", flush=True)
        print(f"--- greedy text (prompt 1):\n{first_text.strip()[:500]}", flush=True)
        text, f = run_job(gen, tokenizer.encode(PROMPTS[0], add_bos=True), 150, DefaultSampler())
        a, rj = f.get("accepted_draft_tokens", 0), f.get("rejected_draft_tokens", 0)
        print(f"--- sampled coherence (ndt={ndt}, {rate(f):.1f} tok/s, acc {a / max(a + rj, 1):.0%}):\n{text.strip()[:600]}\n", flush=True)
        del gen

    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
