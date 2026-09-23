#!/usr/bin/env python3
"""Multi-turn chat coherence probe: the acceptance test for HIP graph replay.

The 7.2.x failure that took graphs off (see rocm/graph_rdna.hip) was NOT
visible on a single prompt: six-turn chats on Laguna degenerated into babble
or loops by the final turn, once several generator jobs had captured and
replayed BC decode graphs against a growing context. So this drives exactly
that shape -- one conversation, each turn a NEW generator job carrying the
whole rendered history, context growing to a few thousand tokens -- and then
a final round with two conversations in flight at once (multi-job replay).

Stops on every end-of-turn id the model declares (as exl3_server does): a
client that stops only on the bare EOS makes a template-terminated model repeat
its finished answer verbatim, which looks exactly like replay corruption and
is not. Ordinary temperature sampling with a fixed seed per run (greedy loop probes are
decoding chaos, per RDNA_NOTES.md). A human still judges the text; the printed
repetition score is only a tripwire: the fraction of 4-grams in a reply that
are repeats of an earlier 4-gram in the same reply. Healthy prose sits well
under 0.15; a graph-replay collapse reads as long verbatim loops and scores
above 0.4.

    rocm_tools/chat_probe.py -m /path/to/model [-t 6] [-n 256] [-s 1234 5678]

Exits via os._exit(): model-loading processes segfault in native teardown.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job
from exllamav3.generator.sampler import DefaultSampler

TURNS = [
    "Explain, in a few paragraphs, how a refrigerator keeps food cold.",
    "Now compare that with how an air conditioner works. What is shared and what differs?",
    "Give me a numbered list of five common failure modes of refrigerators, with one sentence each on diagnosis.",
    "Write a short story, about 200 words, about a repair technician who finds something unexpected inside a fridge.",
    "Summarise everything we have discussed so far in one paragraph.",
    "Finally, write a haiku about cold air, then explain the joke in it if there is one.",
]

SECOND = [
    "List the planets of the solar system in order and give one fact about each.",
    "Which of those would be the hardest to send a probe to, and why?",
]


def rep_score(text: str, n: int = 4) -> float:
    w = text.split()
    if len(w) < n * 3:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    seen, rep = set(), 0
    for g in grams:
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / len(grams)


def render(tokenizer, messages):
    ids = tokenizer.hf_chat_template(messages, add_generation_prompt = True)
    if isinstance(ids, str):
        ids = tokenizer.encode(ids, add_bos = False)
    return ids


def run_jobs(generator, jobs):
    """Run jobs concurrently, return {job: text}."""
    out = {j: [] for j in jobs}
    for j in jobs:
        generator.enqueue(j)
    while generator.num_remaining_jobs():
        for r in generator.iterate():
            if r["stage"] == "streaming":
                out[r["job"]].append(r.get("text", ""))
    return {j: "".join(t) for j, t in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required = True)
    ap.add_argument("-t", "--turns", type = int, default = 6)
    ap.add_argument("-n", "--new_tokens", type = int, default = 256)
    ap.add_argument("-s", "--seeds", type = int, nargs = "+", default = [1234])
    ap.add_argument("-c", "--cache_size", type = int, default = 8192)
    ap.add_argument("--print_chars", type = int, default = 300)
    args = ap.parse_args()

    label = os.path.basename(args.model_dir.rstrip("/"))
    print(f" -- loading {label}", flush = True)
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    tokenizer = Tokenizer.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache_size)
    model.load(progressbar = False)
    generator = Generator(model = model, cache = cache, tokenizer = tokenizer)

    # Stop on every end-of-turn id the model declares, the way exl3_server does
    # (config.eos_token_id_list | tokenizer.eos_token_id). Without this a chat
    # model that ends turns with a template token (Laguna: </assistant>, id 24)
    # runs past its answer and repeats it verbatim -- which is a stop-token
    # bug in the CLIENT, not graph replay, and was mistaken for one once.
    stop_ids = set()
    for t in (getattr(config, "eos_token_id_list", None) or []):
        if t is not None:
            stop_ids.add(int(t))
    if tokenizer.eos_token_id is not None:
        stop_ids.add(int(tokenizer.eos_token_id))
    stop = sorted(stop_ids)
    print(f" -- stop ids: {stop}", flush = True)

    worst = 0.0
    for seed in args.seeds:
        print(f"\n===== seed {seed} =====", flush = True)
        messages = []
        for t, user in enumerate(TURNS[:args.turns], 1):
            messages.append({"role": "user", "content": user})
            ids = render(tokenizer, messages)
            job = Job(input_ids = ids, max_new_tokens = args.new_tokens,
                      sampler = DefaultSampler(), seed = seed + t, stop_conditions = stop)
            text = run_jobs(generator, [job])[job]
            messages.append({"role": "assistant", "content": text})
            rs = rep_score(text)
            worst = max(worst, rs)
            flag = "  <-- LOOPING?" if rs > 0.4 else ""
            print(f"--- turn {t}: ctx {ids.shape[-1]} tok, +{len(tokenizer.encode(text, add_bos=False)[0])} tok, "
                  f"rep4={rs:.2f}{flag}")
            print("   " + text[:args.print_chars].replace("\n", " ") + (" ..." if len(text) > args.print_chars else ""),
                  flush = True)

        # multi-job round: the running conversation plus a fresh one, in flight together
        m2 = [{"role": "user", "content": SECOND[0]}]
        ja = Job(input_ids = render(tokenizer, messages + [{"role": "user", "content": "One more: what temperature should a fridge be set to, and why?"}]),
                 max_new_tokens = args.new_tokens, sampler = DefaultSampler(), seed = seed + 100,
                 stop_conditions = stop)
        jb = Job(input_ids = render(tokenizer, m2), max_new_tokens = args.new_tokens,
                 sampler = DefaultSampler(), seed = seed + 200, stop_conditions = stop)
        res = run_jobs(generator, [ja, jb])
        for name, j in (("A (long ctx)", ja), ("B (fresh)", jb)):
            rs = rep_score(res[j]); worst = max(worst, rs)
            flag = "  <-- LOOPING?" if rs > 0.4 else ""
            print(f"--- concurrent {name}: rep4={rs:.2f}{flag}")
            print("   " + res[j][:args.print_chars].replace("\n", " ") + (" ..." if len(res[j]) > args.print_chars else ""),
                  flush = True)

    print(f"\n===== worst rep4 across run: {worst:.2f} ({'LOOPING' if worst > 0.4 else 'ok'}) =====", flush = True)
    os._exit(0)


if __name__ == "__main__":
    main()
