# exl3_server

A llama.cpp-server-style, OpenAI-compatible HTTP server for exllamav3. Single
file, no config — model loading and default sampling flags are the same ones
`examples/chat.py` uses (they come from `exllamav3.model_init.add_args`), plus a
handful of server flags.

```sh
python rocm_tools/exl3_server/server.py -m ~/models/Laguna-S-2.1-exl3-4.00bpw -cs 32768
# serves on http://127.0.0.1:3953
# python server.py --help lists every flag with its default (except -cs: its help
# shows 32768, but the real default is the model's max context)
```

Dependencies (fastapi, uvicorn, sse-starlette, transformers, jinja2) come with
`pip install -r requirements_rocm.txt`.

The complete flag reference, every flag with what it does, is in the main
[README's Server section](../../README.md#server). In short, all chat.py loader/sampler flags work: `-gs`, `-cs`, `-cq`, `-tp`,
`-mcl/-mcs/-mct`, draft model flags (`-dm`, `-ndt`, `-dds`, `-ngram`, `-mtp`),
`-temp/-minp/-topk/-topp/-repp/-presp/-freqp/-penr`, etc. CLI sampling values
are the *defaults*; each request can override them.

Speculative decoding works like llama.cpp's `--model-draft`: `-dm <dir>` loads
a separate draft model — including **DFlash / EAGLE-3-style drafters**
(exllamav3 has dedicated `DFlashDraftModel` and `DFlashLagunaForCausalLM`
architectures; `-dm ~/models/Laguna-S-2.1-DFlash` loads Poolside's BF16
drafter directly, no quantization needed) — `-mtp` uses the model's own MTP
head (DeepSeek V4 etc.), `-ngram 2` drafts from repeats in the context with no
extra model, and `-dds` skips drafting while the acceptance rate is low. Draft acceptance shows
up in the server log and in the native `/completion` `timings` as
`draft_n`/`draft_n_accepted`.

Measured on gfx1151: with DS4-Flash 2.04bpw (baseline 15.5 t/s), `-mtp` = 13.7
t/s (net loss — DS4's MTP head is known-bad at low bpw, ~70% acceptance) and
`-ngram 2 -dds` = ~14.6 t/s novel / 21.3 t/s (+37%) repetitive — the
recommended cheap default for chat workloads. With Laguna-S 4bpw (baseline
23.2 t/s), the Laguna DFlash drafter reaches good acceptance (~8-9 of 15 per
block) but lands at 21-24 t/s steady with an 8-11 t/s dip on cold context —
parity at best, because batched verify passes on RDNA cost nearly as much as
the highly-optimized single-token decode they replace (`-ndt 8` truncation
makes it worse; the block drafter wants its full window). Verdict: on this
GPU, use `-ngram`; skip `-mtp`/DFlash until batched decode (mgemv) gets
faster. Note: greedy (temp 0) output is not run-to-run deterministic on this
port — split-K atomics jitter logits at ULP level and near-ties flip — so
draft vs no-draft output equivalence can't be checked by string comparison.

Server flags:

| Flag | Meaning |
|---|---|
| `-host` / `-port` | bind address, default `127.0.0.1:3953` |
| `-cs` | cache size in tokens; **default = the model's max context** (long-context models advertise 256K-1M — pass `-cs`/`-cq` to keep the KV cache sane) |
| `-key` | require an API key (`Authorization: Bearer` or `x-api-key`) |
| `-smn` | model name reported by the API (default: model dir name) |
| `-maxr` | server-side cap on response tokens (default: fill remaining context) |
| `-ctk` | default chat-template kwargs as JSON, e.g. `'{"enable_thinking": false}'` |
| `-lw` / `-lmr` | loop-detection stop (off by default) |
| `-pcs N` | prefill chunk size (Generator `max_chunk_size`, default 2048) |
| `-nwu` (alias of model_init's `-nw`) | skip the startup warmups (v1.5.3 `model.warmup()` and the server's two-job Generator warmup) |
| `-ngl` | n-gram table (PLE models, e.g. Qwen3.8-Flash-Next) in RAM like `-ngr`, **and locked** there (`mlock`): never swapped out or reclaimed |

### N-gram table: disk, RAM or locked RAM

PLE models (Qwen3.8-Flash-Next) carry a ~36 GiB hashed n-gram embedding table. You choose where it lives:

- **default: streamed from disk.** Each forward reads only the rows it needs; hot rows stay in the page cache.
  Uses no RAM up front. On a quiet 128 GB box it measured the same speed as `-ngr`.
- **`-ngr`: in RAM.** The table is loaded into ordinary process memory, so there are no disk reads. With swap on, the
  kernel may still swap parts of it out under memory pressure.
- **`-ngl`: in RAM, locked.** The table is loaded as with `-ngr` (one copy), then `mlock`ed, so it cannot be swapped
  out or reclaimed while the server runs.

`-ngl` needs a locked-memory limit (`RLIMIT_MEMLOCK`) at least as large as the table. Check it with `ulimit -l`
(the value is in KiB; `unlimited` is fine). If it is too low, the server stops **before** loading the model and
prints how to raise it:

```sh
ulimit -l unlimited && python rocm_tools/exl3_server/server.py -m ~/models/Qwen3.8-Flash-Next-... -cs 65536 -ngl
# the hard limit must allow it: /etc/security/limits.conf ->   <user>  -  memlock  unlimited   (log in again)
# systemd service:            LimitMEMLOCK=infinity
# or, as root:                prlimit --pid <pid> --memlock=unlimited:unlimited
```

It also refuses when model weights + table + KV cache + 8 GiB headroom (`EXL3_NGRAM_LOCK_HEADROOM_GB`) do not fit
in available RAM, because a locked table can never be given back. When the lock succeeds, the log shows
`n-gram table locked in RAM: 36.4 GiB ... VmLck ...`, `/props` shows `"ngram_table": "ram_locked"`, and
`grep VmLck /proc/<pid>/status` shows the locked amount.

## Endpoints

- `POST /v1/chat/completions` — prompt is built with the **model's own chat
  template** (`tokenizer_config.json`, rendered by HF `apply_chat_template`).
  Streaming and non-streaming, `n > 1` (non-streaming only), `stop`, `logit_bias`, `seed`, `tools`
  (passed to the template), `chat_template_kwargs`, `continue_final_message`.
- `POST /v1/completions` — raw prompt used **verbatim** (special tokens are
  encoded), so the client's own instruct template applies. Extensions:
  `add_bos` (default true), `parse_special` (default true).
- `POST /completion` (alias `/completions`) — **llama.cpp-native** endpoint for
  clients using ST's *llama.cpp* preset and similar tools. Native param names
  (`n_predict`, `repeat_penalty`, `repeat_last_n`, `ignore_eos`, pair-style
  `logit_bias`, `return_tokens`) and native response shape (`content`, `stop`,
  `stop_type`/`stopping_word`, `timings`, no `[DONE]` terminator). Native DRY
  and XTC fields are honored; the remaining unsupported native samplers
  (mirostat, dynatemp, typical_p, grammar) are accepted and ignored.
- `POST /apply-template` — render the model's chat template without generating;
  returns `{"prompt": ...}`. Handy for debugging what the model actually sees.
- `GET /v1/models`, `GET /health`, `GET /props` (includes `chat_template`),
  `POST /tokenize`, `POST /detokenize`.

Sampling fields honored per request (all completion endpoints): `temperature`,
`top_p`, `top_k`, `min_p`, `frequency_penalty`, `presence_penalty`,
`repetition_penalty`, `penalty_range`, `logit_bias`, `seed`, **XTC**
(`xtc_probability`, `xtc_threshold`) and **DRY** (`dry_multiplier`, `dry_base`,
`dry_allowed_length`, `dry_penalty_last_n`, `dry_sequence_breakers`) with
llama.cpp semantics, plus exl3 extras `banned_strings` and
`decode_special_tokens`. Unknown fields are ignored, so any OpenAI-ish client
works. XTC uses exl3's built-in `SS_XTC`; DRY is implemented in
[dry_sampler.py](dry_sampler.py) as a custom sampler step (whole-context scan
by default; matches stop at sequence breakers). CLI defaults:
`-xtcp/-xtct/-drym/-dryb/-dryal/-dryln`. When neither XTC nor DRY is active a
request uses the stock fused sampler path.

Concurrent requests are batched transparently by the dynamic generator.
Disconnecting a client (e.g. SillyTavern's stop button) cancels its job: every
generation loop actively polls `request.is_disconnected()` (TabbyAPI's
pattern), because passive disconnect detection is unreliable for POST + SSE
under uvicorn's ASGI >= 2.4 flow control. Cancellations are logged as
`-- Client disconnected, job cancelled`.

## SillyTavern

Two ways to connect:

- **Chat Completion** (server-side template): API = *Chat Completion*, source
  *Custom (OpenAI-compatible)*, endpoint `http://127.0.0.1:3953/v1`. The model's
  own instruct template is applied by the server.
- **Text Completion** (ST's template overrides the model's): API = *Text
  Completion*, type *Generic (OpenAI-compatible)*, endpoint
  `http://127.0.0.1:3953/v1`. ST formats the prompt with its Advanced
  Formatting / instruct template and the server encodes it verbatim, special
  tokens included. The *llama.cpp* type also works (it uses the native
  `/completion` endpoint).

## Notes

- If the model has no chat template, `/v1/chat/completions` returns 400 (a
  warning is printed at startup) and `/v1/completions` still works.
- A prompt that can't fit the cache alongside at least one response token is
  rejected with 400 — there is no silent context truncation.
- If a Text Completion prompt already starts with BOS (some ST instruct
  templates include it), the server won't add a second one.
- Penalty range defaults to the CLI `-penr` value (1024), not the full context;
  unbounded OAI-style penalties over long contexts are exactly what caused the
  ~8K coherency cliff under TabbyAPI.
- Shutdown (Ctrl-C) takes a few seconds; the process hard-exits after uvicorn
  stops to avoid the known interpreter-teardown segfault.
