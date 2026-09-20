#!/usr/bin/env python3
"""
exl3_server - a llama.cpp-style OpenAI-compatible HTTP server for exllamav3.

Model loading / sampling flags mirror examples/chat.py (they come from
exllamav3.model_init.add_args), so anything you pass to chat.py works here too.

Endpoints:
    GET  /health                  liveness probe
    GET  /props                   model name, context size, chat template info
    GET  /v1/models               OpenAI model list
    POST /v1/chat/completions     OpenAI chat API; prompt built with the model's
                                  own HF chat template (tokenizer_config.json)
    POST /v1/completions          OpenAI text-completion API; the prompt is used
                                  verbatim (special tokens encoded), so clients
                                  like SillyTavern in Text Completion mode can
                                  apply their own instruct template instead
    POST /tokenize                {"content": str} -> {"tokens": [...]}
    POST /detokenize              {"tokens": [...]} -> {"content": str}

Streaming uses standard OpenAI SSE chunks, terminated with "data: [DONE]".
"""

import sys, os
from pathlib import Path

# Prefer an installed exllamav3; fall back to the repo this script lives in
# (rocm_tools/exl3_server -> repo root), then to sibling checkouts
try:
    import exllamav3  # noqa: F401
except ImportError:
    here = Path(__file__).resolve()
    candidates = [here.parents[2]] + [here.parents[2].parent / d for d in ("rocm_exl3", "exllamav3")]
    for p in candidates:
        if (p / "exllamav3" / "__init__.py").exists():
            sys.path.insert(0, str(p))
            break
    import exllamav3  # noqa: F401

import argparse
import asyncio
import json
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from sse_starlette.sse import EventSourceResponse


class CancellingStreamResponse(EventSourceResponse):
    """EventSourceResponse that guarantees the body iterator is closed when streaming
    stops for any reason. Under ASGI spec >= 2.4 a client disconnect surfaces as an
    OSError from send() in *this* class's frame -- the suspended generator would
    otherwise be abandoned un-closed and its job would keep generating until GC
    (issue seen with SillyTavern's stop button). aclose() forces the generator's
    finally block, which cancels the job."""
    async def _stream_response(self, send) -> None:
        try:
            await super()._stream_response(send)
        finally:
            aclose = getattr(self.body_iterator, "aclose", None)
            if aclose is not None:
                await aclose()
from pydantic import BaseModel, ConfigDict, Field

from exllamav3 import AsyncGenerator, AsyncJob, model_init
from exllamav3.constants import PAGE_SIZE
from exllamav3.generator.sampler import (
    ComboSampler, CustomSampler, SS_LogitBias, SS_RepP, SS_PresFreqP, SS_Argmax,
    SS_Temperature, SS_MinP, SS_TopK, SS_TopP, SS_XTC, SS_AdaptiveP, SS_Sample,
)
from dry_sampler import SS_DRY, breaker_token_ids

DEFAULT_DRY_BREAKERS = ("\n", ":", "\"", "*")


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    args: argparse.Namespace = None
    model = None
    config = None
    cache = None
    tokenizer = None
    draft_model = None
    draft_cache = None
    generator: AsyncGenerator | None = None
    model_name: str = "exl3-model"
    context_length: int = 0
    stop_token_ids: list[int] = []
    has_chat_template: bool = False
    default_template_kwargs: dict = {}
    lock = None  # asyncio.Lock for load-time init

state = ServerState()


# ---------------------------------------------------------------------------
# Request schemas (extra fields are ignored so any OpenAI-ish client works)
# ---------------------------------------------------------------------------

class SamplingFields(BaseModel):
    model_config = ConfigDict(extra = "ignore")
    temperature: float | None = None
    top_p: float | None = None
    top_k: int | None = None
    min_p: float | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None   # common extension (ST, vLLM, ...)
    penalty_range: int | None = None          # extension: sustain/decay range in tokens
    logit_bias: dict[str, float] | None = None
    seed: int | None = None
    # llama.cpp-style extended samplers
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    dry_multiplier: float | None = None
    dry_base: float | None = None
    dry_allowed_length: int | None = None
    dry_penalty_last_n: int | None = None
    dry_sequence_breakers: list[str] | str | None = None
    # exllamav3 extensions
    banned_strings: list[str] | None = None
    decode_special_tokens: bool = False

class ChatCompletionRequest(SamplingFields):
    model: str | None = None
    messages: list[dict]
    max_tokens: int | None = None
    max_completion_tokens: int | None = None
    stream: bool = False
    stream_options: dict | None = None
    stop: str | list[str] | None = None
    n: int = 1
    tools: list[dict] | None = None
    # Template control extensions (llama.cpp / vLLM style)
    chat_template_kwargs: dict | None = None
    add_generation_prompt: bool = True
    continue_final_message: bool = False

class CompletionRequest(SamplingFields):
    model: str | None = None
    prompt: str | list = ""
    max_tokens: int | None = None
    stream: bool = False
    stream_options: dict | None = None
    stop: str | list[str] | None = None
    n: int = 1
    # Extensions for raw-prompt clients (SillyTavern Text Completion etc.)
    add_bos: bool = True
    parse_special: bool = True

class NativeCompletionRequest(BaseModel):
    """llama.cpp-native /completion request (subset). Unsupported native samplers
    (DRY, XTC, mirostat, dynatemp, typical_p, grammar) are ignored."""
    model_config = ConfigDict(extra = "ignore")
    prompt: str | list = ""
    n_predict: int = -1
    stream: bool = False
    stop: list[str] | None = None
    temperature: float | None = None
    top_k: int | None = None
    top_p: float | None = None
    min_p: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    ignore_eos: bool = False
    logit_bias: list | dict | None = None
    xtc_probability: float | None = None
    xtc_threshold: float | None = None
    dry_multiplier: float | None = None
    dry_base: float | None = None
    dry_allowed_length: int | None = None
    dry_penalty_last_n: int | None = None
    dry_sequence_breakers: list[str] | str | None = None
    return_tokens: bool = False
    cache_prompt: bool = True   # accepted for compatibility; exl3 dedups prefix pages anyway
    add_bos: bool = True
    parse_special: bool = True

class ApplyTemplateRequest(BaseModel):
    model_config = ConfigDict(extra = "ignore")
    messages: list[dict]
    add_generation_prompt: bool = True
    chat_template_kwargs: dict | None = None

class TokenizeRequest(BaseModel):
    model_config = ConfigDict(extra = "ignore")
    content: str = ""
    add_bos: bool = False
    parse_special: bool = True

class DetokenizeRequest(BaseModel):
    model_config = ConfigDict(extra = "ignore")
    tokens: list[int] = Field(default_factory = list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_sampler(req: SamplingFields):
    """Merge request-level sampling params over the CLI defaults."""
    a = state.args
    def pick(rv, av):
        return rv if rv is not None else av
    logit_bias = None
    if req.logit_bias:
        try:
            logit_bias = {int(k): float(v) for k, v in req.logit_bias.items()}
        except (ValueError, TypeError):
            raise HTTPException(400, "logit_bias keys must be token IDs")
    penalty_range = pick(req.penalty_range, a.penalty_range)
    temperature = pick(req.temperature, a.temperature)
    min_p = pick(req.min_p, a.min_p)
    top_k = pick(req.top_k, a.top_k)
    top_p = pick(req.top_p, a.top_p)
    rep_p = pick(req.repetition_penalty, a.repetition_penalty)
    pres_p = pick(req.presence_penalty, a.presence_penalty)
    freq_p = pick(req.frequency_penalty, a.frequency_penalty)
    temp_last = not a.temperature_first

    xtc_p = pick(req.xtc_probability, a.xtc_probability)
    xtc_t = pick(req.xtc_threshold, a.xtc_threshold)
    dry_mult = pick(req.dry_multiplier, a.dry_multiplier)
    dry_last_n = pick(req.dry_penalty_last_n, a.dry_penalty_last_n)
    use_xtc = xtc_p > 0.0 and xtc_t <= 0.5
    use_dry = dry_mult > 0.0 and dry_last_n != 0

    if not use_xtc and not use_dry:
        # Plain stack; ComboSampler keeps the fused-kernel fast path
        return ComboSampler(
            rep_p = rep_p, pres_p = pres_p, freq_p = freq_p,
            rep_sustain_range = penalty_range, rep_decay_range = penalty_range,
            temperature = temperature, min_p = min_p, top_k = top_k, top_p = top_p,
            temp_last = temp_last,
            adaptive_target = a.adaptive_target, adaptive_decay = a.adaptive_decay,
            logit_bias = logit_bias,
        )

    # Custom stack mirroring ComboSampler's order, with DRY among the penalty steps and
    # XTC after truncation (exllamav2 placement)
    stack = [
        SS_LogitBias(logit_bias or {}),
        SS_RepP(rep_p, penalty_range, penalty_range),
        SS_PresFreqP(pres_p, freq_p, penalty_range, penalty_range),
    ]
    if use_dry:
        breakers = req.dry_sequence_breakers
        if isinstance(breakers, str):
            try:
                breakers = json.loads(breakers)
            except json.JSONDecodeError:
                raise HTTPException(400, "dry_sequence_breakers must be a JSON array of strings")
        if breakers is None:
            breakers = list(DEFAULT_DRY_BREAKERS)
        stack.append(SS_DRY(
            multiplier = dry_mult,
            base = pick(req.dry_base, a.dry_base),
            allowed_length = pick(req.dry_allowed_length, a.dry_allowed_length),
            penalty_last_n = dry_last_n,
            breaker_ids = breaker_token_ids(state.tokenizer, tuple(breakers)),
        ))
    if temperature == 0.0 or top_k == 1:
        stack.append(SS_Argmax())
    else:
        stack += [
            SS_Temperature(temperature if not temp_last else 1.0),
            SS_MinP(min_p),
            SS_TopK(top_k),
            SS_TopP(top_p),
        ]
        if use_xtc:
            stack.append(SS_XTC(xtc_p, xtc_t, tokenizer = state.tokenizer))
        stack.append(SS_Temperature(temperature if temp_last else 1.0))
        if a.adaptive_target != 1.0:
            stack.append(SS_AdaptiveP(a.adaptive_target, a.adaptive_decay))
        else:
            stack.append(SS_Sample())
    return CustomSampler(stack)


def stop_conditions_for(req_stop: str | list[str] | None, ignore_eos: bool = False) -> list:
    conds: list = [] if ignore_eos else list(state.stop_token_ids)
    if req_stop:
        stops = [req_stop] if isinstance(req_stop, str) else list(req_stop)
        conds += [s for s in stops if s]
    return conds


def token_budget(prompt_len: int, requested: int | None) -> int:
    """Largest completion length that still fits the cache (jobs occupy whole pages)."""
    usable = state.context_length // PAGE_SIZE * PAGE_SIZE
    budget = usable - prompt_len - 1
    if budget <= 0:
        raise HTTPException(
            400,
            f"Prompt is {prompt_len} tokens but the cache only holds {state.context_length}. "
            f"Increase --cache_size or shorten the prompt."
        )
    if state.args.max_response_tokens:
        budget = min(budget, state.args.max_response_tokens)
    if requested is not None:
        budget = min(budget, requested)
    return budget


def finish_reason(eos_reason: str | None) -> str:
    return "length" if eos_reason == "max_new_tokens" else "stop"


def make_job(req: SamplingFields, ids: torch.Tensor, max_new: int,
             stop: str | list[str] | None, identifier: int = 0,
             ignore_eos: bool = False) -> AsyncJob:
    a = state.args
    return AsyncJob(
        state.generator,
        input_ids = ids,
        max_new_tokens = max_new,
        stop_conditions = stop_conditions_for(stop, ignore_eos),
        sampler = make_sampler(req),
        seed = req.seed,
        banned_strings = req.banned_strings or [],
        decode_special_tokens = req.decode_special_tokens,
        stop_on_loop = (a.loop_window, a.loop_min_reps) if a.loop_window else None,
        identifier = identifier,
    )


class DisconnectWatch:
    """Cancels a job when the client goes away, polling request.is_disconnected() from a
    background task (TabbyAPI-style active poll: passive detection via send() failure or
    listen_for_disconnect never fires on this stack, and polling receive() also makes
    uvicorn resume reading the socket, without which a client disconnect on a POST request
    can go unnoticed for the entire stream). The poll must NOT run inside the token loop:
    every is_disconnected() await suspends the consumer and hands the event loop to the
    generator's always-ready iteration task, which blocks it for one synchronous decode
    step -- polled per token, delivery cost ~2 decode steps/token and the undelivered half
    of the stream sat in the job queue until EOS flushed it as one burst."""
    def __init__(self, request: Request, job: AsyncJob, interval: float = 0.5):
        self.request = request
        self.job = job
        self.interval = interval
        self.disconnected = False
        self.task = asyncio.create_task(self._watch())

    async def _watch(self):
        while True:
            await asyncio.sleep(self.interval)
            if await self.request.is_disconnected():
                self.disconnected = True
                await self.job.cancel()
                print(" -- Client disconnected, job cancelled", flush = True)
                return

    def stop(self):
        self.task.cancel()


async def collect_job(job: AsyncJob, request: Request | None = None) -> tuple[str, dict]:
    """Run a job to completion, returning (text, final_result). Cancels on task cancellation."""
    text = ""
    final = {}
    watch = DisconnectWatch(request, job) if request is not None else None
    try:
        async for r in job:
            text += r.get("text", "")
            if r.get("eos"):
                final = r
    except asyncio.CancelledError:
        await job.cancel()
        raise
    finally:
        if watch:
            watch.stop()
    return text, final


def usage_dict(final: dict, completion_tokens_hint: int = 0) -> dict:
    pt = final.get("prompt_tokens", 0)
    ct = final.get("new_tokens", completion_tokens_hint)
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct}


def log_request(kind: str, final: dict):
    if not final:
        return
    try:
        pt = final.get("prompt_tokens", 0)
        cached = final.get("cached_tokens", 0)
        nt = final.get("new_tokens", 0)
        tp, tg = final.get("time_prefill", 0), final.get("time_generate", 0)
        pps = (pt - cached) / tp if tp > 0 else 0.0
        tps = nt / tg if tg > 0 else 0.0
        msg = (f" -- {kind}: {pt} prompt tokens ({cached} cached, {pps:.1f} t/s prefill), "
               f"{nt} generated ({tps:.2f} t/s), stop: {final.get('eos_reason', '?')}")
        if "accepted_draft_tokens" in final:
            da, dr = final["accepted_draft_tokens"], final["rejected_draft_tokens"]
            if da + dr > 0:
                msg += f", draft: {da}/{da + dr} accepted"
        print(msg, flush = True)
    except Exception:
        pass


def sse(obj: dict) -> str:
    # Payload only: EventSourceResponse adds the "data: " framing
    return json.dumps(obj, ensure_ascii = False)


def check_auth(request: Request):
    key = state.args.api_key
    if not key:
        return
    auth = request.headers.get("authorization", "")
    x_key = request.headers.get("x-api-key", "")
    if auth == f"Bearer {key}" or x_key == key:
        return
    raise HTTPException(401, "Invalid API key")


def flatten_content(content: Any) -> str:
    """OpenAI message content may be a string or a list of typed parts."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts)
    return str(content)


def chat_prompt_ids(req: ChatCompletionRequest) -> torch.Tensor:
    if not state.has_chat_template:
        raise HTTPException(
            400,
            "This model has no chat template in its tokenizer config. Use /v1/completions "
            "with a client-side instruct template instead."
        )
    messages = [
        {"role": m.get("role", "user"), "content": flatten_content(m.get("content"))}
        for m in req.messages
    ]
    if not messages:
        raise HTTPException(400, "messages must not be empty")

    kwargs = dict(state.default_template_kwargs)
    if req.chat_template_kwargs:
        kwargs.update(req.chat_template_kwargs)
    if req.tools:
        kwargs["tools"] = req.tools

    add_gen = req.add_generation_prompt
    if req.continue_final_message:
        kwargs["continue_final_message"] = True
        add_gen = False

    try:
        ids = state.tokenizer.hf_chat_template(
            messages,
            add_generation_prompt = add_gen,
            **kwargs,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Chat template rendering failed: {e}")
    return ids


def completion_prompt_ids(req: CompletionRequest) -> torch.Tensor:
    prompt = req.prompt
    if isinstance(prompt, list):
        if all(isinstance(t, int) for t in prompt):
            return torch.tensor([prompt], dtype = torch.long)
        if len(prompt) == 1 and isinstance(prompt[0], str):
            prompt = prompt[0]
        else:
            raise HTTPException(400, "prompt must be a string or a single list of token IDs")
    if not isinstance(prompt, str):
        raise HTTPException(400, "prompt must be a string or a list of token IDs")
    ids = state.tokenizer.encode(
        prompt,
        add_bos = req.add_bos,
        encode_special_tokens = req.parse_special,
    )
    # If the client's template already begins with BOS (some SillyTavern instruct templates
    # do), don't double it
    bos = state.tokenizer.bos_token_id
    if req.add_bos and bos is not None and ids.shape[-1] >= 2 \
            and ids[0, 0].item() == bos and ids[0, 1].item() == bos:
        ids = ids[:, 1:]
    return ids


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # AsyncGenerator must be created inside a running event loop
    a = state.args
    state.generator = AsyncGenerator(
        model = state.model,
        cache = state.cache,
        tokenizer = state.tokenizer,
        draft_model = state.draft_model,
        draft_cache = state.draft_cache,
        num_draft_tokens = a.num_draft_tokens,
        ngram_match_min = a.ngram_match_min,
        dynamic_draft_tokens = a.dynamic_draft,
        draft_confidence = a.draft_confidence,
        cpu_cache_size = int(a.cpu_cache_size * 1024 ** 3),
        recurrent_cache_size = int(a.recurrent_cache_size * 1024 ** 3),
    )
    print(f" -- Server ready: http://{a.host}:{a.port} (model: {state.model_name}, "
          f"context: {state.context_length} tokens)", flush = True)
    yield
    await state.generator.close()

app = FastAPI(title = "exl3_server", lifespan = lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_methods = ["*"],
    allow_headers = ["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/props")
async def props():
    hf_tok = getattr(state.tokenizer, "hf_tokenizer", None)
    return {
        "model": state.model_name,
        "model_path": state.args.model_dir,
        "n_ctx": state.context_length,
        "has_chat_template": state.has_chat_template,
        "chat_template": getattr(hf_tok, "chat_template", None) or "",
        "total_slots": state.args.autosplit_max_batch_size,
        "default_generation_settings": {"n_ctx": state.context_length},
        "stop_token_ids": state.stop_token_ids,
        "default_template_kwargs": state.default_template_kwargs,
    }


@app.get("/v1/models")
async def models(request: Request):
    check_auth(request)
    return {
        "object": "list",
        "data": [{
            "id": state.model_name,
            "object": "model",
            "created": int(time.time()),
            "owned_by": "exl3_server",
        }],
    }


@app.post("/tokenize")
async def tokenize(request: Request, body: TokenizeRequest):
    check_auth(request)
    ids = state.tokenizer.encode(
        body.content, add_bos = body.add_bos, encode_special_tokens = body.parse_special
    )
    return {"tokens": ids[0].tolist()}


@app.post("/detokenize")
async def detokenize(request: Request, body: DetokenizeRequest):
    check_auth(request)
    ids = torch.tensor([body.tokens], dtype = torch.long)
    text = state.tokenizer.decode(ids, decode_special_tokens = True)[0]
    return {"content": text}


# ---------------------------------------------------------------------------
# Chat completions
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatCompletionRequest):
    check_auth(request)
    ids = chat_prompt_ids(body)
    prompt_len = ids.shape[-1]
    requested = body.max_completion_tokens or body.max_tokens
    max_new = token_budget(prompt_len, requested)
    cmpl_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.stream:
        if body.n != 1:
            raise HTTPException(400, "n > 1 is not supported with streaming")
        job = make_job(body, ids, max_new, body.stop)
        include_usage = bool((body.stream_options or {}).get("include_usage"))

        async def stream():
            def chunk(delta: dict, fin: str | None = None, usage: dict | None = None):
                d = {
                    "id": cmpl_id,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": state.model_name,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": fin}],
                }
                if usage is not None:
                    d["usage"] = usage
                return sse(d)

            final = {}
            done = False
            watch = DisconnectWatch(request, job)
            try:
                yield chunk({"role": "assistant", "content": ""})
                async for r in job:
                    text = r.get("text", "")
                    if text:
                        yield chunk({"content": text})
                    if r.get("eos"):
                        final = r
                        done = True
                if watch.disconnected:
                    done = True
                    return
                yield chunk({}, fin = finish_reason(final.get("eos_reason")),
                            usage = usage_dict(final) if include_usage else None)
                yield "[DONE]"
                log_request("chat (stream)", final)
            finally:
                watch.stop()
                if not done:
                    await job.cancel()
                    print(" -- Client disconnected, job cancelled", flush = True)

        # EventSourceResponse (not StreamingResponse): under ASGI spec >= 2.4 starlette only
        # notices a disconnect when a write fails, abandoning the generator un-closed and
        # leaking the job until GC; sse_starlette listens for the disconnect and closes the
        # iterator deterministically, so the finally-cancel above always runs
        return CancellingStreamResponse(stream())

    # Non-streaming, n completions
    jobs = [make_job(body, ids, max_new, body.stop, identifier = i) for i in range(body.n)]
    results = await asyncio.gather(*(collect_job(j, request) for j in jobs))
    choices = []
    final = {}
    for i, (text, fin) in enumerate(results):
        final = fin or final
        choices.append({
            "index": i,
            "message": {"role": "assistant", "content": text},
            "finish_reason": finish_reason(fin.get("eos_reason")),
        })
        log_request("chat", fin)
    return JSONResponse({
        "id": cmpl_id,
        "object": "chat.completion",
        "created": created,
        "model": state.model_name,
        "choices": choices,
        "usage": usage_dict(final),
    })


# ---------------------------------------------------------------------------
# Raw completions (client-side template, e.g. SillyTavern Text Completion)
# ---------------------------------------------------------------------------

@app.post("/v1/completions")
async def completions(request: Request, body: CompletionRequest):
    check_auth(request)
    ids = completion_prompt_ids(body)
    prompt_len = ids.shape[-1]
    max_new = token_budget(prompt_len, body.max_tokens)
    cmpl_id = f"cmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    if body.stream:
        if body.n != 1:
            raise HTTPException(400, "n > 1 is not supported with streaming")
        job = make_job(body, ids, max_new, body.stop)
        include_usage = bool((body.stream_options or {}).get("include_usage"))

        async def stream():
            def chunk(text: str, fin: str | None = None, usage: dict | None = None):
                d = {
                    "id": cmpl_id,
                    "object": "text_completion",
                    "created": created,
                    "model": state.model_name,
                    "choices": [{"index": 0, "text": text, "finish_reason": fin}],
                }
                if usage is not None:
                    d["usage"] = usage
                return sse(d)

            final = {}
            done = False
            watch = DisconnectWatch(request, job)
            try:
                async for r in job:
                    text = r.get("text", "")
                    if text:
                        yield chunk(text)
                    if r.get("eos"):
                        final = r
                        done = True
                if watch.disconnected:
                    done = True
                    return
                yield chunk("", fin = finish_reason(final.get("eos_reason")),
                            usage = usage_dict(final) if include_usage else None)
                yield "[DONE]"
                log_request("completion (stream)", final)
            finally:
                watch.stop()
                if not done:
                    await job.cancel()
                    print(" -- Client disconnected, job cancelled", flush = True)

        # EventSourceResponse (not StreamingResponse): under ASGI spec >= 2.4 starlette only
        # notices a disconnect when a write fails, abandoning the generator un-closed and
        # leaking the job until GC; sse_starlette listens for the disconnect and closes the
        # iterator deterministically, so the finally-cancel above always runs
        return CancellingStreamResponse(stream())

    jobs = [make_job(body, ids, max_new, body.stop, identifier = i) for i in range(body.n)]
    results = await asyncio.gather(*(collect_job(j, request) for j in jobs))
    choices = []
    final = {}
    for i, (text, fin) in enumerate(results):
        final = fin or final
        choices.append({
            "index": i,
            "text": text,
            "finish_reason": finish_reason(fin.get("eos_reason")),
        })
        log_request("completion", fin)
    return JSONResponse({
        "id": cmpl_id,
        "object": "text_completion",
        "created": created,
        "model": state.model_name,
        "choices": choices,
        "usage": usage_dict(final),
    })


# ---------------------------------------------------------------------------
# llama.cpp-native endpoints (SillyTavern "llama.cpp" preset and similar tools)
# ---------------------------------------------------------------------------

def native_logit_bias(lb: list | dict | None) -> dict[str, float] | None:
    """Normalize llama.cpp logit_bias ([[token, bias|false], ...] or {token: bias})
    to the OpenAI dict form make_sampler expects."""
    if not lb:
        return None
    out: dict[str, float] = {}
    pairs = lb.items() if isinstance(lb, dict) else lb
    for pair in pairs:
        if isinstance(lb, dict):
            k, v = pair
        else:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                raise HTTPException(400, "logit_bias entries must be [token, bias] pairs")
            k, v = pair
        if not (isinstance(k, int) or (isinstance(k, str) and k.lstrip("-").isdigit())):
            raise HTTPException(400, "logit_bias tokens must be token IDs")
        out[str(k)] = -1e30 if v is False else float(v)
    return out


@app.post("/apply-template")
async def apply_template(request: Request, body: ApplyTemplateRequest):
    check_auth(request)
    if not state.has_chat_template:
        raise HTTPException(400, "Model has no chat template")
    messages = [
        {"role": m.get("role", "user"), "content": flatten_content(m.get("content"))}
        for m in body.messages
    ]
    kwargs = dict(state.default_template_kwargs)
    if body.chat_template_kwargs:
        kwargs.update(body.chat_template_kwargs)
    try:
        rendered = state.tokenizer.hf_render_chat_template(
            messages, add_generation_prompt = body.add_generation_prompt, **kwargs
        )
    except Exception as e:
        raise HTTPException(400, f"Chat template rendering failed: {e}")
    return {"prompt": rendered}


@app.post("/completion")
@app.post("/completions")
async def native_completion(request: Request, body: NativeCompletionRequest):
    check_auth(request)

    # Map native param names onto the shared sampler builder
    sf = SamplingFields(
        temperature = body.temperature,
        top_p = body.top_p,
        top_k = body.top_k,
        min_p = body.min_p,
        frequency_penalty = body.frequency_penalty,
        presence_penalty = body.presence_penalty,
        repetition_penalty = body.repeat_penalty,
        penalty_range = body.repeat_last_n if body.repeat_last_n else None,
        logit_bias = native_logit_bias(body.logit_bias),
        seed = body.seed if (body.seed is not None and body.seed >= 0) else None,
        xtc_probability = body.xtc_probability,
        xtc_threshold = body.xtc_threshold,
        dry_multiplier = body.dry_multiplier,
        dry_base = body.dry_base,
        dry_allowed_length = body.dry_allowed_length,
        dry_penalty_last_n = body.dry_penalty_last_n,
        dry_sequence_breakers = body.dry_sequence_breakers,
    )
    ids = completion_prompt_ids(CompletionRequest(
        prompt = body.prompt, add_bos = body.add_bos, parse_special = body.parse_special
    ))
    prompt_len = ids.shape[-1]
    requested = body.n_predict if body.n_predict > 0 else None
    max_new = token_budget(prompt_len, requested)
    job = make_job(sf, ids, max_new, body.stop, ignore_eos = body.ignore_eos)

    def native_stop_type(final: dict) -> tuple[str, str]:
        reason = final.get("eos_reason")
        if reason == "stop_string":
            return "word", final.get("eos_triggering_string", "")
        if reason == "max_new_tokens":
            return "limit", ""
        return "eos", ""

    def final_payload(text: str, tokens: list[int], final: dict) -> dict:
        stop_type, stopping_word = native_stop_type(final)
        pt = final.get("prompt_tokens", prompt_len)
        cached = final.get("cached_tokens", 0)
        nt = final.get("new_tokens", 0)
        tp, tg = final.get("time_prefill", 0.0), final.get("time_generate", 0.0)
        timings = {
            "prompt_n": pt - cached,
            "prompt_ms": tp * 1000.0,
            "prompt_per_second": (pt - cached) / tp if tp > 0 else 0.0,
            "predicted_n": nt,
            "predicted_ms": tg * 1000.0,
            "predicted_per_second": nt / tg if tg > 0 else 0.0,
        }
        if "accepted_draft_tokens" in final:
            timings["draft_n"] = final["accepted_draft_tokens"] + final.get("rejected_draft_tokens", 0)
            timings["draft_n_accepted"] = final["accepted_draft_tokens"]
        return {
            "index": 0,
            "content": text,
            "tokens": tokens if body.return_tokens else [],
            "stop": True,
            "model": state.model_name,
            "tokens_predicted": nt,
            "tokens_evaluated": pt,
            "tokens_cached": cached,
            "truncated": False,
            "stop_type": stop_type,
            "stopping_word": stopping_word,
            "generation_settings": {
                "n_ctx": state.context_length,
                "n_predict": max_new,
                "model": state.model_name,
            },
            "timings": timings,
        }

    if body.stream:
        async def stream():
            # Native streaming: chunks carry {content, stop: false}; the last event
            # carries stop: true with full stats. No "data: [DONE]" terminator.
            text_all = ""
            tokens_all: list[int] = []
            final = {}
            done = False
            watch = DisconnectWatch(request, job)
            try:
                async for r in job:
                    text = r.get("text", "")
                    tids = r.get("token_ids")
                    if tids is not None:
                        tokens_all += tids.flatten().tolist()
                    if text:
                        text_all += text
                        yield sse({"index": 0, "content": text, "tokens": [], "stop": False})
                    if r.get("eos"):
                        final = r
                        done = True
                if watch.disconnected:
                    done = True
                    return
                yield sse(final_payload("", tokens_all, final))
                log_request("native completion (stream)", final)
            finally:
                watch.stop()
                if not done:
                    await job.cancel()
                    print(" -- Client disconnected, job cancelled", flush = True)

        # EventSourceResponse (not StreamingResponse): under ASGI spec >= 2.4 starlette only
        # notices a disconnect when a write fails, abandoning the generator un-closed and
        # leaking the job until GC; sse_starlette listens for the disconnect and closes the
        # iterator deterministically, so the finally-cancel above always runs
        return CancellingStreamResponse(stream())

    text = ""
    tokens_all: list[int] = []
    final = {}
    watch = DisconnectWatch(request, job)
    try:
        async for r in job:
            text += r.get("text", "")
            tids = r.get("token_ids")
            if tids is not None:
                tokens_all += tids.flatten().tolist()
            if r.get("eos"):
                final = r
    except asyncio.CancelledError:
        await job.cancel()
        raise
    finally:
        watch.stop()
    log_request("native completion", final)
    return JSONResponse(final_payload(text, tokens_all, final))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@torch.inference_mode()
def main(args):
    state.args = args
    state.model_name = args.served_model_name or Path(args.model_dir).name

    # Default the cache to the model's native max context (like llama.cpp's n_ctx_train
    # default). Long-context models advertise huge maximums, so warn about the allocation.
    if args.cache_size is None:
        from exllamav3 import Config
        probe = Config.from_directory(args.model_dir, layer_map = args.layer_map)
        args.cache_size = probe.max_position_embeddings // PAGE_SIZE * PAGE_SIZE
        print(f" -- No -cs/--cache_size given; defaulting to the model's max context: "
              f"{args.cache_size} tokens", flush = True)
        if args.cache_size > 131072:
            print(f" !! That is a very large KV cache allocation and may not fit in memory. "
                  f"Pass -cs to cap it (e.g. -cs 32768), or -cq to quantize it.", flush = True)

    # Load model, cache, tokenizer, optional draft model (same as chat.py)
    (state.model, state.config, state.cache, state.tokenizer,
     state.draft_model, _draft_config, state.draft_cache) = model_init.init(args)
    state.context_length = state.cache.max_num_tokens

    # Stop tokens: model EOS list plus tokenizer EOS
    stop_ids = set()
    if state.config.eos_token_id_list:
        stop_ids |= {t for t in state.config.eos_token_id_list if t is not None}
    if state.tokenizer.eos_token_id is not None:
        stop_ids.add(state.tokenizer.eos_token_id)
    state.stop_token_ids = sorted(stop_ids)

    # Default chat-template kwargs from CLI
    if args.chat_template_kwargs:
        state.default_template_kwargs = json.loads(args.chat_template_kwargs)
        assert isinstance(state.default_template_kwargs, dict), \
            "--chat_template_kwargs must be a JSON object"

    # Validate the model's chat template up front so /v1/chat/completions fails loudly now,
    # not on the first request
    try:
        state.tokenizer.hf_render_chat_template(
            [{"role": "user", "content": "ping"}],
            add_generation_prompt = True,
            **state.default_template_kwargs,
        )
        state.has_chat_template = True
        print(" -- Chat template OK (using the model's tokenizer_config template)")
    except Exception as e:
        print(f" !! No usable chat template ({e.__class__.__name__}: {e})")
        print(" !! /v1/chat/completions is disabled; /v1/completions still works with a "
              "client-side template")

    uvicorn.run(app, host = args.host, port = args.port, log_level = "warning")

    # Known issue: interpreter teardown after unloading a model can SIGSEGV in native code.
    # All work is done at this point, so exit hard instead.
    os._exit(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        allow_abbrev = False,
        description = "OpenAI-compatible server for exllamav3 (llama.cpp-server style)",
    )
    model_init.add_args(
        parser,
        cache = True,
        add_sampling_args = True,
        add_draft_model_args = True,
        default_cache_size = 32768,
        default_autosplit_max_batch_size = 4,
    )
    # Override model_init's -cs default: None means "use the model's max context" (resolved in
    # main(); the -cs help text still shows model_init's 32768, which no longer applies)
    parser.set_defaults(cache_size = None)
    parser.add_argument("-host", "--host", type = str, default = "127.0.0.1", help = "Bind address, default: 127.0.0.1")
    parser.add_argument("-port", "--port", type = int, default = 3953, help = "Port, default: 3953")
    parser.add_argument("-key", "--api_key", type = str, default = None, help = "Require this API key (Bearer token or x-api-key header)")
    parser.add_argument("-smn", "--served_model_name", type = str, default = None, help = "Model name reported by the API, default: model directory name")
    parser.add_argument("-maxr", "--max_response_tokens", type = int, default = None, help = "Server-side cap on tokens per response, default: fill remaining context")
    parser.add_argument("-ctk", "--chat_template_kwargs", type = str, default = None, help = "Default kwargs for the chat template as JSON, e.g. '{\"enable_thinking\": false}'")
    parser.add_argument("-lw", "--loop_window", type = int, default = 0, help = "Loop detection window in tokens, 0 to disable (default)")
    parser.add_argument("-lmr", "--loop_min_reps", type = int, default = 3, help = "Min. reps for loop detection, default = 3")
    parser.add_argument("-xtcp", "--xtc_probability", type = float, default = 0.0, help = "XTC probability, 0 to disable (default)")
    parser.add_argument("-xtct", "--xtc_threshold", type = float, default = 0.1, help = "XTC threshold, default = 0.1")
    parser.add_argument("-drym", "--dry_multiplier", type = float, default = 0.0, help = "DRY multiplier, 0 to disable (default)")
    parser.add_argument("-dryb", "--dry_base", type = float, default = 1.75, help = "DRY base, default = 1.75")
    parser.add_argument("-dryal", "--dry_allowed_length", type = int, default = 2, help = "DRY allowed repeat length, default = 2")
    parser.add_argument("-dryln", "--dry_penalty_last_n", type = int, default = -1, help = "DRY scan range in tokens, -1 = whole context (default), 0 disables")
    main(parser.parse_args())
