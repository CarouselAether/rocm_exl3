"""
DRY (Don't Repeat Yourself) repetition penalty as an exllamav3 CustomSampler step.

Implements the algorithm from https://github.com/oobabooga/text-generation-webui/pull/5677
(same semantics as llama.cpp's dry_* parameters): a candidate token that would extend a
verbatim repeat of length L >= allowed_length is penalized by

    multiplier * base ** (L - allowed_length)

where L is measured as the longest suffix of the context that also occurred earlier,
immediately before that candidate token. Matches cannot span sequence-breaker tokens.

Lives in exl3_server (not in the exllamav3 tree) on purpose; it only uses the public
sampler-step API (SS_Base, SamplingState, reqs_past_ids).
"""

import torch
from exllamav3.generator.sampler.custom import SS_Base, SS, SamplingState

# Cap on how far a suffix match is extended. 1.75^(64-2) is ~1e15, which already acts as a
# hard ban after subtraction from the logit; longer matches change nothing but cost time.
MAX_MATCH_LEN = 64

_breaker_cache: dict[tuple, torch.Tensor] = {}


def breaker_token_ids(tokenizer, breakers: tuple[str, ...]) -> torch.Tensor:
    """
    Token IDs that act as sequence breakers: any token whose decoded piece contains one of
    the breaker strings, plus direct encodings of each breaker. Cached per breaker set.
    """
    key = (id(tokenizer), breakers)
    cached = _breaker_cache.get(key)
    if cached is not None:
        return cached
    ids = set()
    pieces = tokenizer.get_id_to_piece_list()
    for tid, piece in enumerate(pieces):
        for b in breakers:
            if b and b in piece:
                ids.add(tid)
                break
    for b in breakers:
        if not b:
            continue
        enc = tokenizer.encode(b, add_bos = False, encode_special_tokens = True)
        ids.update(enc.flatten().tolist())
    t = torch.tensor(sorted(ids), dtype = torch.long)
    _breaker_cache[key] = t
    return t


class SS_DRY(SS_Base):
    """
    Penalty step; goes at the head of the sampler stack with the other penalty steps
    (operates on raw logits, requires past IDs).
    """
    def __init__(
        self,
        multiplier: float,
        base: float = 1.75,
        allowed_length: int = 2,
        penalty_last_n: int = -1,
        breaker_ids: torch.Tensor | None = None,
    ):
        """
        :param multiplier:
            Penalty scale; 0 disables the step
        :param base:
            Exponential base for the penalty
        :param allowed_length:
            Repeats up to this length are free; each token beyond it multiplies the penalty by `base`
        :param penalty_last_n:
            Number of most recent tokens scanned for repeats; <0 = whole context, 0 = disabled
        :param breaker_ids:
            1-D long tensor of sequence-breaker token IDs (see breaker_token_ids)
        """
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = max(1, allowed_length)
        self.penalty_last_n = penalty_last_n
        self.breaker_ids = breaker_ids if breaker_ids is not None else torch.empty(0, dtype = torch.long)

    def _row_penalties(self, seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """For one context row, return (candidate token ids, penalty per candidate)."""
        empty = (torch.empty(0, dtype = torch.long), torch.empty(0))
        n = seq.numel()
        window = n if self.penalty_last_n < 0 else min(self.penalty_last_n, n)
        lo = n - window
        if n - lo < 2:
            return empty
        last = seq[-1]
        if (self.breaker_ids == last).any():
            return empty

        # Occurrences of the final token, excluding the final position itself. The token after
        # each occurrence is the candidate a repeat would extend to.
        js = (seq[lo : n - 1] == last).nonzero(as_tuple = True)[0] + lo
        if js.numel() == 0:
            return empty

        # Extend the suffix match backward from each occurrence in lockstep
        lengths = torch.ones(js.numel(), dtype = torch.long)
        active = torch.ones(js.numel(), dtype = torch.bool)
        is_breaker = lambda t: (self.breaker_ids == t).any()
        for k in range(1, min(MAX_MATCH_LEN, window)):
            tgt = n - 1 - k
            if tgt < lo or is_breaker(seq[tgt]):
                break
            prev = js - k
            ok = active & (prev >= lo)
            ok &= seq[prev.clamp(min = 0)] == seq[tgt]
            active = ok
            if not active.any():
                break
            lengths[active] += 1

        candidates = seq[js + 1]
        eligible = lengths >= self.allowed_length
        if not eligible.any():
            return empty
        exponents = (lengths[eligible] - self.allowed_length).clamp(max = MAX_MATCH_LEN).float()
        penalties = self.multiplier * torch.pow(self.base, exponents)
        return candidates[eligible], penalties

    def run(self, state: SamplingState):
        match state.state:
            case SS.INIT:
                state.logits = state.in_logits.float().clone()
            case SS.LOGITS:
                pass
            case _:
                raise ValueError("Sampling logic error")
        state.state = SS.LOGITS

        past = state.past_ids
        if past is None:
            return
        past = past.cpu()
        if past.dim() == 1:
            past = past.unsqueeze(0)
        for row in range(state.bsz):
            seq = past[row if row < past.shape[0] else -1].flatten()
            cand, pen = self._row_penalties(seq)
            if cand.numel() == 0:
                continue
            # Keep the max penalty when a candidate appears after several repeats
            acc = torch.zeros(state.dim)
            acc.scatter_reduce_(0, cand, pen, reduce = "amax")
            state.logits[row] -= acc.to(state.logits.device)

    def alt(self):
        if self.multiplier <= 0.0 or self.penalty_last_n == 0:
            from exllamav3.generator.sampler.custom import SS_NoOp
            return SS_NoOp()
        return None

    def reqs_past_ids(self):
        return True
