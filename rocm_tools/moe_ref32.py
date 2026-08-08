"""Measure fused exl3_moe AND the per-expert path against an fp32 reference.

    python rocm_tools/moe_ref32.py -m /path/to/model [-b 1,8] [-l 2]

moe_check.py compares two fp16 implementations against each other, which cannot
say which one is wrong -- a ~1% mean disagreement is the expected result of two
different fp16 accumulation orders. This builds an actual reference: expert
weights dequantized to a dense fp32 matrix via get_weight_tensor() (which applies
the Hadamard pre-multiply and the suh/svh scales, so it is the true effective
weight), then the MoE evaluated in fp32 torch.

Read the two error columns together:
  fused ~= torch, both small     -> exl3_moe is fine, moe_check was measuring noise
  fused >> torch                 -> exl3_moe has a real defect
  both large                     -> the shared quant/dequant path is suspect, not MoE

Isolation: shared_experts is cleared for both runs (it is applied after the routed
part behind an `if`, so this yields routed-only output, which is what the fp32
reference computes). bc / support_quant_paths are cleared so the fused run takes
branch 1057 and the reference run reaches the torch per-expert path -- without
that, bsz <= MAX_BSZN silently routes both through mgemm and the diff reads 0.

Assumes no router_pre_norm / routed_pre_norm (true for glm4_moe): the expert input
is then just x. Asserts rather than silently comparing the wrong tensors.

Slow by construction -- it dequantizes every active expert to fp32. Keep -b small;
bsz=64 can touch all 128 experts.
"""

import argparse, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP


def err(a, b):
    d = (a - b).abs()
    scale = b.abs().max().clamp_min(1e-6)
    return float(d.max()), float(d.mean()), float(d.max() / scale), float(d.mean() / scale)


@torch.inference_mode()
def fp32_reference(mod, x, sel, wts):
    y = x.reshape(-1, mod.hidden_size).float()
    out = torch.zeros_like(y)
    act_limit = getattr(mod, "act_limit", None)

    for e in torch.unique(sel).tolist():
        if e < 0 or e >= len(mod.ups):
            continue
        hit = (sel == e)                       # [tokens, top_k]
        rows = hit.any(dim = 1).nonzero().flatten()
        if rows.numel() == 0:
            continue
        # routing weight for expert e on each of its tokens
        w_e = (wts * hit.float()).sum(dim = 1)[rows].unsqueeze(1)
        xe = y[rows]

        Wu = mod.ups[e].inner.get_weight_tensor().float()
        u = xe @ Wu
        if mod.ups[e].inner.bias is not None:
            u = u + mod.ups[e].inner.bias.float()
        del Wu

        if mod.gated:
            Wg = mod.gates[e].inner.get_weight_tensor().float()
            g = xe @ Wg
            if mod.gates[e].inner.bias is not None:
                g = g + mod.gates[e].inner.bias.float()
            del Wg
            if act_limit:
                g = g.clamp(-act_limit, act_limit)
                u = u.clamp(-act_limit, act_limit)
            a = (F.gelu(g) if mod.activation_fn == "gelu" else F.silu(g)) * u
        else:
            a = F.relu(u) ** 2

        Wd = mod.downs[e].inner.get_weight_tensor().float()
        o = a @ Wd
        if mod.downs[e].inner.bias is not None:
            o = o + mod.downs[e].inner.bias.float()
        del Wd, a, u

        out[rows] += o * w_e
        del o
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model_dir", required = True)
    ap.add_argument("-b", "--batches", default = "1,8")
    ap.add_argument("-cs", "--cache_size", type = int, default = 32768)
    ap.add_argument("-l", "--layers", type = int, default = 2)
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]

    print(f" -- loading {args.model_dir}")
    config = Config.from_directory(args.model_dir)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens = args.cache_size)
    model.load(progressbar = True)
    tok = Tokenizer.from_config(config)

    try:
        from exllamav3.rocm_py import describe
        print("ROCm patches:"); print(describe())
    except Exception:
        pass

    state = {"n": 0, "bsz": 0}
    orig_forward = BlockSparseMLP.forward

    def forward_cmp(self, x, params, out_dtype = None):
        if getattr(self, "fused_mode_buffers", None) is None or state["n"] >= args.layers:
            return orig_forward(self, x, params, out_dtype)

        assert getattr(self, "router_pre_norm", None) is None and \
               getattr(self, "routed_pre_norm", None) is None and \
               not getattr(self, "alt_residual_channel", False), \
               "this model norms between module input and expert input; reference would be wrong"

        cap = {}
        orig_rfn = self.routing_fn

        def rfn(bsz, cfg, y, p):
            s, w = orig_rfn(bsz, cfg, y, p)
            cap["sel"] = s.clone()
            cap["w"] = w.float().clone()
            return s, w

        saved = (self.shared_experts, self.bc, self.support_quant_paths,
                 self.fused_mode_buffers, self.routing_fn)

        def run(fused):
            p = dict(params)
            if "residual" in p and isinstance(p["residual"], torch.Tensor):
                p["residual"] = p["residual"].clone()
            self.shared_experts = None
            self.bc = None
            self.support_quant_paths = False
            self.routing_fn = rfn
            self.fused_mode_buffers = saved[3] if fused else None
            try:
                return orig_forward(self, x.clone(), p, out_dtype)
            finally:
                (self.shared_experts, self.bc, self.support_quant_paths,
                 self.fused_mode_buffers, self.routing_fn) = saved

        out_fused = run(True).float().reshape(-1, self.hidden_size)
        out_torch = run(False).float().reshape(-1, self.hidden_size)
        ref = fp32_reference(self, x, cap["sel"], cap["w"])

        fmax, fmean, frmax, frmean = err(out_fused, ref)
        tmax, tmean, trmax, trmean = err(out_torch, ref)

        verdict = "fused OK" if frmean <= max(3 * trmean, 1e-4) else "FUSED WORSE"
        print(f"   bsz={state['bsz']:<5} {self.key:40}")
        print(f"       fused vs fp32 : max={fmax:9.4g} mean={fmean:9.4g}  relmean={frmean:8.4%}")
        print(f"       torch vs fp32 : max={tmax:9.4g} mean={tmean:9.4g}  relmean={trmean:8.4%}   -> {verdict}")
        state["n"] += 1
        return out_fused.reshape(x.shape[:-1] + (self.hidden_size,))

    BlockSparseMLP.forward = forward_cmp

    g = torch.Generator().manual_seed(1234)
    vocab = int(getattr(config, "vocab_size", 0) or tok.actual_vocab_size or 32000)
    full = torch.randint(0, max(vocab - 1, 1), (1, max(batches)), generator = g)

    print("\n fused exl3_moe and per-expert torch, each vs an fp32 reference\n")
    for b in batches:
        state["n"] = 0
        state["bsz"] = b
        with torch.inference_mode():
            model.prefill(input_ids = full[:, :b], params = {"attn_mode": "flash_attn_nc"})
        print()

    print(" If both relmean are the same order, exl3_moe is as accurate as the")
    print(" per-expert path and moe_check.py was measuring fp16 ordering noise.")


if __name__ == "__main__":
    main()
