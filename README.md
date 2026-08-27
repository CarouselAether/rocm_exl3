
# <img src="doc/cat.png" width="40"> ExLlamaV3 — ROCm / RDNA fork

This is a **ROCm fork of [ExLlamaV3](https://github.com/turboderp-org/exllamav3)** by turboderp, tracking
upstream v1.4.4. If you are on NVIDIA, you want [the upstream repo](https://github.com/turboderp-org/exllamav3) —
this one builds for CUDA too, but adds nothing there.

The Python package is still named `exllamav3`, so it is a drop-in replacement (including for TabbyAPI).
Only the repository is renamed.

### What this fork changes

The CUDA kernels that cannot compile for RDNA are replaced with hand-written HIP/WMMA siblings under
`exllamav3_ext/rocm/`, reached through a compat shim and include-path redirection. Python divergences live in
`exllamav3/rocm_py/` and are applied as monkeypatches at import.

**No upstream C++ or CUDA source is modified — not one.** Verify it yourself:

```sh
git diff --stat v1.4.4 -- '*.cu' '*.cuh' '*.cpp' '*.h' ':(exclude)exllamav3/exllamav3_ext/rocm'
# (empty)
```

Outside `rocm/` and `rocm_py/`, exactly four upstream files differ from v1.4.4:

| file | change |
|---|---|
| `setup.py` | ROCm backend selector and `hipcc` builder. All ROCm behaviour is inside `HIPBuildExtension`, so a CUDA build is untouched upstream code. |
| `exllamav3/__init__.py` | Six lines calling `rocm_py.apply()`. Returns immediately when `torch.version.hip` is `None`, so it is inert on CUDA. |
| `exllamav3/modules/attention_fn/triton_paged.py` | Selects the narrow-KV prefill tile explicitly on RDNA instead of relying on `get_device_capability()` accidentally reporting `(11, 5)`, plus measured notes on decode split counts. |
| `README.md` | This section. |

```sh
git diff --stat v1.4.4 -- . ':(exclude)exllamav3/exllamav3_ext/rocm' ':(exclude)exllamav3/rocm_py' ':(exclude)rocm_tools'
```

That is the whole surface. Rebasing onto a new upstream means re-applying four files, none of them kernels.

### Requirements

| | |
|---|---|
| ROCm | **7.2.4 or newer** — the build hard-fails below this |
| GPU | RDNA3 / RDNA3.5 / RDNA4: `gfx1100`, `gfx1101`, `gfx1102`, `gfx1150`, `gfx1151`, `gfx1200`, `gfx1201` |
| Python | 3.10+ (whatever the ROCm torch index publishes a wheel for) |
| Torch | ROCm build, from `download.pytorch.org/whl/rocmX.Y` — see below |

You do **not** need FlashAttention. Upstream uses Triton paged attention, so the FA2 dependency that
earlier ROCm forks required is gone.

### Install

```sh
git clone https://github.com/CarouselAether/rocm_exl3
cd rocm_exl3

# 1. ROCm torch + triton-rocm + everything else.
#    Do NOT use requirements.txt on ROCm -- it resolves torch from PyPI, which is the CUDA build.
pip install -r requirements_rocm.txt

# 2. Build and install the extension against that torch.
pip install --no-build-isolation .
```

`--no-build-isolation` is required, not optional: pip otherwise builds in an isolated environment with no
torch in it, and a torch C++ extension has to be compiled against the same torch it will run against.
Building without it fails with an explanation rather than silently installing an empty package.

The build compiles ~100 sources with `hipcc` in parallel (`MAX_JOBS` to limit it, e.g. on a low-memory
machine).

### Tested

Developed on a Ryzen AI Max 395+ (Strix Halo, **gfx1151**, 128 GB unified) — Ubuntu 24.04, ROCm 7.2.4,
torch 2.13.0+rocm7.2, triton-rocm 3.7.1, Python 3.12. Verified end to end with GLM-4.6V (MoE, 3.55 bpw),
Gemma-4-31B (dense) and DeepSeek-V4-Flash (DSA sparse attention, 2.04 bpw). The other architectures in the
supported list above should work but are untested — reports welcome.

### Known limitations on ROCm

- **Tensor-parallel is not available.** The `parallel/` kernels are excluded from the ROCm build.
- **Vision/multimodal is untested.** Text generation is what has been verified.
- Kernel behaviour can be bisected at runtime with the `EXL3_ROCM_*` environment switches — see
  `exllamav3/rocm_py/__init__.py`, which documents each one and why it exists.

### Using with TabbyAPI

TabbyAPI's `start.py` offers only `cu12` / `cu13` GPU options and will install a **CUDA** `exllamav3` wheel
straight over this one. Use `--nowheel`, which skips the extras entirely:

```sh
python start.py --nowheel
```

Install this fork *after* TabbyAPI, or re-install it if `start.py` has already clobbered it.

---

ExLlamaV3 is an inference library for running local LLMs on modern consumer GPUs. Headline features:

- New [EXL3](doc/exl3.md) quantization format based on QTIP
- Flexible tensor-parallel and expert-parallel inference for consumer hardware setups
- OpenAI-compatible server provided via [TabbyAPI](https://github.com/theroyallab/tabbyAPI/) 
- Continuous, dynamic batching
- HF Transformers plugin (see [here](examples/transformers_integration.py))
- HF model support (see [supported architectures](#architecture-support))
- Speculative decoding
- 2-8 bit cache quantization
- Multimodal support
- LoRA support

The official and recommended backend server for ExLlamaV3 is [TabbyAPI](https://github.com/theroyallab/tabbyAPI/), which provides an OpenAI-compatible API for local or remote inference, with extended features like HF model downloading, embedding model support and support for HF Jinja2 chat templates.

## Architecture support

- **AFM** (ArceeForCausalLM)
- **AfMoE** (AfmoeForCausalLM)
- **Apertus** (ApertursForCausalLM)
- **Command-R** etc. (CohereForCausalLM)
- **Command-A**, **Command-R7B**, **Command-R+** etc. (Cohere2ForCausalLM)
- **DeciLM**, **Nemotron** (DeciLMForCausalLM)
- **Deepseek V3** (DeepseekV3ForCausalLM)
- **Deepseek V4** (DeepseekV4ForCausalLM)
- **dots.llm1** (Dots1ForCausalLM) (`n_group>1` currently not supported)
- **ERNIE 4.5** (Ernie4_5_ForCausalLM, Ernie4_5_MoeForCausalLM)
- **EXAONE 4.0** (Exaone4ForCausalLM)
- **Gemma 2** (Gemma2ForCausalLM)
- **Gemma 3** (Gemma3ForCausalLM, Gemma3ForConditionalGeneration) *- multimodal*
- **Gemma 4** (Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration) *- multimodal* (E2B/E4B currently not supported)
- **GLM 4**, **GLM 4.5**, **GLM 4.5-Air**, **GLM 4.6** (Glm4ForCausalLM, Glm4MoeForCausalLM)
- **GLM 4.1V**, **GLM 4.5V** (Glm4vForConditionalGeneration, Glm4vMoeForConditionalGeneration) *- multimodal*
- **GPT-OSS** (GptOssForCausalLM)
- **HyperCLOVAX** (HyperCLOVAXForCausalLM, HCXVisionV2ForCausalLM) *- multimodal*
- **Hy3** (HYV3ForCausalLM)
- **IQuest-Coder** (IQuestCoderForCausalLM)
- **Laguna 2.1** (LagunaForCausalLM)
- **LFM 2.5** (Lfm2MoeForCausalLM)
- **Llama**, **Llama 2**, **Llama 3**, **Llama 3.1-Nemotron** etc. (LlamaForCausalLM)
- **MiMo-RL** (MiMoForCausalLM)
- **MiniMax-M2** (MiniMaxM2ForCausalLM)
- **Mistral**, **Ministral 3**, **Devstral 2**, **Mistral-4** etc. (MistralForCausalLM, Mistral3ForConditionalGeneration) *- multimodal*
- **Mixtral** (MixtralForCausalLM)
- **NemotronH, Nemotron-3** (NemotronHForCausalLM)
- **Olmo 3.1** (Olmo3ForCausalLM)
- **Olmo-Hybrid** (OlmoHybridForCausalLM)
- **Phi3**, **Phi4** (Phi3ForCausalLM)
- **Qwen 2**, **Qwen 2.5**, **Qwen 2.5 VL** (Qwen2ForCausalLM, Qwen2_5_VLForConditionalGeneration) *- multimodal*
- **Qwen 3** (Qwen3ForCausalLM, Qwen3MoeForCausalLM)
- **Qwen 3-Next** (Qwen3NextForCausalLM)
- **Qwen 3-VL** (Qwen3VLForConditionalGeneration)  *- multimodal*
- **Qwen 3-VL MoE** (Qwen3VLMoeForConditionalGeneration) *- multimodal*
- **Qwen 3.5** (Qwen3_5ForConditionalGeneration) *- multimodal*
- **Qwen 3.5 MoE** (Qwen3_5MoeForConditionalGeneration) *- multimodal*
- **Seed-OSS** (SeedOssForCausalLM)
- **SmolLM** (SmolLM3ForCausalLM)
- **SolarOpen** (SolarOpenForCausalLM)
- **Step 3.5 Flash** (Step3p5ForCausalLM)
- **Step 3.7 Flash** (Step3p7ForConditionalGeneration) *- multimodal*

Always adding more, stay tuned.


## What's missing?

Currently on the to-do list:

- ROCm support

As for what is implemented, expect that some things may be a little broken at first. Please be patient, raise issues and/or contribute. 👉👈 


## How to?

**On ROCm, see [Install](#install) at the top of this file** — the methods below are upstream's CUDA
instructions and are kept for the CUDA path. There is no prebuilt ROCm wheel yet, so building from source
is currently the only ROCm route.

[TabbyAPI](https://github.com/theroyallab/tabbyAPI/) has a startup script that manages and installs prerequisites if you want to get started quickly with inference in an OAI-compatible client. 

Otherwise, start by making sure you have the appropriate version of [PyTorch](https://pytorch.org/get-started/locally/) installed (CUDA 12.4 or later) since the Torch dependency is not automatically handled by `pip`. Then pick a method below:

### Method 1: Installing from prebuilt wheel (recommended if you're unsure)

Pick a wheel from the [releases page](https://github.com/turboderp-org/exllamav3/releases), then e.g.:

```sh
pip install https://github.com/turboderp-org/exllamav3/releases/download/v0.0.6/exllamav3-0.0.6+cu128.torch2.8.0-cp313-cp313-linux_x86_64.whl
```

### Method 2: Installing from PyPi:

```sh
pip install exllamav3
```
Note that the PyPi package does not contain a prebuilt extension and requires the CUDA toolkit and build prerequisites (i.e. VS Build Tools on Windows, gcc on Linux, `python-dev` headers etc.).    

### Method 3: Building from source

Before building, make sure you have an appropriate version of Torch installed. Install a `flash-attn-2` wheel, e.g. from [here](https://mjunya.com/flash-attention-prebuild-wheels/). 

On Windows, you should also make sure you have the `triton-windows` package installed. ExLlamaV3 may work without it, but many things will work suboptimally.   

```sh
# Clone the repo
git clone https://github.com/turboderp-org/exllamav3
cd exllamav3

# (Optional) switch to dev branch for latest in-progress features
git checkout dev

# Install requirements (make sure you install Torch separately)
pip install -r requirements.txt
```

At this point you should be able to run the conversion, eval and example scripts from the main repo directory, e.g. `python convert.py -i ...`

To install the library for the active venv, run from the repo directory:

```sh
pip install .
```

Relevant env variables for building:
- `MAX_JOBS`: by default ninja may launch too many processes and run out of system memory for compilation. Set this to a reasonable value like 4 in that case.  
- `EXLLAMA_NOCOMPILE`: set to install the library without compiling the C++/CUDA extension. Torch will build/load it at runtime instead.

ROCm-specific build variables:
- `EXL3_BACKEND`: `cuda` or `rocm`, forcing the backend. Otherwise it follows the installed torch.
- `MAX_JOBS`: also honoured by the ROCm builder, which drives `hipcc` directly. It defaults to a value
  bounded by both core count and RAM (~2.5 GB budgeted per job), so lower it if you still run out of memory.
- `PYTORCH_ROCM_ARCH` / `GPU_ARCHS`: semicolon-separated `gfx` list to build for. Defaults to what `rocminfo` reports, filtered against the supported list. `PYTORCH_ROCM_ARCH` takes precedence.
- `EXL3_RDNA_SMEM_MAX`: LDS budget in bytes. Defaults to the device's `sharedMemPerBlock` (64 KB on RDNA).
- `EXL3_RDNA_MOE_TILESIZE_K`: `32` (default) or `16`. 16 forces the MoE GEMMs onto the single-K path — the
  tile geometry every RDNA shape is validated on — at a cost of roughly 1.4–1.6× MoE throughput. It is the
  first thing to try if fused MoE output ever looks wrong.
- `EXL3_SKIP_ROCM_VERSION_CHECK`: bypass the ROCm >= 7.2.4 requirement. Not advised — older ROCm builds this
  extension successfully and then computes wrong results.


## Conversion

To convert a model to EXL3 format, use:

```sh
# Convert model
python convert.py -i <input_dir> -o <output_dir> -w <working_dir> -b <bitrate>

# Resume an interrupted quant job
python convert.py -w <working_dir> -r

# More options
python convert.py -h
```

The working directory is temporary storage for state checkpoints and for storing quantized tensors until the converted model can be compiled. It should have enough free space to store an entire copy of the output model. Note that while EXL2 conversion by default resumes an interrupted job when pointed to an existing folder, EXL3 needs you to explicitly resume with the `-r`/`--resume` argument.    

See [here](doc/convert.md) for more information.


## Examples

A number of example scripts are provided to showcase the features of the backend and generator. Some of them have hardcoded model paths and should be edited before you run them, but there is a simple CLI chatbot that you can start with:

```sh
python examples/chat.py -m <input_dir> -mode <prompt_mode> 

# E.g.:
python examples/chat.py -m /mnt/models/llama3.1-8b-instruct-exl3 -mode llama3

# Wealth of options
python examples/chat.py -h
```

## EXL3 quantization

<div align="center">
    <a href="doc/exl3.md" target="_blank">
        <img src="doc/llama31_8b_instruct_bpw.png" width="640">
    </a>
</div>

Despite their amazing achievements, most SOTA quantization techniques remain cumbersome or even prohibitively expensive to use. For instance, **AQLM** quantization of a 70B model takes around **720 GPU-hours** on an A100 server, costing $850 US at the time of writing. ExLlamaV3 aims to address this with the **EXL3** format, which is a streamlined variant of [**QTIP**](https://github.com/Cornell-RelaxML/qtip) from Cornell RelaxML. The conversion process is designed to be simple and efficient and requires only an input model (in HF format) and a target bitrate. By computing Hessians on the fly and thanks to a fused Viterbi kernel, the quantizer can convert a model in a single step, taking a couple of minutes for smaller models, up to a few hours for larger ones (70B+) (on a single RTX 4090 or equivalent GPU.)

The [Marlin](https://github.com/IST-DASLab/marlin)-inspired GEMM kernel achieves roughly memory-bound latency under optimal conditions (4bpw, RTX 4090), though it still needs some work to achieve the same efficiency on Ampere GPUs and to remain memory-bound at lower bitrates.

Since converted models largely retain the original file structure (unlike **EXL2** which renames some tensors in its quest to turn every model into a Llama variant), it will be possible to extend **EXL3** support to other frameworks like HF Transformers and vLLM.

There are some benchmark results [here](doc/exl3.md), and a full writeup on the format is coming soon.

Fun fact: Llama-3.1-70B-EXL3 is coherent at 1.6 bpw. With the output layer quantized to 3 bpw and a 4096-token cache, inference is possible in under 16 GB of VRAM. 


### Community

You are always welcome to join the [ExLlama discord server](https://discord.gg/NSFwVuCjRq) ←🎮  


### 🤗 HuggingFace repos

A selection of EXL3-quantized models is available [here](https://huggingface.co/collections/turboderp/exl3-models-67f2dfe530f05cb9f596d21a). Also shout out the following lovely people:
 
- [ArtusDev](https://huggingface.co/ArtusDev)
- [MikeRoz](https://huggingface.co/MikeRoz) 
- [MetaphoricalCode](https://huggingface.co/MetaphoricalCode) 
- [Ready.Art](https://huggingface.co/ReadyArt) 
- [isogen](https://huggingface.co/isogen/models)


## Acknowledgements

This project owes its existence to a wonderful community of FOSS developers and some very generous supporters (🐈❤️!) The following projects in particular deserve a special mention:

- [ExLlamaV3](https://github.com/turboderp-org/exllamav3)
- [TabbyAPI](https://github.com/theroyallab/tabbyAPI/)
- [PyTorch](https://github.com/pytorch/pytorch)
- [FlashAttention](https://github.com/Dao-AILab/flash-attention)
- [QTIP](https://github.com/Cornell-RelaxML/qtip)
- [Transformers](https://github.com/huggingface/transformers)
- [Marlin](https://github.com/IST-DASLab/marlin)
- [Flash Linear Attention](https://github.com/fla-org/flash-linear-attention)
