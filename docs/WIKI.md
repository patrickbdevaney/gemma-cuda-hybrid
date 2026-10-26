# gemma-cuda-hybrid — Engineering Wiki

A pure-CUDA/C++ inference engine and OpenAI-compatible server for **`google/gemma-4-26B-A4B-it`**
served from the **RedHatAI NVFP4** checkpoint, with the **`z-lab/gemma-4-26B-A4B-it-DFlash`**
block-diffusion draft head, hand-tuned for **NVIDIA Jetson Thor (`sm_110a`, CUDA 13.0)**.
No PyTorch, no Python on the hot path. Single binary (`build/forward`, 3.1 MB as built).

This document reconciles the repo's own documents (which were written at different times and
disagree in places — every disagreement found is recorded in §11). Everything here is traceable
to a file; where something is undocumented it is called out as undocumented rather than inferred.

Repo history: 81 commits, all dated **2026-07-01 → 2026-07-03**. The repo is a fork of the sibling
`~/gemma-cuda-server` ("v1.0 banked", DFlash 82 tok/s) at commit `db8de1b`. Anything in
`../dev/archive/OPTIMIZATION_LOG.md` cycles 0–25 and `../dev/archive/CHAMPION.md` describes the *parent* repo's history, not
work done here.

---

## 1. What it is, what it runs on, and the headline numbers

### 1.1 Hardware target

From `CUDA_ENGINEERING_CONSTITUTION.md` §1 (marked "VERIFIED, some corrected the hard way"):

| | |
|---|---|
| Arch | Jetson Thor, Blackwell **`sm_110a`** (renumbered from `sm_101` in CUDA 12.8/12.9), CUDA 13.0 |
| SMs | **20** (the widely-quoted "96" is max-config *tensor cores*), 2560 CUDA cores |
| SMEM | 228 KB/SM (datacenter-class) |
| Memory | ~273 GB/s LPDDR5X unified; **GPU realizes only ~73% (~200 GB/s measured by memcpy)** — shared EMC arbiter |
| Features | TMEM + tcgen05 + TMA present (like B200, unlike DGX Spark / RTX 50) |
| Build | `nvcc -O2 -arch=sm_110a --default-stream per-thread` (`scripts/build.sh`) |

**Thermal throttling is real and is the single most important measurement caveat in this repo:**
absolute tok/s for the *same binary* drifts **94 ↔ 108** across a long session. The constitution
mandates back-to-back A/B (stash baseline → rebuild → compare → restore) for every change and
says never to trust absolute numbers across time.

### 1.2 Headline performance — and exactly what was measured

The champion benchmark is fixed and narrow. From `CUDA_ENGINEERING_CONSTITUTION.md` §0 and
`scripts/bench.sh`:

- Prompt: **"List the first 40 prime numbers, comma separated."** (`/tmp/share/bench_primes.txt`)
- Command: `DFLASH=1 GEN=80 ./build/forward`, **median of 5–6 back-to-back runs**
- **Greedy** (T=0; `SAMPLE` unset → `k_argmax`), **BF16 KV** (the `FP8KV` default only applies to `SERVE=1`)
- `DK=14` draft tokens per block ⇒ verify forward is **M=15**
- Prompt ≈ 25 tokens, decode 80 tokens — i.e. a **short-context, highly-predictable** workload

| Engine | tok/s | acceptance τ | ms/step | source |
|---|---|---|---|---|
| **This engine, DFlash** | **~118** | **13.33 / 14 ≈ 95%** | ~123 | `CUDA_ENGINEERING_CONSTITUTION.md` header, git `f59daf3` |
| This engine, base (no spec) | **~44.5–45** | — | — | constitution §0, `../dev/archive/LOOP_LOG.md` |
| vLLM 0.22.1rc1 + DFlash, same prompt/weights/box | **107.5** | 9.21 | 85 | `BENCHMARK_COMPARISON.md` |
| vLLM base (no spec) | 52 | — | — | `../dev/archive/RESEARCH_SYNTHESIS.md`, `reference/PERF_RESEARCH.md` |

So the "+10% vs vLLM" headline is **118 / 107.5 = +9.8% on this one short greedy prompt**.
Derived ceiling quoted throughout: τ 13.33 at vLLM's 85 ms step ⇒ **~157 tok/s**.

**The most important caveat, and it is in the repo:** `REVERSE_ENGINEERING.md` measured this
engine at **82 tok/s on 80-token primes but 58 tok/s on 300-token primes (τ 10.0)** while vLLM
held ~100 tok/s (τ 7.84) on the harder workload — i.e. this engine degrades faster with length.
That controlled comparison was done at the **82** champion and **was never repeated after the arc
to 118**. There is no measurement in the repo of this engine on long context, code, or reasoning
workloads. `AGENTIC_OPTIMIZATION_METHODOLOGY.md` §0 flags exactly this ("optimizing only `primes`
blinds us"), and the head-pack attention win (+0.6% on primes) is explicitly described as a
long-context lever that the benchmark cannot see.

`traces/summary.json` (18 prompts, categories: short_factual, reasoning, code_gen/edit,
explanation, multi_turn, agentic_format, long_context-needle) contains **vLLM reference numbers
only** — decode 9.7–114 tok/s, mean accepted/draft 1.0–9.0. These are the oracle traces, not this
engine's results.

### 1.3 Memory

- Target checkpoint `model.safetensors` = **16.42 GB** single shard, 47,648 tensors (verified by
  reading the header; matches `tests/test_loader.cpp`'s expected count).
- Draft checkpoint = **0.859 GB**, 58 tensors, all BF16.
- Server at `CTX=65536` with FP8 KV: `server/README.md` states ≈ **7 GB KV + 4.3 GB taps + 16 GB
  model**. The taps buffer is `taps_ctx[(CAP+32)*6*H]` fp32 = 4.43 GB at 64K — the DFlash context
  taps are the second-largest allocation in the process.

---

## 2. Repository map

```
src/forward.cu        1076 lines — model, all target kernels, engine, CUDA graphs, HTTP server
src/draft.cu           307 lines — the DFlash draft model + propose()
src/megakernel.cu      160 lines — flag-gated (MEGAKERNEL=1) fused/persistent kernels (measured negative)
kernels/tc_verify_gemm.cu       Marlin-class raw mma.sync W4A16 + bf16 TC GEMMs  ← the champion kernel
kernels/fp4_gemm.cu             fp4_gemv (M=1), w4a16_gemm (M≤16), cuBLASLt W4A4 nvfp4_gemm (tests only)
kernels/nvfp4_quant.cu          activation FP4 quantizer (tests only; runtime is W4A16)
kernels/attention.cu            original correctness-first sdpa() (tests only; runtime uses sdpa_cache_kernel)
kernels/cutlass_moe.cu          CUTLASS NVFP4 TC GEMM + scale swizzle (linked, never called from src/)
kernels/elementwise.cu          rmsnorm, rope_rotate_half, gelu_tanh, add_inplace, softcap
include/tokenizer.h             pure-C++ gemma-4 BPE + chat/thinking/tool grammar
include/webui.h                 embedded single-file chat UI
include/safetensors.h           mmap safetensors reader
server/chat.cpp                 terminal streaming client; server/tok_test.cpp — tokenizer gate (5/5)
tests/                          test_loader, test_elementwise, test_attention, test_fp4_gemm,
                                test_linear_real, test_draft
tools/dflash_ref.py             PyTorch fp32 oracle for one draft_propose  ← settled the embed-scale question
tools/vllm_dflash_ref.py        vLLM acceptance-rate oracle
scripts/                        build.sh, gate_self.sh, gate_forward.sh, bench.sh, ref_forward.py (numpy
                                fp32 whole-model oracle), nvfp4_ref.py, gen_*_ref.py, capture_traces.py,
                                prompt_suite.json, serve_reference.sh (the vLLM ground-truth server)
reference/                      DFLASH_SPEC.md, ARCHITECTURE.md, MODELING_NOTES.md,
                                CUBLASLT_FP4_RECIPE.md, PERF_RESEARCH.md + the actual configs and
                                extracted modeling_gemma4.py / vllm gemma4.py sources
```

Build and run:

```bash
bash scripts/build.sh                    # -> build/forward
DFLASH=1 GEN=80 ./build/forward          # the champion benchmark
SERVE=1 PORT=8080 ./build/forward        # OpenAI server + WebUI at /
g++ -O2 -std=c++17 -I include server/chat.cpp -o build/chat -lpthread && ./build/chat
```

`build.sh` compiles `cutlass_moe.cu` separately (needs `~/cutlass/include`, `-gencode
arch=compute_110a,code=sm_110a`) only when stale, then links everything with `-lcublasLt`.
`--default-stream per-thread` is **required** for graph capture.

---

## 3. The model, exactly: gemma-4-26B-A4B-it

Two documents in `reference/` are the ground truth, and they were built from the real artifacts:
`ARCHITECTURE.md` (read from `RedHatAI/gemma-4-26B-A4B-it-NVFP4/config.json`, sha 24eef43) and
`MODELING_NOTES.md` (read from `modeling_gemma4.py` extracted from the vLLM container, cross-checked
against vLLM's `gemma4.py`, with `file:line` citations). Where the original project directive
disagreed, `ARCHITECTURE.md` marks the directive wrong with ⚠️.

### 3.1 Shapes and constants

`src/forward.cu:53-59` (all values confirmed against `reference/redhatai_nvfp4/config.json`):

```c
static const int H=2816, NLAYER=30, NHEAD=16, HD_S=256, NKV_S=8, HD_F=512, NKV_F=2;
static const int VOCAB=262144, NEXP=128, TOPK=8, MOE_INT=704, MLP_INT=2112, SWIN=1024;
static const float EPS=1e-6f, SOFTCAP=30.0f;
static const float EMB_SCALE=53.06599664f;              // sqrt(2816)
static inline bool is_full(int L){ return L==5||L==11||L==17||L==23||L==29; }
```

| Field | Value |
|---|---|
| hidden_size | 2816 |
| num_hidden_layers | 30 |
| num_attention_heads | 16 (both layer types) |
| vocab_size | 262144, `tie_word_embeddings: true` |
| rms_norm_eps | 1e-6 |
| hidden_activation | `gelu_pytorch_tanh` (⚠️ **not** silu — the draft uses silu, the target does not) |
| final_logit_softcapping | 30.0 |
| max_position_embeddings | 262144 |
| `hidden_size_per_layer_input` | **0** ⇒ **PLE is OFF** for this checkpoint |
| `num_kv_shared_layers` | 0 ⇒ no KV sharing |
| `use_double_wide_mlp` | false |

### 3.2 Attention — hybrid, and the two layer types differ in more than window

`layer_types` = 5×sliding then 1×full, repeated. **Full/global layers = indices [5, 11, 17, 23, 29]**
(5 of them); the other 25 are sliding.

| | sliding (25) | full/global (5) |
|---|---|---|
| head_dim | **256** | **512** (`global_head_dim`) |
| kv heads | **8** (GQA group 2) | **2** (GQA group 8) |
| sliding_window | 1024, causal, inclusive of self | none |
| RoPE type | `default` | `proportional` (p-RoPE) |
| RoPE theta | 1e4 | 1e6 |
| partial_rotary_factor | 1.0 | **0.25** |
| projections | q[4096,2816] k[2048,2816] v[2048,2816] o[2816,4096] | q[8192,2816] k[1024,2816] **no v_proj** o[2816,8192] |

- **`attention_k_eq_v: true`** applies only to full-attn layers (`MODELING_NOTES.md` §3a):
  `v_proj` does not exist; `V = v_norm(k_proj(x))` (no RoPE, no weight — `with_scale=False`),
  `K = RoPE(k_norm(k_proj(x)))`. The checkpoint confirms this: **25 `v_proj.*` tensors, not 30**.
  `src/forward.cu:611` implements V as a device-to-device copy of the raw k projection.
- **Attention scaling = 1.0** — no `1/sqrt(d)`, no `query_pre_attn_scalar`. Temperature is set
  entirely by the learned q_norm/k_norm RMS weights. `MODELING_NOTES.md` §3d calls this "UNUSUAL"
  and it was verified in the logit gate. `include/attention.h` documents it in the API comment;
  `forward.cu:625` passes `scaling=1.0f`.
- **No attention-logit softcap.** Gemma-2's ~50 cap is gone in gemma-4 text (`MODELING_NOTES.md` §3e).
- **Per-head QK norm**: q_norm and k_norm are RMSNorms of length head_dim with weights; v_norm
  normalizes with no weight. Order: project → reshape to (heads, head_dim) → norm → RoPE(Q,K only).

**p-RoPE, exactly** (`MODELING_NOTES.md` §4b, from `rope_proportional.txt`), for full layers:
`rope_angles = int(0.25 * 512 // 2) = 64`; `inv_freq_rotated[j] = 1/(1e6 ** (2j/512))` for j=0..63,
then zero-padded to 256. With `rotate_half` pairing (i, i+256), the rotated coordinates are indices
{0..63} ⊗ {256..319} — **128 of 512 dims rotate**, everything else is NoPE pass-through. Note the
exponent denominator is the **full head_dim 512**, unlike `default` which divides by its own dim.
`src/forward.cu` encodes this as `rope_angles = 64` for full layers, `128` for sliding.

### 3.3 Norms — plain `weight`, not `1+weight`

`Gemma4RMSNorm`: fp32, `eps` added **inside** the mean-square before the inverse sqrt,
`x * pow(mean(x²)+eps, -0.5) * weight`, weight initialised to ones. **Do not add 1.0** —
this differs from Gemma-2/Gemma-3 (`MODELING_NOTES.md` §2, §8c).

There are **12 norms per decoder layer** (constitution rounds this to "~10 RMSNorm/layer"), listed
in dataflow order in `MODELING_NOTES.md` §2: input_layernorm, q_norm, k_norm, v_norm(no scale),
post_attention_layernorm, pre_feedforward_layernorm, router.norm(no scale),
pre_feedforward_layernorm_2, post_feedforward_layernorm_1, post_feedforward_layernorm_2,
post_feedforward_layernorm, post_per_layer_input_norm (PLE branch — inactive here). Plus the final
`model.norm`. At M=1 each of these is a 1-CTA launch; `../dev/archive/LOOP_LOG.md` iter2 measures the base-decode
norms at ~9% and ~2.3 ms/token, which is what motivated the megakernel attempt.

### 3.4 MoE — a dense MLP *in parallel with* 128 routed experts

`enable_moe_block: true` ⇒ **every** layer runs both branches and sums them:

```
residual = h
h_mlp = mlp( pre_feedforward_layernorm(h) )          # DENSE MLP, intermediate 2112, gated gelu_tanh
hs1   = post_feedforward_layernorm_1(h_mlp)
probs,topw,topi = router(residual)                    # router reads the PRE-FF-norm residual
hs2   = post_feedforward_layernorm_2( experts( pre_feedforward_layernorm_2(residual), topi, topw ) )
h     = post_feedforward_layernorm(hs1 + hs2);  h = residual + h
h    *= layer_scalar                                  # bf16 [1] per-layer learned scalar
```

Router (`MODELING_NOTES.md` §5b): `hn = router.norm(h)` (RMSNorm, **no** weight);
`hn *= router.scale * 2816**-0.5`; `scores = hn @ router.proj^T` → **softmax over all 128**
(not sigmoid) → top-8 → **renormalize to sum 1** → `*= per_expert_scale[topi]`.
Experts: `gate/up → gelu_tanh(gate)*up → down`, `moe_intermediate_size = 704`, scaled by the
router weight and summed over the token's 8 experts.

Note the asymmetry that is easy to get wrong: the dense branch consumes
`pre_feedforward_layernorm(h)` while the MoE branch consumes `pre_feedforward_layernorm_2(residual)`
— two different norms over the same input.

### 3.5 Embedding scale and final logits

- `inputs_embeds = embed_tokens(ids) * sqrt(2816)`, and the scale is **cast to bf16 before the
  multiply** (53.0660 → nearest bf16). `EMB_SCALE=53.06599664f` in `forward.cu`.
- The tied lm_head uses the **raw** embedding table — it is *not* pre-scaled by the embed scale.
- `logits = 30 * tanh(logits/30)` after lm_head.
  **The production path does not apply this.** `forward.cu:987` says "no softcap; argmax-invariant".
  `SOFTCAP=30.0f` is referenced only by `k_lmhead`, which is never launched. This is fine for greedy
  (tanh is monotonic), but see §10 for what it means for temperature sampling.

### 3.6 The NVFP4 checkpoint — what ships at which precision

Format: `compressed-tensors`, `nvfp4-pack-quantized`, llm-compressor v0.14.1. **W4A4 on paper**:
both weights and `input_activations` are 4-bit, group_size 16, `strategy: tensor_group`, symmetric,
`scale_dtype: float8_e4m3fn`; activations `dynamic: local`, `observer: static_minmax`.
`recipe.yaml` ignore list: `lm_head`, `re:.*embed.*`, `re:.*router`, `re:.*vision_tower.*`.

Verified by reading the safetensors header directly (host-only, no GPU):

| Tensor class | Count | dtype |
|---|---|---|
| `layers.L.{self_attn.{q,k,v,o}_proj, mlp.{gate,up,down}_proj, experts.E.{gate,up,down}_proj}.weight_packed` | 4 per Linear | **U8** (two E2M1 codes per byte) |
| …`.weight_scale` | 4 per Linear | **F8_E4M3** (one per 16-element group) |
| …`.weight_global_scale` | 4 per Linear | **F32** (per tensor) |
| …`.input_global_scale` | 4 per Linear | **F32** (per tensor; unused — this engine is W4A16) |
| `embed_tokens.weight` | 1 | **BF16** (tied lm_head) |
| all `*_layernorm.weight`, `q_norm`, `k_norm`, `norm.weight` | | **BF16** |
| `router.proj.weight`, `router.scale`, `router.per_expert_scale` | 30 each | **BF16** |
| `layer_scalar` | 30 | **BF16** |
| `v_proj.*` | **25** (not 30) | — confirms `k_eq_v` on the 5 full layers |
| `vision_tower.*`, `embed_vision.*` | 27 layers | **BF16** — present in the file, **never loaded** (text-only) |

There is **no `v_norm.weight`** anywhere (it is a `with_scale=False` norm) and **no PLE tensors**
(`embed_tokens_per_layer`), consistent with `hidden_size_per_layer_input = 0`.

**Precision policy actually used by the engine: W4A16, everywhere.** FP4 weights × fp16
activations, no activation quantization. Constitution §10A: W4A4 was tested and measured at 5.5%
relative-L2 GEMM error (`TC_FEASIBILITY.md`) and the concern is that it would erode the acceptance
edge; W4A16 also dodges vLLM's ~75 µs per-step FP4 activation-quant cost at batch 1. The
`k_fp4_roundtrip` kernel (`forward.cu:237`, `W4A4_TAPS=1`) exists to emulate W4A4 for that
experiment. The cuBLASLt W4A4 path (`kernels/fp4_gemm.cu::nvfp4_gemm`) is tested but never called
from `src/`.

Global-scale convention (matters, and vLLM shipped a bug here per `PERF_RESEARCH.md`):
`gscale = 2688/amax` = `448*6/amax`, e4m3 group scale with exponent bias 7, plus a separate per-tensor
fp32 global scale. `include/nvfp4_quant.h` documents it; `forward.cu:961` uses it when quantizing
the embedding table.

### 3.7 Tokenizer and chat grammar

`include/tokenizer.h` is a from-scratch gemma-4 BPE reading the HF `tokenizer.json`
(vocab 262144, 514906 merges, byte_fallback). Normalizer: `' ' → U+2581 (▁)`. Byte fallback maps a
missing char to ids `238 + byte`. The merge loop is a naive O(n²) lowest-rank scan.

Control ids (`tokenizer.h:18-22`, re-resolved from the vocab at load):

```
eos=1  bos=2  <|turn>=105  <turn|>=106  <|channel>=100  <channel|>=101
<|tool>=46  <tool|>=47  <|tool_call>=48  <tool_call|>=49  <|think|>=98
<|tool_response>=50  <tool_response|>=51
stop_ids = {1, 106, 50}          # matches generation_config.json eos_token_id [1,106,50]
```

Chat template (`chat_prompt`, `tokenizer.h:63-79`):

```
<bos>
[ if system OR tools OR enable_thinking:
    <|turn>system\n [<|think|>\n] <system text> [<|tool>declaration:name{...}<tool|>]* <turn|>\n ]
per message: <|turn>{role, assistant→model}\n{content}<turn|>\n
<|turn>model\n                      # generation prompt
```

Note the evolution: `../dev/archive/SERVER_PLAN.md` §TOKENIZER SPEC describes the *old* Gemma
`<start_of_turn>`/`<end_of_turn>` template; the shipped code uses gemma-4's `<|turn>`/`<turn|>`
markers. Git `cc2753d` records the fix from a buggy "channel pre-fill" to a real `<|think|>`
system marker (per the fixed gemma-4 template, HF discussion #42).

Validated **5/5** against HF reference vectors (`server/tok_test.cpp`), e.g.
`"Hello, world!" → {9259,236764,1902,236888}`, `"\n" → {107}`, `"  spaces" → {138,35220}`,
plus a decode round-trip.

---

## 4. The DFlash draft head for this model

`reference/DFLASH_SPEC.md` is the authoritative spec — it was reverse-engineered from
`reference/dflash_src/vllm_dflash_full.txt` (three concatenated vLLM files: `dflash.py` 1–308,
`llm_base_proposer.py` 309–2026, `qwen3_dflash.py` 2027–2643) with line citations. `src/draft.cu`
implements it and departs from it in three measured places (§4.5).

### 4.1 What DFlash is

> "DFlash is *parallel* (single-forward) block-diffusion drafting. The drafter is a tiny 5-layer
> transformer that does **cross-attention**: its Keys/Values are precomputed from the **target
> model's fused hidden states** (context), and its Queries are `[bonus_token, MASK×15]`. One draft
> forward denoises all 15 MASK positions at once → 15 draft tokens." — `DFLASH_SPEC.md` preamble

This matters for reasoning about the profile: **the draft is ONE forward per block, not k
sequential forwards.** `RESEARCH_FINDINGS.md` §META-LESSON records a deep-research agent getting
this wrong (it modeled 15 sequential autoregressive forwards and concluded "draft is 76% of the
step, we're draft-depth-bound"); the nsys profile says the draft linears are **4.5%** and the draft
attention **1.8%**.

### 4.2 The checkpoint

`z-lab/gemma-4-26B-A4B-it-DFlash` — `DFlashDraftModel`, `model_type: qwen3`, **5 layers**, all BF16,
58 tensors, 0.859 GB. `draft_config.json` and `src/draft.cu:31-36`:

```c
H=2816, NL=5, NH=32, HD=128, NKV=8            // 32 q-heads / 8 kv-heads, GQA 4:1
QD=4096, KD=1024, FFN=5632, FCIN=6*2816=16896
VOCAB=262144, MASK=4, BLK=16, NDRAFT=15
EPS=1e-6, THETA=1e6, SCALE=1/sqrt(128)=0.08838834764831845
```

Per layer: `q_proj[4096,2816] k_proj[1024,2816] v_proj[1024,2816] o_proj[2816,4096]`,
`q_norm[128] k_norm[128]` (no v_norm, no biases), `mlp.{gate,up}[5632,2816] down[2816,5632]` with
**SiLU** (Qwen2MLP). Plus two model-level tensors: **`fc.weight [2816, 16896]`** (16896 = 6×2816,
fuses the 6 target taps) and **`hidden_norm.weight [2816]`**, and the final `norm.weight [2816]`.

`layer_types` = 4×sliding + 1×full; `sliding_window: 2048` — the draft's **own** window, independent
of the target's 1024. `final_logit_softcapping: 30.0`. `tie_word_embeddings: false`.

**The draft checkpoint contains no `embed_tokens` and no `lm_head`** — both are shared from the
target (`_maybe_share_embeddings` [1602-1664], `_maybe_share_lm_head` [1671-1720]). This engine
shares them from the **NVFP4-quantized** target embedding (`ewp/ews/egs`), which turned out to be a
τ win, not just a bandwidth win (§7.2).

### 4.3 Target taps

```
target_layer_ids = [1, 6, 11, 17, 22, 27]     # into the target's 30 decoder layers
mask_token_id    = 4
```

`src/forward.cu:112`: `static const int TAP_LAYERS[6]={1,6,11,17,22,27};`, captured by `k_tap`
(`forward.cu:108`) into `taps[mtok, 6, H]` — the **residual stream after** each of those decoder
layers (EAGLE3 "aux hidden state" convention), for every position the target forwards over
(prefill: all prompt positions; verify: all block positions). `../dev/archive/OPTIMIZATION_LOG.md` pre-loop history
records that the off-by-one tap set `[0,5,10,16,21,26]` was tried and is **worse** — the tap indices
are load-bearing.

### 4.4 The propose algorithm as implemented (`src/draft.cu:224-296`)

**(A) Context K/V — incremental.** For newly committed positions `[ctx_done, C)` only:

```
fused   = fc.weight @ concat6(taps)          # [C,16896] -> [C,2816]
fused_n = rmsnorm(fused, hidden_norm.weight)
for l in 0..4:
    K_l = k_proj[l] @ fused_n ;  V_l = v_proj[l] @ fused_n
    K_l = per_head_rmsnorm(K_l, k_norm[l]) ; K_l = rope(K_l, ctx_pos, theta=1e6, neox)   # V NOT roped
    store into Kc[l]/Vc[l] at slot ctx_pos[i]
```

**Every layer's context K/V is projected from the SAME `fused_n`**, not from a per-layer residual
stream — `DFLASH_SPEC.md` calls this "Key DFlash trick". Note the asymmetry the spec flags: context
K/V use `hidden_norm`, query tokens use the per-layer `input_layernorm`. The draft KV cache is fp32,
`slots = cap + BLK`. Making this incremental (was O(C) per propose) was +12.4% back in the parent
repo (`../dev/archive/OPTIMIZATION_LOG.md` cycle 7).

**(B) Query forward** over `BLK=16` positions: `qids = [next_token, MASK×15]`,
`qpos = qslot = [P0 .. P0+15]` with `P0 = ctx_pos[C-1]+1`. Then 5 qwen3 decoder layers:
`resid += h; hn = rmsnorm(resid, input_layernorm)` → q/k/v → per-head q_norm/k_norm → RoPE(neox
rotate-half, θ=1e6) → append query K/V to the cache at the block slots → attention over
**context slots ∪ block slots** with scale 1/√128 → o_proj → `resid += attn; hn = rmsnorm(resid,
post_attention_layernorm)` → SiLU MLP → next layer. Final: `resid += h; hfin = rmsnorm(resid, norm)`.

**(C)** `logits = tc_w4a16_gemm(target's NVFP4 embed, hfin[rows 1..15])`, then
`k_argmax_draft<<<k,256>>>` returns only the **first k** ids to host (avoiding a 15.7 MB D2H).
No softcap (argmax-invariant).

### 4.5 The three places the implementation departs from the spec — all measured

**(1) Q-embed scale: spec says unscaled; the code scales. The code won.**
`DFLASH_SPEC.md` §Q7 flags this as the single most likely silent mismatch:

> "In vLLM/Gemma the `* embed_scale` happens in the *target's* model.forward, not inside the
> embedding module, so the shared table returns the **raw, unscaled** row to the drafter. **To match
> this reference, feed the drafter UNSCALED embeddings.**"

`src/draft.cu:242-244`:

```c
// EMPIRICAL: draft wants SCALED embeddings (sqrt(2816)) — tau 4.0 vs 2.67 unscaled, despite
// DFLASH_SPEC.md Q7's UNSCALED guess. Default scaled; DEMB_OFF=1 reverts to unscaled.
float esc = getenv("DEMB_OFF") ? 1.0f : 53.06599664f;
```

Scaled is the default; `DEMB_OFF=1` reverts. `tools/dflash_ref.py` takes `argv[1]=='unscaled'` to
flip its `ESC` — that script is how the question was settled. (The kernel is still named
`k_embed_unscaled`, a leftover.)

**(2) Block attention: spec says non-causal everywhere; the code is causal for layers 0–3.**
`src/draft.cu:275`:

```c
k_attn<<<...>>>(ao,q,d->Kc[l],d->Vc[l],kslot_dev,nkeys,NH,NKV,HD,SCALE, nkeys-BLK, (l<4)?1:0);
// sliding layers 0-3: CAUSAL within block; full layer 4: bidirectional (per DFlash layer_types)
```

`DFLASH_SPEC.md` §Q2 says "**Attention is NON-CAUSAL** within the query block by default"
(`dflash_causal = dflash_config.get("causal", False)` [72]). `../dev/archive/OPTIMIZATION_LOG.md` pre-loop history
records the opposite as a **bug fix**: "DFlash causal-within-block mask fix (sliding layers): code
accept 3.3→4.1". `tools/dflash_ref.py` implements the same `causal = (l<4)` rule, so the oracle
agrees with the code, not with the spec text. **The spec text was not updated.** The file header
comment of `draft.cu:8-12` also still says "NON-CAUSAL attn over context ∪ block" — stale.
Inside `k_attn` the mask only restricts *block* keys (`if(causal && j>=blkstart && (j-blkstart)>i)
continue;`).

**(3) `sliding_window: 2048` is declared by the draft config and is not implemented.** All context
keys are attended in every layer, in all 5 layers. Not documented anywhere as a deliberate choice.

### 4.6 Block size and DK

The draft always runs the full `BLK=16` query block and always computes `NDRAFT=15` mask logits,
but the engine's default is **`DK=14`** (`forward.cu:1060`, clamped to [1,15]) — so the 15th draft
logit is computed and **discarded**, and the verify forward is **M = DK+1 = 15** positions
(`[bonus, d0..d13]`). `DFLASH_SPEC.md` (and vLLM) use `num_speculative_tokens = 15`, block 16.
`../dev/archive/SERVER_PLAN.md` asserts "BLK=16 baked-in (static const), DK=14 already-optimal"; the tuning that
produced DK=14 is referenced ("tau 4.0 → tuned DK=14 → tau 13.33", constitution §10A) but the
sweep itself is not recorded. τ 13.33 is therefore out of **14**, ≈95%.

`RESEARCH_FINDINGS.md` lever #2 proposes BLK 16→24/32 as the only remaining τ lever (we are
acceptance-saturated) — but the drafter was trained for block 16, and `../dev/archive/SERVER_PLAN.md` concludes
"bigger block needs draft retraining (separate ML project). Bank 118." **It was never tested.**

### 4.7 Verification and the decode loop

`engine_generate_dflash` (`forward.cu:741`): propose k ids → build the block `[tok, d0..d_{k-1}]` →
one target verify forward (M=k+1, CUDA-graph replayed) producing per-position argmax + taps →
accept the longest prefix where `draft_ids[i] == allarg[i]` → commit `n_accepted + 1` tokens (the
bonus is `allarg[na]`) → the committed positions' taps become the next propose's context.
This is the standard greedy longest-prefix rule of `DFLASH_SPEC.md` §Q4 — one target forward per
block, 1..k+1 tokens committed.

Three acceptance modes exist:
- **greedy** (default): `draft == target_argmax`.
- **sample-match** (`SAMPLE=1`, `TEMP>0`): `k_sample` does Gumbel-max sampling on the target logits
  (seeded by `__device__ g_sample_seed` via a splitmix64 hash of `seed ^ pos ^ vocab_id`, so it is
  graph-capture-safe), and acceptance is `draft == that sample`. This is **lossless** — committed
  tokens are genuine target samples; `temp=0` reduces to exact greedy. Git `ff652b7`.
- **typical acceptance** (`TYP_EPS>0`): `k_typical_prob` accepts when the target's softmax prob of
  the draft token ≥ ε. `TYP_EPS=0` is byte-identical to greedy (exact-parity verified). Measured a
  **no-op on primes** at ε 0.02–0.15, T 0.3–2.0, because the draft there has no acceptance headroom;
  kept as a production feature for low-acceptance workloads (`../dev/archive/OPTIMIZATION_LOG.md` cycle 16).

Stop tokens in the DFlash loop are hard-coded `1` and `106` — **not** `Tokenizer::stop_ids`, which
also contains 50 (`<|tool_response>`).

### 4.8 The draft is the moat, and it must stay bf16

The single most-repeated finding in this repo. Quantizing the draft body — to FP4 or FP8 —
collapses acceptance **13.33 → 11.14** in both cases:

- FP4 draft weights: `../dev/archive/RESEARCH_SYNTHESIS.md`, DFlash 84.3→75.1 **and** τ 13.33→11.14.
- FP8 e4m3 draft weights: `../dev/archive/LOOP_LOG.md` iter5, DFlash 84.7→71.1, τ 13.33→11.14 — "IDENTICAL drop".

`../dev/archive/LOOP_LOG.md` iter5 draws the general conclusion: the research literature's "FP8 keeps acceptance"
holds for typical EAGLE drafts, **not** for this block-diffusion draft that shares embed+lm_head and
injects target K/V. The only draft tensor that benefits from FP4 is the **shared lm_head**, which
is byte-bound at N=VOCAB *and* gains consistency with the verify head (§7.2).

The draft path is **not** graph-captured: it `cudaMalloc`s ~14 buffers and frees them every
`draft_propose` call, and its attention context grows. Cycle 15 measured the malloc churn as
**free** (the driver caches allocations) and the draft's launch sensitivity at only ~4%.

---

## 5. The engine: kernel stack

Constitution §3 lists the champion stack; here is what the code actually does.

### 5.1 Dense linear dispatch (`forward.cu:539 linear()`)

| M | kernel |
|---|---|
| M = 1 | `fp4_gemv` — CUDA-core warp-per-output GEMV, HW FP4 decode, fp16 acts, half2 FMA |
| 1 < M ≤ 16 and N ≤ 8192 | **`tc_w4a16_gemm`** (unless `NOTCVERIFY=1`) |
| otherwise (prefill) | `w4a16_gemm` — CUDA-core warp-per-output-column, U=4 K-unroll prefetch |

### 5.2 `tc_w4a16_gemm` — the champion kernel (`kernels/tc_verify_gemm.cu`, 135 lines)

Raw **`mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`** (fp16 MMA — see §11 on why not an FP4
MMA). One warp = 8 N-columns, `WARPS=1` per block for maximum grid fill. Key properties, each of
which was a separately measured win:

- **A fragment read directly from L2-cached global `x`** — no shared staging, no `__syncthreads`.
- **B dequantized in-register**: `__hmul2(tcv_fp4x2(byte), sc2)` straight into the MMA B fragment,
  with `global · block_scale` pre-folded into one `__hmul2`. No shared round-trip.
- **Offline weight repack** into `[n_block][k_group8][lane][16B]`, memoized in a
  `std::unordered_map` keyed by the source weight pointer. The repack does `cudaMalloc` +
  `cudaDeviceSynchronize`, so it **must** happen during warm-up, before graph capture — this is why
  the engine always runs one eager step before `cudaStreamBeginCapture`.
- **16-byte `int4` loads with `__ldcs`** (evict-first: weights are read once and must not thrash the
  L2 that activations reuse).
- **Software-pipelined weight prefetch, U=8.**

It handles all M≤16 dense verify GEMMs **and both lm_heads** (draft + verify). `tc_bf16_gemm` is
the same structure for bf16 weights (no dequant) and serves the draft linears.

The MMA fragment layout is written down in constitution §5.3, which is worth keeping: A row-major
8 halves/thread, B col-major 4 halves/thread, C 4 f32, with `gid=lane/4, t4=lane%4`.

### 5.3 MoE

`k_router_hn` → `k_router_scores` (warp per (token,expert)) → `k_router_top8` (128 threads, one per
expert: softmax → top-K → renorm → `per_expert_scale`) → `k_moe_invert` (expert→token map) →
`k_moe_gateup_grouped` (weight read once per expert, reused across the ≤4 tokens routed to it, U=4
prefetch) → `k_moe_down_bw` (weight-resident, per-assignment partials, **no atomics**) →
`k_moe_finalize` (8-way weighted reduce). Base M=1 decode uses the ungrouped `k_moe_gateup` /
`k_moe_down`.

`k_moe_invert` caps assignments at `EL=16` per expert per block and silently drops the overflow —
safe only because the grouped path is gated to `seq>1 && seq<=16`.

### 5.4 Attention

`sdpa_cache_kernel` (`forward.cu:129`) — **head-packed**: one block per **(query, kv_head)** so all
`G = nheads/nkv` sibling heads share a single KV read (was 4× GQA-redundant); q and accumulators
live in registers, only `red[G*hd]` in shared (16 KB even at hd=512/G=8 — a full layer's naive
shared-memory attention would exceed 48 KB). Online softmax, window or full, bf16 or fp8 KV,
`scaling = 1.0`. This mirrors what vLLM does via FlashInfer's `BatchPrefillWithPagedKVCache`
head-packing (constitution §6).

### 5.5 CUDA graphs

Two capture points, both preceded by one eager warm-up step:

1. **Speculative verify** (`forward.cu:761-775`) — the M=15 target forward + lm_head + argmax/sample.
2. **Base decode** (`forward.cu:1026-1048`) — the whole M=1 step including `k_advance`, which
   self-feeds the argmax token and increments `g_base` on-device.

Graph safety rests on: no `cudaMalloc` in the hot path (`DScratch` preallocated, `MAXM=16`), all
position dependence routed through `__device__ int g_base` and `g_sample_seed` pushed by
`cudaMemcpyToSymbolAsync`, lazy weight repacks done at warm-up, and `--default-stream per-thread`.
`NOGRAPH=1` falls back to eager.

Measured value: base decode graph was **+11.2%** (cycle 2). Whole-step graphing of DFlash was
measured at only **~7%** (93% GPU-busy, `../dev/archive/RESEARCH_SYNTHESIS.md`) and deprioritized.

### 5.6 Profile at the champion (nsys, DFlash ~118)

Constitution §2 / §8:

| component | share |
|---|---|
| **MoE verify (gateup_grouped 19% + down_bw 14% + base 8%)** | **~40%** |
| `tc_w4a16` (both lm_heads + dense verify) | 25% |
| `sdpa` (verify attention) | 10% |
| `tc_bf16` (draft linears) | 4.5% |
| `k_attn` (draft attention) | 1.8% |

**The bottleneck moved twice during the arc** and the constitution insists on re-profiling after
every win: it was draft-lm_head-dominated (22%) until that was routed to the TC kernel, then
draft-linear-dominated (12%) until those were too, and is now MoE.

### 5.7 Environment flags

`DFLASH`, `GEN`, `DK` (14), `CTX` (8192 CLI / 65536 server), `SERVE`, `PORT` (8080), `FP8KV`
(server default on), `SAMPLE`, `TEMP` (1.0), `SEED` (1), `TYP_EPS` (0), `NOGRAPH`, `NOTCVERIFY`,
`NOBF16TC` (draft.cu), `DEMB_OFF` (draft.cu), `MEGAKERNEL`, `PROF`, `SPARSITY`, `W4A4_TAPS`,
`DUMP`, `DBG_DRAFT`.

---

## 6. The server

`run_server` (`forward.cu:880-946`), pure C++, cpp-httplib + nlohmann/json vendored under
`include/third_party/`. Single global `std::mutex` serializes all GPU work — one request at a time,
no continuous batching.

- `POST /v1/chat/completions` — streaming SSE and non-streaming; `GET /v1/models`; `GET /` → the
  embedded WebUI. Completion id is the fixed string `"chatcmpl-thor"`.
- **Lossless temperature sampling** — Gumbel-max target sampling + sample-match acceptance (§4.7).
  `top_p` is accepted and **ignored** (`server/README.md`: "nucleus sort = TODO").
- **Prefix caching (Phase 2)** — single-slot LCP cache over `g_cached_ids`; re-prefills only the new
  suffix at `base=p`, reusing both KV **and** the DFlash taps. Bit-exact by construction (an LCP
  guarantees the cached KV[0:p) is identical). Verified: "system prompt reused across turns
  (41/55 tokens)". Limits: multi-user interleaving thrashes the single slot; an identical repeated
  prompt still re-runs one token (`p` is forced to `len-1`).
- **FP8 (e4m3) KV + configurable context (Phase 3)** — `CTX=65536`, `FP8KV=1` by default for
  `SERVE`; the benchmark and the gate stay BF16, which is the bit-exact path. All 30 layers allocate
  the full `CAP` — the 25 sliding layers do not use a `SWIN=1024` ring buffer (noted as future work).
- **Reasoning delineation (Phase 4)** — `ChanRouter` (`forward.cu:827`) parses
  `<|channel>thought…<channel|>` into `reasoning_content` vs `content` for both streaming deltas and
  the non-stream field; triggered by `enable_thinking` / `chat_template_kwargs.enable_thinking` /
  `reasoning_effort != "none"`, which inject the `<|think|>` marker into the system turn. This is the
  hand-written equivalent of vLLM's `--reasoning-parser gemma4`. Note `server/README.md`'s honest
  aside: **"gemma-4-A4B-it emits empty thoughts"** — the infrastructure is verified, the model
  mostly does not use it. (Git `cc2753d` claims verified real separated CoT after the `<|think|>`
  fix; the two statements are in tension — see §11.)
- **Tool calling (Phase 4)** — request `tools` → `<|tool>declaration:name{...}<tool|>` in the system
  turn; the model's `<|tool_call>call:name{args}<tool_call|>` is parsed into OpenAI `tool_calls` with
  JSON `arguments` and `finish_reason: "tool_calls"`. Verified end-to-end on `get_weather`.
  Streaming tool-call deltas and nested-arg escaping are not done.
- **Clients** — `include/webui.h` (offline single-file dark-theme chat: streaming, mini-markdown with
  code-copy, collapsible Thinking, settings, tok/s) and `server/chat.cpp` (streaming terminal REPL,
  multi-turn, dimmed thinking, tok/s).

---

## 7. The optimization ledger

### 7.1 Reconciling ../dev/archive/CHAMPION.md, ../dev/archive/OPTIMIZATION_LOG.md and the constitution

These three documents describe **three different eras** and only the constitution is current:

- **`../dev/archive/CHAMPION.md` is stale and internally inconsistent.** It reports BASE 44.5 (a cycle-21 number)
  alongside "DFlash (predictable, DK=14) 37.42 tok/s, OLD: 30.58, accept 11.14/14" (cycle-5/6
  numbers), dated 2026-06-30, commit "cycle23", with a "Target: well-exceed 110". It predates this
  repo's entire history. **Do not use it.**
- **`../dev/archive/OPTIMIZATION_LOG.md`** is the parent repo's append-only log, cycles 0–25, ending at DFlash 82 /
  base 44.5 and a "researched, decided AGAINST tensor cores" verdict — which the hybrid repo then
  overturned.
- **`CUDA_ENGINEERING_CONSTITUTION.md`** is the living state document: DFlash ~118, base ~45,
  τ 13.33, gate PASS. Its §4 and §10 are the consolidated won/lost/neutral ledger.

### 7.2 The arc, in order, with numbers

**Parent repo (`gemma-cuda-server`), decode 0 → 82** — constitution §10B/C, `../dev/archive/OPTIMIZATION_LOG.md`:

| Change | Effect |
|---|---|
| warp-per-output GEMV (vs block-per-output) | 6.45 → 11.02 (+71%) |
| HW FP4 decode + fp16 acts + half2 | 15.81 → 18.52 (+17%); MoE 18.52 → 21.40 (+15%) |
| MoE down: warp-per-(t,d), 8 experts fused into one shfl reduce | 21.3 → 25.9 (+22%) |
| parallel router top-8 (128-thread reduction-argmax) | +5.5% |
| device argmax (no 1 MB D2H + host argmax) | 29.89 → 30.73 (+2.8%) |
| **CUDA graph over the base decode step** | 30.73 → **34.18 (+11.2%)** |
| draft `k_linear_bf16` → half2 | DFlash 31.34 → 37.42 (+19.4%) |
| incremental draft context K/V | 37.42 → 42.05 (+12.4%) |
| verify lm_head → half2 warp-per-vocab | 42.05 → 51.22 (+21.8%) |
| grouped verify MoE gateup (weight read once per expert) | 51.87 → 59.72 (+15.1%) |
| MLP/K-unroll prefetch U=4 on gateup + w4a16 | base 34.18 → 34.9, DFlash 60 → 60.9 |
| **NVFP4 lm_head (base)** — quantize the tied embed to E2M1+e4m3 | base 34.78 → **44.5 (+28%)** |
| NVFP4 **verify** lm_head | DFlash 60.9 → 65.1 (+6.9%) |
| **NVFP4 draft lm_head** | DFlash 65.1 → **81.97 (+26%)**, and **τ 11.14 → 13.33 (+20%)** |

That last one is the most interesting result in the repo: quantizing the *draft's* shared lm_head to
the *same* NVFP4 table the verify uses made the draft's proposals agree with the target's FP4 argmax
— a **consistency** win stacked on the byte reduction. Cumulative FP4-lm_head effect: base +28%,
DFlash +35%.

**Hybrid repo, 82 → 118** — `BENCHMARK_COMPARISON.md` is the running log:

| Change | Effect |
|---|---|
| MoE weight-resident down (`k_moe_down_bw`, no atomics + finalize) | 82 → 84.3 (+2.8%) |
| MoE C_LUT → HW `cvt.e4m3` | 84.35 → 84.7 (+0.5%) |
| naive wmma TC verify GEMM | **−5%** — flag-gated, kept as the foundation |
| Marlin step 1: coalesced uint load + KT=32 | 85.65 → 87.5 (+2.2%) |
| **raw `mma.sync` TC with in-register FP4 dequant, default-on** | 85.65 → 88.4 (+3.3%) |
| MoE gateup U=4 prefetch | 88.65 → 89.7 (+1.2%) |
| MoE down U=4 prefetch | 89.7 → 93.1 (+3.7%) |
| dense-TC: remove shared A-stage + WARPS=1 grid-fill | 93.6 → 96.5 (+3.1%) |
| offline weight repack (coalesced 64 B/warp) | 96.4 → 97.9 (+1.6%) |
| tc weight prefetch U=8 | 95.5 → 97 (+1.9%) |
| **route draft+verify LM_HEADS through the tc kernel** (they were still CUDA-core w4a16, and were the biggest kernel at 22%) | **+9.7% → 104.5** |
| 16-byte `int4` loads + `__ldcs` evict-first | +3.5% → **108 (past vLLM 107)** |
| attention head-pack (KV read once) | +0.6% short-ctx (bigger on long ctx, unmeasured) |
| **draft linears through `tc_bf16_gemm`** | **+9.3% → ~118**, τ 13.33 unchanged, output bit-exact |

**Meta-lesson the constitution draws from this (§4 KEY META-LESSON):** several "LOST/NEUTRAL"
verdicts were **shallow**. The TC kernel was neutral on the lm_head *until* it had repack +
prefetch + 16-byte loads; then routing the lm_heads through it was the single biggest win of the
project. **Re-test dismissed levers after the kernel they'd use improves.**

### 7.3 LOST — measured, reverted, do not retry

| Lever | Result | Why |
|---|---|---|
| **Draft weights → FP4** | 84.3 → 75.1, **τ 13.33 → 11.14** | draft is precision-sensitive |
| **Draft weights → FP8 e4m3** | 84.7 → 71.1, **τ 13.33 → 11.14** | identical collapse; the bf16 draft IS the moat |
| FP8 e4m3 **lm_head** (per-row scaled) | base −5.5%, DFlash −3.7% | fp8→half decode is not free on Thor; must beat the decode cost |
| `uint4` lm_head loads | 51.22 → 49.2–50.1 | ILP/register cost offsets fewer load instructions; uint2 optimal |
| MoE gateup ILP (2 accumulators) | 34.18 → 33.55 | register pressure |
| MoE gateup **RB=2 multiple-outputs-per-warp** (a code-proven llama.cpp recipe) | 117.8 → 116.3 (−1.3%); with `launch_bounds(256,6)` → 107 (−9%) | gateup has 4 weight streams (gate+up × 2 rows) = 2× `down_bw`'s register load. `down_bw` already does RB=2 and works *because* it is single-stream |
| MoE down U=4 prefetch (8 experts' weights) | base −2%, DFlash −2.8% | already ILP-rich; 16 prefetch registers crushed occupancy |
| `cp.async` double-buffer on base MoE gateup | 44.6 → 42.4 (−5%) | SMEM round-trip + pipeline overhead > register-freeing gain at M=1 |
| **`cp.async` pipeline on the tc GEMM** (STAGES 2/3/4/6 all ~107 vs 108.3) | −1% | SoL showed 90.9% L1TEX scoreboard stall — *looked* like cp.async territory, but max-grid-fill (1 warp/block, 32768 blocks for the lm_head) already hides the stall at **block** level. Marlin's cp.async is for few large register-heavy blocks |
| base gateup 128-bit `uint4` widen | **crashed** (misaligned) | per-expert FP4 pointers are not 16 B aligned |
| Tree / multi-round verify | +16% tokens-per-draft but **−3% speed** (59.07 → 57.27) | each extra round is a full weight-bound forward; can't pay for itself at 95% base acceptance |
| TC **grouped** MoE (CUTLASS ex.92) | ~17% projected, not 2× | M-tile floor 128 vs ~2 tokens/expert = 64–84× padding waste |
| lm_head → FP4 in the hybrid era | ~−1% | `k_lmhead_batched` (bf16, block-per-vocab + shared reuse) was already efficient |
| Megakernel M1 (fused rmsnorm+QKV, sentinel counter) | 0% (break-even after grid-sizing fix) | bit-exact; the *pattern* was the win |
| Megakernel M2 (o_proj+post_norm+resid, grid-barrier) | −1.5% | grid-barrier concurrency cap |
| Full-step Stage-1 (grid-cooperative rmsnorm+QKV, `grid_bar`) | **−4.5%** | ~60 barriers/step; a full step needs ~390 (13 ops × 30 layers) |
| `fp4_gemv` M=1 prefetch U=4 | 44.5 → 43.9 | single-accumulator GEMV is register-sensitive |
| L2 persistence on the MoE activation (`cudaAccessPolicyWindow`) | neutral | Thor's L2 already holds the 90 KB activation |
| `__launch_bounds__(256,6)` on gateup | neutral | compiler spilled to fit |
| persistent draft query scratch (kill per-propose malloc) | identical (60.50 vs 60.51) | the driver already caches the allocations |

### 7.4 The two black swans that were killed by measurement, not argument

**Activation sparsity (TEAL / CATS)** was ranked the single highest-EV lever by deep research
(1.53× @40% sparsity, 1.8× @50%, stacks multiplicatively on NVFP4). Before building the gather
kernel, the `SPARSITY=1` probe (`measure_sparsity`, `forward.cu:19`, still in the tree, guarded,
zero production overhead) measured gemma-4's actual MoE activations at verify (M=15):

- `x2` (gate/up input, H=2816): hard zeros **0.1–0.2%**; retained-L2 at 50% sparsity 0.986, at 40% 0.994
- `hbuf` (down input, 704): hard zeros **0.5–0.9%**; retained-L2 at 50% 0.994, at 40% 0.997

Verdict: the activations are magnitude-**dense**, so TEAL/CATS is inherently soft/lossy here
(~1.4% x2-norm loss at 50%). The 40%-of-step bottleneck is the **verify** MoE, which must be
bit-exact (a perturbed verify silently produces a non-greedy output — nothing catches it), and the
draft, which *is* lossy-safe, is not the bottleneck. **Refuted, profile-first, before the build.**
Git `29ffa45`: "Profile-first saved the gather-kernel build."

**PowerInfer / hot-cold expert offload** was refuted analytically: every such technique needs a
two-tier bandwidth hierarchy (VRAM/RAM, HBM/DDR, DRAM/flash). Thor is **one** 273 GB/s unified pool
— a cold expert costs exactly what a hot one costs. `RESEARCH_FINDINGS.md` §5: "the seductive black
swan that isn't."

### 7.5 The "dead on arrival" list, with reasons

From constitution §8 DEAD/DON'T and `RESEARCH_FINDINGS.md`:

| Dead | Reason |
|---|---|
| FP8/FP4 draft | tested, τ collapse to 11.14 |
| Tree / multi-candidate verify | expert-union grows per branch; depth-dominated; measured net-negative |
| Swapping the drafter (EAGLE-3, Medusa, MTP, GRIFFIN) | all are τ **downgrades** (τ~6, ~1.9, ~5) vs block-diffusion's 13.33. We are acceptance-saturated at 95% — that whole frontier has no runway |
| MoE verify pruning (EVICT/FASER) | a tree-regime lever; our linear 95%-accept draft commits nearly everything and the accepted tokens need ~the whole expert union anyway |
| KV quantization below ~32K context | fixed overhead makes FP8-KV net-negative under ~7K; crossover >~50K. Breaks bit-exact |
| PowerInfer / hot-expert / neuron streaming | unified memory, no tier (above) |
| **tcgen05 / TMEM** | block-scaled NVFP4 MMA_M is locked to {128, 256}; at M=15 we are **60× under the compute roof** (crossover M≈900). Padding M→128 adds alloc/cp/ld latency for zero HBM saving. `mma.sync` is correct |
| DSMEM clusters | Thor is capped at a 2-SM cta_group |
| split-K / stream-K / TMA | occupancy/latency levers, not bandwidth levers; FlashInfer's split-KV doesn't help low-BW GPUs |
| `cp.async` on the tc GEMM | block-parallelism already hides the latency (above) |
| Warp-specialized ping-pong | large-batch only |
| Sub-4-bit / 2:4 sparsity on the target | breaks bit-exact; needs pruning + calibration |
| Full-step persistent megakernel | measured −1.5% to −4.5%; wrong regime (Hazy's 2.5× was Llama-1B with big ops; our M=1 MoE ops are tiny relative to the barrier) |

### 7.6 Roofline estimates — they changed five times

Be careful with any single roofline number in this repo; the estimate was revised repeatedly as the
byte accounting improved:

| Doc | Ceiling | Basis |
|---|---|---|
| `../dev/archive/CHAMPION.md` | ~190 tok/s base | 1.4 GB FP4 weight/token at 273 GB/s |
| `../dev/archive/LOOP_LOG.md` header | ~152 | — |
| `../dev/archive/LOOP_LOG.md` iter3 | **137** (89 at 65% efficiency) | research pass |
| `../dev/archive/OPTIMIZATION_LOG.md` cycle 16 | ~94 | 2.9 GB/step incl. 1.5 GB bf16 lm_head |
| `RESEARCH_FINDINGS.md` §3 (**latest, corrected**) | **~78** base at 200 GB/s achievable (107 at full 273) | NVFP4 = 0.5625 B/wt, ~4B active → 2.25 GB weights + ~0.3 GB KV = **2.55 GB/token** |

At the current base of 45 tok/s that is **58% of achievable roofline** — i.e. the engine is *not*
bandwidth-bound; the remaining base headroom is small-batch GEMV inefficiency.

---

## 8. The engineering constitution and methodology

Two documents encode the process. They are unusually specific about measurement discipline, which
is the part worth internalizing.

### 8.1 Non-negotiable rules (`CUDA_ENGINEERING_CONSTITUTION.md` §7)

1. **Back-to-back A/B for every change.** Thermal drift makes absolute numbers lie. The prescribed
   shape: `A=$(run5); git stash; rebuild; B=$(run5); stash pop; rebuild; A2=$(run5)` with
   median-of-5.
2. **Bit-exact gate on every change.** `DFLASH=1 GEN=8` must produce exactly
   `236778 236764 236743 236800 236764 236743 236810 236764` **and** `gate_self.sh` must PASS.
3. Commit each champion atomically with before/after numbers in the message. **Revert regressions
   and neutrals immediately** (the log shows this actually happened — many entries are "reverted").
4. Weight-repack caches keyed by source pointer, lazy on warm-up, so graph capture never sees a
   `cudaMalloc`.

### 8.2 Profiling discipline (§5.1)

`ncu --section SpeedOfLight` and read Memory% vs Compute%:

- both < 60% ⇒ **latency-bound** ⇒ prefetch (raise memory-level parallelism)
- "grid too small to fill device" ⇒ **more blocks** (fewer warps/block; `WARPS=1`)
- Memory > 60% with low Compute ⇒ **bandwidth** ⇒ 16-byte `int4` loads + evict-first + repack

This taxonomy is what produced the U-unroll prefetch wins, the `WARPS=1` grid-fill win, and the
repack win — in that order, each from a different SoL reading of the same kernel.

### 8.3 The research loop (`AGENTIC_OPTIMIZATION_METHODOLOGY.md`)

The central claim: **research is the MAP, the profiler is the TERRITORY, and you need both.**

- Research alone produced **confident errors**: an agent's #1 pick was FP8-draft (a tested dead-end),
  and another mismodeled the block-diffusion draft as 15 sequential forwards.
- Measurement alone produced **shallow local tuning** — "the 2%-scraping trap".
- Both big wins (+9.7% lm_heads, +9.3% draft linears) needed the profile to say *where* and the
  research to say *how*.

Operational consequences: the scarce resource is **serial GPU wall-clock** (speed tests cannot be
parallelized — concurrent model instances contend for the same 273 GB/s), so parallelism goes to
GPU-free research and only research-vetted experiments get serial GPU time. **Inject the
constitution's dead-end ledger and profile into the research prompt** — described as "the single
highest-leverage prompt move". Reserve a **black-swan budget**: exactly one high-variance
roofline-breaking bet per cycle (this project's was activation sparsity), so the loop does not
degenerate into 2% grinds. **Dimensions are not independent** — the bottleneck moves (Amdahl), so
"sum of improvements" overcounts and order matters.

---

## 9. Correctness gates and invariants

### 9.1 The invariant chain

1. **W4A16 everywhere** ⇒ DFlash output == base output == vLLM top-1 on confident prompts
   (constitution §10A). The precision unification is what makes bit-exactness possible at all.
2. **Verify guarantees the output.** The draft's numerics affect only τ, never the emitted tokens —
   which is why the draft could be moved to a bf16 TC GEMM with "output BIT-EXACT, τ 13.33
   unchanged" (git `28f6380`).
3. **Every optimization is bit-exact against the champion**, checked as an 8-token id sequence.
4. **Prefix caching is bit-exact by construction** (an LCP guarantees the reused KV is identical).
5. **`TYP_EPS=0` is byte-identical to greedy**; `temp=0` sampling reduces to exact greedy.
6. **FP8 KV is the one deliberately lossy path** and is excluded from the gate (benchmark and gate
   run BF16 KV).

### 9.2 The gates, and what oracle each uses

| Gate | Oracle |
|---|---|
| `scripts/gate_forward.sh` | **A live vLLM reference server on :8000** — `/v1/completions` with `max_tokens=1, logprobs=1`; compares the reference's top token to ours on 4 confident prompts. The header explains why only confident prompts are valid: `model.norm` amplifies quantization noise (588×), so gibberish/max-entropy prompts are not a valid gate |
| `scripts/gate_self.sh` | **Self-contained**: 4 prompts, greedy, substring-checked against hard-coded expected outputs — `"2, 3, 5, 7, 11, 13, 17, 19, 23, 29"` (GEN=40), `"Paris"`, `"Blue"`, `"Au"` (GEN=3) |
| `scripts/ref_forward.py` | **Pure-numpy fp32 whole-model reference** (own safetensors reader, NVFP4 dequant, proportional RoPE, router, experts) checked against the known vLLM top-5 for a fixed token sequence; dumps per-layer hidden states to `/tmp/refhs` for layerwise CUDA diffing |
| `tests/test_elementwise.cu`, `test_attention.cu` | numpy fp64 references from `gen_*_ref.py`; max-abs-diff, "GATE PASS only if ALL pass" |
| `tests/test_fp4_gemm.cu`, `test_linear_real.cu` | numpy float64 / `nvfp4_ref.py`, the latter on a **real checkpoint weight** |
| `tools/dflash_ref.py` | **PyTorch fp32 oracle for one `draft_propose`** — consumes the `DBG_DRAFT=1` dumps, recomputes fc → hidden_norm → 5 qwen3 layers with explicit per-head masks (`causal = l<4`), reports `hfin` max/mean diff and `match: n/k` on the proposed ids. This is how the embed-scale question was settled |
| `tools/vllm_dflash_ref.py` | vLLM's own DFlash acceptance counters → `mean_accept/block`, the τ oracle |
| `server/tok_test.cpp` | 5 HF reference vectors + decode round-trip; **5/5** |
| `tests/test_loader.cpp` | checkpoint structure, expects 47,648 tensors |
| `tests/test_draft.cu` | **structural only** — checks proposed ids are in range; no numerical oracle |

Validated numerics worth knowing: the NVFP4 loader + scale swizzle was validated at **maxrel 0.4%**
against an fp32 reference on the real 16 GB checkpoint; the CUTLASS wrapper hit **maxrel 0.0039**
(bf16 rounding) end-to-end after the swizzle converter.

### 9.3 A real hazard in the gate scripts

**`scripts/gate_self.sh:8` and `scripts/bench.sh:4` both `cd ~/gemma-cuda-server`, not
`~/gemma-cuda-hybrid`.** As written, they build/run the *sibling* repo's binary. `scripts/build.sh`
correctly `cd`s to `~/gemma-cuda-hybrid`. Any "gate PASS" claim made by running these scripts
verbatim from this repo needs re-checking. This is not mentioned in any document.

---

## 10. Known limitations, stated honestly

**Measurement scope**
- The 118 number is one prompt, one length (80 tokens), greedy, short context. The only controlled
  long-workload comparison in the repo (`REVERSE_ENGINEERING.md`) showed **58 tok/s on 300-token
  primes vs vLLM's 100**, and it was never redone after the arc to 118.
- No measurement of this engine on the 18-prompt suite exists. `traces/` holds vLLM's numbers only.
- Absolute tok/s drifts 94↔108 thermally; only back-to-back deltas are meaningful.

**Correctness / semantics**
- The production lm_head **omits the `30*tanh(x/30)` softcap**. Argmax-invariant for greedy, but
  `k_sample` (Gumbel) and `k_typical_prob` operate on **uncapped** logits, so temperature and
  typical-acceptance semantics differ from the HF/vLLM reference. Not documented anywhere.
- Stop tokens in the DFlash loop are hard-coded `{1, 106}`; `Tokenizer::stop_ids` also has `50`.
- The draft's declared `sliding_window: 2048` is not implemented.
- `top_p` is accepted and ignored.
- FP8 KV (server default) is lossy and outside the bit-exact gate; its quality delta was never
  quantified beyond "verified coherent at 64K ('Paris'; fruit list)".

**Engineering**
- Single-instance, mutex-serialized, single GPU, **no continuous batching**, no paged attention, no
  tensor parallelism, no multi-LoRA. `README.md` states this trade plainly: "this is a *specialist*,
  not a *generalist*".
- All 30 KV layers allocate the full `CAP`; the 25 sliding layers could use a 1024 ring buffer.
- Prefix cache is a **single** LCP slot — multi-user interleaving thrashes it.
- The draft path is not graph-captured and `cudaMalloc`/frees ~14 buffers per propose (measured free,
  but it blocks whole-step graphing).
- `main()`'s `run` lambda (`forward.cu:972-1000`) is a **verbatim duplicate** of `forward_block`
  (`forward.cu:706`) — the refactor left the original in the benchmark path, so changes must be made
  twice.
- Dead/unused code: `k_transpose`, `k_scale_rows`, `k_gather`, `k_lmhead`, `k_lmhead_batched_h2`,
  `expert_ffn`+`k_scatter_add`, `bf16_host()`, `Model::{wpacked,wscale,norm}`, `C_LUT`/`init_clut()`,
  host `rope_tables()`. `Scratch`/`SC` is allocated in both `main()` and `run_server()` and never
  dereferenced — a pure allocation leak. `attention_cached()` takes a `base` argument it ignores.
- Linked but never called from `src/`: `kernels/cutlass_moe.cu` (the whole CUTLASS TC path),
  `kernels/attention.cu::sdpa`, `kernels/nvfp4_quant.cu`, `kernels/fp4_gemm.cu::nvfp4_gemm`.
  `src/megakernel.cu::mega_attn_front` is implemented and never wired.
- `forward.cu:949` unconditionally runs `system("mkdir -p /tmp/cudahs")` at startup.

**Untried / unfinished**
- **BLK 16 → 24/32** — flagged as the only remaining τ lever (we are acceptance-saturated), bit-exact
  and low-effort to test, but the drafter was trained for block 16 so it needs a τ measurement.
  `../dev/archive/SERVER_PLAN.md` declines it as needing retraining.
- **Adaptive draft length** (AdaEDL entropy-stop) — lossless, +5–15% projected, never built.
- **FlashNorm** (fold RMSNorm into the next GEMM; gemma folds plain `weight`, not `1+g`) and fused
  add+RMSNorm / fused gate+up SwiGLU — the cheap bit-exact Tier-0 items, +1–3% and +3–6% projected.
  Never built.
- **MoE structural restructures**: fuse router+gather+GEMM+scatter; fuse gate+up into one GEMM;
  persistent expert-major kernel. The MoE is 40% of the step and has resisted both ILP levers
  (2-way split and RB=2) at the register wall.
- CUTLASS grouped-GEMM "step 3" (`kernels/BUILD_cutlass.md`) — steps 1 and 2 are validated, step 3
  was never started (and the padding analysis says it would not pay anyway).

---

## 11. Where the repo contradicts itself

Recorded because a new engineer will hit these:

1. **`../dev/archive/CHAMPION.md` is stale and self-inconsistent** — base 44.5 (cycle 21) next to DFlash 37.42 /
   accept 11.14 (cycle 5–6), dated 2026-06-30. The live numbers are DFlash ~118 / base ~45 / τ 13.33
   in the constitution header. Both files are in the repo; only the constitution is current.
2. **Constitution header vs its own §0.** The header says "DFlash ~118 … Arc 85.65→118 (+38%)";
   §0 two lines later says "Arc: 82 → 85.65 → **108** (+26%)". The 108→118 step is the
   draft-linears-to-bf16-TC win (git `f59daf3`); §0 was not updated.
3. **vLLM's number is workload-dependent and quoted three ways**: 107.5 tok/s / τ 9.21 / 85 ms on the
   80-token primes prompt (`BENCHMARK_COMPARISON.md`); 100–105 tok/s / τ 7.84 on the 300-token
   workload (`REVERSE_ENGINEERING.md`); "110" described as **unsourced** in `../dev/archive/OPTIMIZATION_LOG.md`
   cycle 25. The +10% headline uses the 107.5 figure, which is the apples-to-apples one.
4. **`DK=14` vs `num_speculative_tokens=15`.** `README.md` and the code use k=14 (verify M=15);
   `DFLASH_SPEC.md` and vLLM use 15 (verify M=16). The draft still computes all 15 mask logits and
   throws the last one away.
5. **Q-embed scale**: `DFLASH_SPEC.md` §Q7 says feed the drafter **unscaled** embeddings; the code
   scales by sqrt(2816) and cites τ 4.0 vs 2.67 as the evidence. The spec was not amended.
6. **Block causality**: `DFLASH_SPEC.md` §Q2 says non-causal within the block; the code is causal on
   layers 0–3 and non-causal only on layer 4, and the parent repo's log calls that a *bug fix* worth
   accept 3.3→4.1. `draft.cu`'s own file header comment still says "NON-CAUSAL".
7. **FP4 tensor-core availability.** `TC_FEASIBILITY.md` (2026-07-01) says CUTLASS 4.6's NVFP4
   tcgen05 GEMM "COMPILES … RUNS + PASSES correctness on Thor sm_110a", 41 µs at M=16.
   `FP4_COMPUTE_NOTE.md` (2026-07-03, HEAD) says FP4 compute is "BLOCKED on Thor sm_110 via EVERY
   path" and that "NVIDIA's own CUTLASS SM110 FP4 kernel is reported non-functional (dev-forum)".
   These are directly opposed. The later note is specifically about hand-written PTX
   (`mma.sync.kind::f8f6f4`, `tcgen05.*`, `mma…kind::mxf4.block_scale` all rejected by ptxas for
   `.target sm_110`) and cuBLASLt (**0 algos**, `CUBLAS_STATUS_NOT_SUPPORTED` — probed by
   `experiments/cublas_fp4_probe.cu`), while the earlier note is about CUTLASS's own codegen path
   with a patched host arch guard. Both cannot be true as stated; treat the shipping fact as
   authoritative: **the champion kernel uses the fp16 MMA `m16n8k16.f32.f16.f16.f32` with FP4
   dequantized in registers**, not any FP4 MMA.
8. **`../dev/archive/OPTIMIZATION_LOG.md` cycle 25 says "Thor sm_110a = tcgen05/TMEM only (NO warp mma.sync)"** —
   refuted by the hybrid repo's champion, which is built on warp `mma.sync`. That cycle-25 research
   verdict ("decided AGAINST tensor cores") was overturned by the entire hybrid arc.
   `RESEARCH_FINDINGS.md` compounds it by writing "our mma.sync.m16n8k16 / m16n8k64-for-NVFP4 is
   correct" — there is no NVFP4 mma in the code.
9. **Reasoning/CoT**: `server/README.md` says "gemma-4-A4B-it emits empty thoughts (model behavior);
   infra ready"; git `cc2753d` says "VERIFIED: enable_thinking gives real separated CoT (reasoning +
   answer both populate)". The commit is later, so the README line is probably describing the
   pre-fix behavior — but nothing states this explicitly.
10. **`README.md` claims a "~1.5 MB binary"**; `build/forward` as built is **3.1 MB**.
11. **`README.md` compares against "vLLM / SGLang"** throughout, but **no SGLang measurement exists
    anywhere in the repo** — SGLang appears only as a source of borrowed techniques (constitution §6:
    `BLOCK_SIZE_M=16`, single-CTA `moe_align`, the 1.78×-at-bs=1 MoE claim).
12. **`../dev/archive/SERVER_PLAN.md`'s tokenizer spec** describes the old `<start_of_turn>` Gemma template; the
    shipped tokenizer uses gemma-4's `<|turn>`/`<turn|>`. The plan was not updated after the fix.
13. **Roofline**: five different ceilings across the docs (190 / 152 / 137 / 94 / 78). The 78 figure
    in `RESEARCH_FINDINGS.md` §3 is the corrected one and supersedes the rest.
14. **`../dev/archive/STATUS.md` is the oldest doc in the tree** (2026-06-30) and describes Thor as "GB10,
    sm_120/121" and lists Phase 3.6 (DFlash) and 3.8 (server) as **not done**. Both are done. It also
    records the pre-hybrid vLLM trace results (mean τ 4.61, avg decode 41.7 tok/s) which are the
    `traces/` numbers, not comparable to the later 107.5 benchmark.

---

## 12. Twenty-minute orientation for a new engineer

1. Read `CUDA_ENGINEERING_CONSTITUTION.md` §0–§5 and §8. It is the only current state document.
   Ignore `../dev/archive/CHAMPION.md`; treat `../dev/archive/OPTIMIZATION_LOG.md` and `../dev/archive/STATUS.md` as the parent repo's history.
2. Read `reference/MODELING_NOTES.md` before touching any target kernel — the four things people get
   wrong on gemma-4 are: plain-`weight` RMSNorm (no `1+`), `scaling=1.0` attention, `k_eq_v` on the
   5 full layers, and the dense-MLP-in-parallel-with-MoE block with two different pre-norms.
3. Read `reference/DFLASH_SPEC.md` before touching `src/draft.cu`, then read §4.5 above for the three
   places the code deliberately diverges.
4. Before proposing an optimization, check constitution §4 and §10D and `RESEARCH_FINDINGS.md`'s
   DO-NOT list. Most obvious ideas have already been measured and lost, with a recorded reason.
   The one exception the constitution itself flags: **re-test a dismissed lever after the kernel it
   would use has improved.**
5. Workflow for any change: build → `DFLASH=1 GEN=8` must reproduce the 8 reference token ids →
   `gate_self.sh` PASS (fix its `cd` first) → back-to-back A/B median-of-5 with a stashed baseline →
   commit with before/after in the message, or revert.
6. Profile with `ncu --section SpeedOfLight` and use the three-way taxonomy in §8.2. Then re-profile:
   the bottleneck has moved every single time.
