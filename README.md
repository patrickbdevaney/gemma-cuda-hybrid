# gemma-cuda-hybrid

A pure-CUDA inference engine and OpenAI-compatible server for
[`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) on one NVIDIA Jetson AGX Thor
(`sm_110a`).
- **Weights:** the [`RedHatAI/gemma-4-26B-A4B-it-NVFP4`](https://huggingface.co/RedHatAI/gemma-4-26B-A4B-it-NVFP4)
  checkpoint.
- **Speculative decoding:** the
  [`z-lab/gemma-4-26B-A4B-it-DFlash`](https://huggingface.co/z-lab/gemma-4-26B-A4B-it-DFlash) block-diffusion
  drafter.

The model, engine, tokenizer, chat grammar, HTTP server and web UI are one 3.1 MB binary. There is no PyTorch and
no Python on the request path.

This was the first of the author's pure-CUDA Thor servers. Its kernels, gates and optimization method were carried
into the later engines for DeepSeek-V4-Flash, Laguna, GLM-5.3 and Qwen3.8.

## Status (2026-07-26, final)

This repository is frozen. It is a research release from a single maintainer: one model, one GPU architecture, one
request at a time.

| Component | State |
|---|---|
| Engine: NVFP4 W4A16 target, bf16 DFlash drafter, CUDA graphs | Complete; greedy output bit-exact against the gate's reference tokens |
| Server: OpenAI chat API, streaming, sampling, prefix cache, FP8 KV, reasoning split, tool calls | Complete |
| Clients: built-in web UI, C++ terminal client | Complete |
| Benchmarks beyond the single headline prompt | Not run (see [Limitations](#limitations)) |
| Task accuracy on standard benchmarks | Not measured |

## Results

All numbers were measured on one Jetson AGX Thor.
- **Benchmark conditions:** one fixed prompt ("List the first 40 prime numbers"), greedy, 80 generated tokens,
  short context, back-to-back runs, median of 5–6.
- **Comparison baseline:** vLLM 0.22.1rc1 with DFlash, on the same weights, drafter, prompt and device
  ([`docs/BENCHMARK_COMPARISON.md`](docs/BENCHMARK_COMPARISON.md)).

| | This engine | vLLM + DFlash |
|---|---:|---:|
| Speculative decode | **~118 tok/s** | 107.5 tok/s |
| Mean accepted tokens per step (τ), 14-token drafts | **13.33** | 9.21 |
| Time per verify step | ~123 ms | 85 ms |
| Plain decode, no speculation | ~45 tok/s | 52 tok/s |

**How to read this.**
- The +10% comes entirely from acceptance: this engine accepts 13.33 tokens per step against 9.21, while each of
  its steps is slower.
- Plain decode is slower than vLLM's.
- Absolute tok/s drifts between 94 and 108 for the same binary as the device heats, so only back-to-back deltas
  are meaningful.
- On a longer, 300-token generation, an earlier build ran at 58 tok/s against vLLM's ~100. That comparison was not
  repeated after the optimizations that followed.

**Optimization history.** The parent repository
([gemma-cuda-server](https://github.com/patrickbdevaney/gemma-cuda-server)) took speculative decode from 5 to
82 tok/s. This repository took it from 82 to 118. Every step kept greedy output bit-exact
([`docs/WIKI.md`](docs/WIKI.md) §7):

| Change | Effect |
|---|---|
| Tensor-core W4A16 GEMM: raw `mma.sync` with FP4 dequantized in registers | 85.7 → 88.4 |
| 4-deep weight prefetch in the MoE gate/up and down kernels | → 93.1 |
| One warp per block for maximum grid fill; no shared-memory staging of activations | → 96.5 |
| Offline weight repack and 8-deep prefetch | → 97.9 |
| Both `lm_head`s through the tensor-core kernel, the largest single kernel at 22% of the step | → 104.5 (+9.7%) |
| 16-byte loads with an evict-first cache hint | → 108 |
| Drafter linears through a bf16 tensor-core GEMM | **→ ~118** (+9.3%), τ unchanged |

## Findings

**1. The drafter's precision is the speed advantage.**
- Quantizing the drafter's weights to FP4 or to FP8 dropped τ from 13.33 to 11.14 and lost 11–16% of speed. The
  drafter stays bf16.
- Quantizing the drafter's *shared* `lm_head` to the same NVFP4 table the target verifies with did the opposite:
  τ rose from 11.14 to 13.33 and speed by +26%. Drafter and target then agree on the FP4 argmax, a consistency
  gain on top of the byte saving.

**2. At 15 tokens per verify, every large kernel was latency-bound, not bandwidth-bound.**
- Maximum grid fill (one warp per block, up to 32,768 blocks) hid the load latency.
- A `cp.async` multi-stage pipeline, which the profiler's 90.9% scoreboard stall seemed to call for, measured −1%
  at every depth.

**3. Re-test a rejected lever after the kernel it depends on improves.** Routing the `lm_head`s through the
tensor-core kernel measured neutral early on. After repack, prefetch and 16-byte loads, it became the largest single
win in the project.

**4. FP4 tensor-core math is not reachable on `sm_110a` with CUDA 13.0.**
- `ptxas` rejects the FP4 `mma` and `tcgen05` forms for `sm_110`, and cuBLASLt returns no FP4 algorithms
  ([`experiments/cublas_fp4_probe.cu`](experiments/cublas_fp4_probe.cu)).
- The shipping kernel uses fp16 `mma.sync.m16n8k16` with FP4 dequantized in registers
  ([`docs/FP4_COMPUTE_NOTE.md`](docs/FP4_COMPUTE_NOTE.md)).

**5. Tree or multi-round verification does not pay at 95% acceptance.** Re-verifying the draft tail raised τ by
16% but cost 3% speed, because every extra round is a full weight-bound forward. The same result later held on
GDN-hybrid models in vLLM.

**6. Several ideas were measured and rejected, each with a recorded reason**
([`docs/WIKI.md`](docs/WIKI.md) §7.3):
- an FP8 `lm_head`;
- a second accumulator per row in the MoE gate/up kernel, and two output rows per warp, both of which hit the
  register limit;
- a CUTLASS grouped MoE, which pads 64–84× at ~2 tokens per expert;
- three megakernel variants;
- activation sparsity, which measured dense: 0.1–0.9% exact zeros.

## Features

**Engine**
- NVFP4 W4A16: E2M1 weights, E4M3 group-16 scales and an FP32 global scale, decoded with hardware FP4 and FP8
  intrinsics.
- Grouped MoE that holds weights resident: 128 experts with top-8, alongside a dense MLP in parallel, without
  atomics.
- Grouped-query attention that reads KV once per KV head. Sliding layers use head dim 256 with 8 KV heads; full
  layers use head dim 512 with 2.
- DFlash drafter: 5 layers, 14 draft tokens per block, sharing the target's embedding and `lm_head`, with taps from
  6 target layers.
- CUDA-graph capture of the verify step.

**Server** (`SERVE=1`)
- **API:** `POST /v1/chat/completions` with SSE streaming, `GET /v1/models`, and a web UI at `/`.
- **Lossless temperature sampling.** Gumbel-max sampling plus sample-match acceptance, which reduces to exact greedy
  at temperature 0.
- **Prefix caching.** A single slot reuses the longest common prefix across turns.
- **KV cache.** FP8 (e4m3), with 64K context by default (`CTX`, `FP8KV`).
- **Reasoning.** `enable_thinking` adds gemma-4's `<|think|>` marker. Thought channels are routed to
  `reasoning_content` and the answer to `content`.
- **Tool calling.** Request `tools` are rendered as gemma-4 declarations. `<|tool_call>` output is parsed into
  OpenAI `tool_calls` with JSON arguments and `finish_reason: "tool_calls"`.

**Text front end.** A pure-C++ gemma-4 BPE tokenizer, validated against Hugging Face (5/5 cases).

**Clients.** A single-file web UI with no CDN or build step, streaming markdown and collapsible thinking. A C++
terminal client with streaming multi-turn chat.

## Getting started

Requires a Jetson AGX Thor (`sm_110a`, CUDA 13.0). The build links cuBLASLt, and the CUTLASS headers are needed at
`~/cutlass/include`. Weights take about 16.4 GB, plus 0.86 GB for the drafter.

```bash
huggingface-cli download RedHatAI/gemma-4-26B-A4B-it-NVFP4 --local-dir ~/models/gemma-4-26B-A4B-it-NVFP4
huggingface-cli download z-lab/gemma-4-26B-A4B-it-DFlash   --local-dir ~/models/gemma-4-26B-A4B-DFlash

bash scripts/build.sh                     # -> build/forward
DFLASH=1 GEN=80 ./build/forward           # the headline benchmark
bash scripts/gate_self.sh                 # correctness gate
SERVE=1 PORT=8080 ./build/forward         # OpenAI server + web UI at http://localhost:8080
g++ -O2 -std=c++17 -I include server/chat.cpp -o build/chat -lpthread && ./build/chat
```

These are the default paths. The target checkpoint can also be passed as the first argument; the drafter path is
fixed in `src/forward.cu`.
Server environment: `CTX` (default 65536), `FP8KV` (default 1), `DK` (draft tokens, default 14). Per request:
`temperature`, `max_tokens`, `stream`, `enable_thinking`, `tools`.

## Compatibility

| Platform | State |
|---|---|
| Jetson AGX Thor (`sm_110a`), CUDA 13.0 | Tested |
| Other Blackwell parts | Not tested. `mma.sync` and the FP4 conversion intrinsics exist there, but grid-fill and prefetch depths are tuned to Thor's 20 SMs |
| Ampere or Hopper | Not supported: no hardware FP4 decode |
| Other checkpoints | This model only. Shapes, tokenizer, chat grammar and drafter are model-specific |

## Limitations

- **The headline is one prompt.** 118 tok/s is an 80-token, short-context, greedy run. The engine has not been
  benchmarked on code, reasoning, long context or the 18-prompt suite in `traces/`, which holds vLLM's reference
  numbers only.
- **No accuracy evaluation.** Greedy output is bit-exact against its own reference tokens, but task accuracy was
  never measured. In a later unrelated test (October 2026, under vLLM), this same RedHat checkpoint scored far below
  the base model on comprehension (Belebele-en 0.43), with vLLM warning that q/k/v carry different global scales.
  Whether this engine, which reads each projection's scale separately, is affected is unknown.
  [`nvidia/Gemma-4-26B-A4B-NVFP4`](https://huggingface.co/nvidia/Gemma-4-26B-A4B-NVFP4) is the better-measured
  alternative, though this engine has not been run on it.
- **Sampling semantics differ from the reference.** The production `lm_head` omits gemma's `30·tanh(x/30)`
  softcap. Greedy output is unaffected, but temperature sampling acts on uncapped logits. `top_p` is accepted and
  ignored.
- **FP8 KV (the server default) is lossy** and outside the bit-exact gate.
- **Single stream.** One mutex-serialized request at a time, a single prefix-cache slot, no batching.
- **Code debt** is listed in [`docs/WIKI.md`](docs/WIKI.md) §10. It includes a duplicated forward loop in the
  benchmark path and linked-but-unused CUTLASS and megakernel code.

## Repository map

| Path | Contents |
|---|---|
| [`src/forward.cu`](src/forward.cu) | Model, target kernels, engine, CUDA graphs, HTTP server |
| [`src/draft.cu`](src/draft.cu) | DFlash drafter |
| [`kernels/`](kernels/) | `tc_verify_gemm.cu` (the tensor-core W4A16 and bf16 GEMMs), FP4 GEMV, attention, elementwise, CUTLASS experiments |
| [`include/`](include/) | Tokenizer and chat grammar, web UI, safetensors reader, vendored HTTP and JSON libraries |
| [`server/`](server/) | Terminal client, tokenizer validation |
| [`tests/`](tests/), [`scripts/`](scripts/), [`tools/`](tools/) | Unit tests, gates, NumPy and PyTorch oracles, vLLM reference server, benchmark scripts |
| [`reference/`](reference/) | Architecture and modeling notes with `file:line` citations; DFlash spec; checkpoint configs |
| [`traces/`](traces/) | vLLM reference outputs for an 18-prompt suite |
| [`experiments/`](experiments/) | cuBLASLt FP4 probe |
| [`docs/`](docs/) | Start at [`WIKI.md`](docs/WIKI.md), which reconciles every other document and lists their contradictions. Also the [engineering constitution](docs/CUDA_ENGINEERING_CONSTITUTION.md) (won/lost/neutral ledger), the [optimization methodology](docs/AGENTIC_OPTIMIZATION_METHODOLOGY.md), the vLLM comparison, research findings and FP4 notes |
| [`dev/archive/`](dev/archive/) | Superseded status files, plans and the parent repository's optimization log |

## License

- **Code in this repository:** MIT ([`LICENSE`](LICENSE)).
- **Third-party material keeps its own license:**
  - `include/third_party/` (cpp-httplib, nlohmann/json) is MIT.
  - `third_party/72a_blackwell_nvfp4_bf16_gemm.cu` is NVIDIA CUTLASS, BSD-3-Clause.
  - Upstream sources kept for reference under `reference/` (Transformers, vLLM) are Apache-2.0.
- **Weights** are not distributed here. Gemma 4, the RedHatAI NVFP4 checkpoint and the z-lab drafter are
  Apache-2.0.

## Citation

```bibtex
@misc{devaney2026gemmacuda,
  title  = {gemma-cuda-hybrid: a pure-CUDA DFlash server for Gemma-4-26B-A4B NVFP4 on Jetson AGX Thor},
  author = {Devaney, Patrick},
  year   = {2026},
  url    = {https://github.com/patrickbdevaney/gemma-cuda-hybrid}
}
```

Builds on Gemma 4 (Google), DFlash (z-lab, arXiv:2602.06036), Marlin-style mixed-precision GEMM design, and RedHatAI's
NVFP4 quantization.
