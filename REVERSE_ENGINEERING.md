# Reverse-engineering vLLM DFlash -> pure-CUDA hybrid (2026-07-01)

## Ground truth (measured this session)
- **vLLM DFlash = 100-105 tok/s** decode (gemma-4-26B-A4B-it-NVFP4, greedy primes, single-stream,
  image vllm/vllm-openai:gemma-aarch64-cu130). The "~110" target is REAL.
- **My pure-CUDA server = 82 tok/s** (DFlash) / 44 base. Gap ~22%.

## vLLM's architecture (from serve_reference.sh + load log + checkpoint inspection)
- MoE/dense: **flashinfer/cutlass FP4 tensor-core GEMM** (fp4_gemm autotuner runs at load).
- Attention: **TRITON_ATTN** (FA2/FA3 incompatible w/ gemma-4 heterogeneous head dims).
- KV cache: **BF16** (DFlash REJECTS quantized KV, vLLM issue #41559 — so fp8-KV is NOT part of the fast path).
- **Full CUDA graphs**: PIECEWISE (mixed prefill-decode) + FULL (decode) — captures the whole step incl. draft.
- **lm_head = BF16** (checkpoint embed is bf16, tied; vLLM does NOT quantize it). num_spec=15 (k=15, block 16).

## The two asymmetric advantages
- **MINE that vLLM lacks:** NVFP4 lm_head (4x lighter, my +28% base / +35% DFlash win). vLLM eats the full 1.5GB bf16.
- **vLLM's that I lack:** full-step CUDA graph + tuned attention/MoE + scheduler overlap. Collectively beats my
  hand-written kernels by ~22% DESPITE its bf16 lm_head.

## Key finding that reframes the plan
- tcgen05 (TC FP4) COMPILES on this box (CUDA 13.0, sm_110a) — a hand-written TC kernel is *possible*.
- BUT flashinfer's TC FP4 at M=128 measured only ~1.5 GFLOP/us = **overhead-bound at small M** (matches research:
  TC below its M-crossover). So a pure-CUDA TC MoE is NOT a guaranteed win at M=15 — the vLLM gap is more likely
  SYSTEM-level (full-step graph + attention + scheduler) than a single missing kernel.

## Paths to >110 (ranked by effort/confidence)
- **(B, highest confidence, fastest):** add the NVFP4 lm_head to the vLLM stack. vLLM 100 + lm_head byte-reduction
  (~+15-25% since bf16 lm_head is ~1.5GB/step) -> likely 115-125. Requires quantizing the tied embed in the
  checkpoint (compressed-tensors nvfp4) or a small vLLM lm_head patch. NOT pure-CUDA.
- **(A, pure-CUDA, high effort):** add full-DFlash-step CUDA graph (draft has GROWING ctx -> device-seqlen refactor)
  + optionally a tcgen05 TC MoE, keeping the FP4 lm_head. Matches vLLM's system wins by hand. Multi-day, uncertain.
