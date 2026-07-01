# vLLM DFlash vs pure-CUDA server — gemma-4-26B-A4B, IDENTICAL prompt (2026-07-01)
Prompt: "List the first 40 prime numbers, comma separated." (T=0 greedy). Same NVFP4 weights + same DFlash draft.

| engine            | tok/s | tau   | ms/step | accept-rate | notes |
|-------------------|-------|-------|---------|-------------|-------|
| vLLM 0.22.1 DFlash| 107.5 | 9.21  | 85      | 55%         | FULL decode cudagraph, triton_attn, flashinfer-cutlass NVFP4 MoE |
| our pure-CUDA     | 85    | 13.33 | 157     | 95%         | hand CUDA-core w4a16 + grouped MoE, graphed |
vLLM other tasks: code256=82 code512=48 reason256=34 (tau 4.5-7.6). Our step breakdown: w4a16 verify GEMM 39%, MoE 22%, draft 11%.

## VALIDATED CONCLUSIONS
1. TAU MOAT IS REAL: 13.33 vs 9.21 (1.45x) on identical prompt. Our greedy-exact verify accepts more per step.
2. STEP GAP IS REAL: 157 vs 85 ms (1.85x) = pure kernel efficiency (vLLM's Marlin/CUTLASS + full graph vs our CUDA-core).
3. CEILING = tau / step. At vLLM's 85ms step with our tau 13.33 -> ~157 tok/s = 1.46x vLLM's 107. The moat converts to real headroom.
4. Roofline: gemma NVFP4 ~13GB -> ~48ms/step weight-streaming floor. vLLM 85ms=1.8x floor; ours 157ms=3.3x floor -> ~70ms recoverable.

## PLAN: close the 1.85x verify-step gap (base + draft untouched; the moat is the draft, keep it)
Ranked by our profiled share + research (vLLM/SGLang steal-list):
 A. VERIFY GEMM (w4a16, 39% of step) -> Marlin-class NVFP4 at M=15: async weight loads, holds near-4x through M<=16.
    (CUTLASS grouped FP4 is BROKEN off-datacenter Blackwell per research -> hand Marlin-style per-expert W4 is the path.)
 B. MoE at M=15 (22%): grid occupancy (avoid idle-SM), minimize BLOCK_SIZE_M padding at ~2 tok/expert, keep kernel-granularity slack.
 C. Full-step CUDA graph over the verify forward (kill residual launch bubbles; vLLM uses FULL-on-uniform-decode).
 D. Ragged-Q / head-packed verify attention (FlashInfer-style; one varlen launch, KV-bandwidth-bound).
 E. (later) FP8 KV cache.
Target: 157ms -> ~85ms step -> ~130-157 tok/s (past vLLM's 107).
