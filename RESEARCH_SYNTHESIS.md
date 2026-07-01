# Decisive synthesis (2 deep agents, 2026-07-01): the 2× is NOT tensor cores

## Verdict: decode is BANDWIDTH-BOUND. TC is the wrong hammer (confirmed by our own break-even benchmark).
Both agents converge. The 2× gap to vLLM = (a) host/launch tax [huge on ARM] + (b) memory-BW bubbles.
Our validated CUTLASS TC GEMM broke even with half2 because BOTH wait on HBM — TC is ~2500x over-provisioned
for decode's ~1.5 FLOP/byte arithmetic intensity. STOP chasing tensor cores for decode.

## The real levers, ranked for Thor (ARM, 273 GB/s):
1. **FULL-STEP CUDA GRAPH (#1, biggest on ARM).** ARM host is launch-bound 4x longer than x86 (Grace data:
   2.8x bs=1 latency from single-thread CPU). Our DRAFT is eager (~50 launches/step). Collapse draft+verify+accept
   into ONE graph: WHILE conditional node for the draft-K loop, IF/ELSE for accept/reject (cudaGraphSetConditional,
   CUDA 12.4+ conditional nodes), device-resident seqlen tensor + pre-allocated max KV (like our base-decode g_base).
   vLLM/SGLang/TRT-LLM all do device-seqlen + static buffers. Pure CUDA, no precision risk.
2. **BANDWIDTH-OPTIMAL MoE (#2).** Target = HBM roofline ~286us/layer (down); our half2 = 787us = 2.7x ABOVE floor
   because it re-reads each expert weight PER-TOKEN. Fix: reuse each weight byte across ALL 15 tokens routed to that
   expert (weight-resident, activations in regs/SMEM), fully-coalesced + double-buffered (cp.async) FP4 weight stream.
   CUDA-core dequant+FMA is fine (bandwidth-bound). Measure GB/s not TFLOPS; done at ~250 GB/s. NOT tensor cores.
   Layout: vLLM moe_align_block_size (BLOCK_M=16) or DeepGEMM masked (graph-capturable, no CPU per-expert counts).
3. **FP8/FP4 KV cache (#3).** Attention is memory-bound on KV; FP8 KV ~up to 2x. GQA head-stacking on M axis,
   SWA tile-skip, fused gemma soft-cap. No split-KV at short ctx.
4. Fused RMSNorm+residual+quant (#5): 1-4% cleanup (weight_bias=1 for gemma +1 centering).
Megakernel: deferred (1.0-1.7x vs tuned; MoE forces dynamic scheduling; draft+verify-in-one-megakernel unsolved).

## Our earlier TC work (steps 1-2) = a validated NEGATIVE result: TC proven to work on Thor but break-even for
## decode. Kept as a primitive (cutlass_moe.cu) for any future compute-bound (prefill/large-batch) path. NOT the
## decode lever. PIVOT the integration to #1 (graph) + #2 (bandwidth MoE).

## gemma-4 verification TODO before MoE build: confirm top-k (8?) + renormalization from HF transformers module.
