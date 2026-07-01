# Autonomous zenith loop — base decode + draft (2026-07-01), targeting >100 tok/s
Baseline: base 44 tok/s (~29% of ~152 roofline), DFlash 84.3. Champion metric per mode; correctness gate MUST pass.
## Levers already closed (see RESEARCH_SYNTHESIS.md): TC (wrong hammer), full-graph (7%), draft->FP4 (hurts accept),
## MoE grouped (padding). Base-decode M=1 MoE is latency/compute-bound at 13.6% BW = the biggest headroom (3.2x).
## Research running: M=1 FP4 MoE zenith kernel (hit the HBM floor).

## iter1: fp4_gemv M=1 K-unroll prefetch U=4 -> LOST (44.5->43.9). Single-accumulator GEMV is register-sensitive;
## large-N (qkv N=4096, lmhead N=262144) already has enough warps for MLP -> prefetch just adds regs. Reverted.
## Base decode profile: fp4_gemv(dense+lmhead) 32.7%, MoE gateup 19.3%+down 12% =31%, rmsnorm 9% (launch-inflated,
## hidden in graph). Await M=1-GEMV-zenith research for the specific floor-hitting technique.

## iter2: base rmsnorm = ~10 gemma-4 double-norms/layer, each 1-CTA/1-SM at M=1 (latency-bound, ~2.3ms/token).
## Structural fusion won't fix it (sequential, different data) — needs a MEGAKERNEL (fuse the layer). Research pending.
## LOOP STATE: 2 research streams running (M=1-GEMV-zenith + draft/megakernel-zenith). Kernel-guessing regresses
## (fp4_gemv prefetch, draft-FP4 both LOST) -> awaiting PROVEN techniques from research before next grind.
## base 44.5 / DFlash 84.3 intact. Next: grind per research findings (M=1 GEMV floor kernel + megakernel MVP).

## iter3: MoE C_LUT -> HW cvt.e4m3 (research #1, kill divergent constant lookup). base 44.35->44.6, DFlash 84.3->84.7
## (marginal +0.5%; my constant-cache lookup wasn't the big bottleneck the research assumed, but it's cleaner). KEPT.
## RESEARCH DELIVERED (2 streams): CRITICAL - Thor = 20 SMs not 96 (SM-idle 5x smaller; launch/megakernel is #1).
## NVFP4 roofline = ~137 tok/s (89 @65%); base 44 = ~50% -> ~2x headroom. Ranked levers:
##  1. CUDA graphs (base done; DFlash draft eager but only ~7% measured). 2. FP8 DRAFT weights (not FP4 - keeps
##  acceptance, halves draft bytes, ~2x draft). 3. Megakernel (draft first, hand-writable; base via MPK). 4. FlashNorm
##  fold RMSNorm into GEMM (12-35% norms). 5. Fuse gate+up+SiLU register-intermediate (Cursor warp-decode 1.84x).
## Kernel micro-opts near limit (marginal). Next: FP8 draft (real ~2x draft potential) or fuse-gate+up (Cursor).
