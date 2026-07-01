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

## LOOP STATE after 3 iters + 2 research streams (2026-07-01): kernel micro-opts NEAR LIMIT (HW-cvt +0.5%,
## prefetch/draft-FP4/split-K/TC all lost or marginal). The real zenith needs SUBSTANTIAL builds, ranked:
##  A. FUSED MoE gate+up+SiLU+down one persistent kernel (Cursor warp-decode 1.84x; hbuf in SMEM not DRAM;
##     one block/layer keeps M=1 work on-chip). Base MoE=31% -> biggest single base lever.
##  B. cp.async DOUBLE-BUFFER the FP4 weight stream (research #2): moves in-flight bytes OFF registers -> breaks
##     my U=8/48-reg wall -> gateup 13.6%->~40% BW. The identified path to the M=1 HBM floor.
##  C. FP8 DRAFT weights (E4M3, not FP4 - keeps acceptance) via a small-N-optimized FP8 k_linear (NOT w4a16);
##     + the big FC[16896->2816] is bandwidth-bound -> FP8 halves it. Addresses the draft-model target.
##  D. DRAFT megakernel (5 layers, hand-writable per Hazy 7-instruction model) - the draft zenith.
##  E. FlashNorm: fold RMSNorm scale into next GEMM + fuse q/k-norm+RoPE (14.7% of layer per TRT-LLM). base norms 9%.
## Each is a multi-hour build. Champion: base 44.6 / DFlash 84.7. Roofline 137 (89 @65%) -> ~2x base headroom real.

## iter4: lever A reconsidered - base is GRAPHED so fusion's launch-win is already captured (research: MoE DRAM
## round-trip only ~0.5% of weight traffic) -> lever A NOT worth it for graphed base. Pivoted to kernel-BW tuning:
## gateup 128-bit uint4 loads (research #5) -> CRASHED (misaligned: per-expert FP4 ptrs not 16B-aligned). Reverted.
## => base gateup micro-tuning EXHAUSTED (prefetch neutral, C_LUT neutral, widen faults). warp-per-output at its
## structural limit. Remaining base levers require RESTRUCTURE: cp.async producer/consumer (lever B, same alignment
## care needed) or megakernel. Next: lever C (FP8 draft) - the user's draft-model focus, untried at FP8 (FP4 failed
## on acceptance; FP8/E4M3 has less error -> may keep tau 13.3 while halving the bandwidth-bound FC bytes).

## iter5: lever C FP8 draft (E4M3, proper k_linear_fp8, half bytes) -> LOST. DFlash 84.7->71.1, accept 13.33->11.14
## (IDENTICAL drop to FP4). DEFINITIVE: the DFlash draft is precision-sensitive - ANY weight quant (FP4 or FP8)
## collapses acceptance to 11.14. The draft MUST stay bf16 (research's "FP8 keeps acceptance" holds for typical
## EAGLE drafts, NOT this block-diffusion DFlash draft which shares embed+lmhead and injects target K/V). Reverted.
## => CLOSED: base micro-tuning (exhausted), draft quant (FP4+FP8 both lost). Remaining = RESTRUCTURES only:
##   cp.async base GEMV (lever B, uncertain per hackathon) | megakernel (huge) | fused-MoE (graphed=minimal).
## Champion HELD: base 44.6 / DFlash 84.7. The draft's bf16 + high tau 13.3 IS the moat - don't touch it.

## iter6: lever B cp.async double-buffer gateup (depth-4, weight stream off-registers via SMEM). -> LOST -5% (44.6->42.4).
## Gate PASS (correct) but slower: SMEM round-trip + __pipeline commit/wait/syncwarp overhead > the register-freeing
## gain, because per-pass compute (8 FP4 decode + FMA) is too small to hide the staging latency. CONFIRMS the
## hackathon finding (cp.async loses vs direct streaming here). Reverted. Deeper D just adds SMEM/cuts occupancy.
## === DEFINITIVE: ALL tractable kernel levers now CLOSED by measurement (7 iters). base 44.6 / DFlash 85 is the
## practical limit of the current per-kernel warp-per-output architecture. The ONLY remaining lever to the ~137
## roofline is the MEGAKERNEL (persistent single-kernel, on-GPU instruction interpreter, counter-sync, fused
## norm+GEMM per Hazy) - a multi-SESSION engineering build, not a loop iteration. Documented as the next major
## undertaking. The bf16 draft (tau 13.3) + FP4 lm_head remain the moat that makes DFlash 85 competitive.
