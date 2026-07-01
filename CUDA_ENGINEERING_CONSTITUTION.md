# CUDA Engineering Constitution — gemma-4-26B-A4B + DFlash on Jetson Thor
**Ground-truth state & cumulative champion knowledge. Read this first. Maintained across context resets.**
Last major state: DFlash **~108 tok/s** (cool ~111), base ~45, gate PASS, tau 13.33. **We BEAT vLLM (107).** Repos: `~/gemma-cuda-hybrid` (active), `~/gemma-cuda-server` (v1.0 banked).

---

## 0. MISSION & CURRENT STATE
Purpose-built pure-CUDA NVFP4 + DFlash spec-decode server for `google/gemma-4-26B-A4B-it` on Jetson Thor. NEVER sacrifice correctness (`scripts/gate_self.sh` must PASS; every change verified BIT-EXACT vs champion on `/tmp/share/bench_primes.txt` GEN=8 tokens `236778 236764 236743 236800 236764 236743 236810 236764`).
- **Champion metric:** `DFLASH=1 GEN=80 ./build/forward` median-of-5-6, on `/tmp/share/bench_primes.txt` (prompt = "List the first 40 prime numbers, comma separated.").
- **Arc:** 82 (start) → 85.65 → **108** (+26% this arc). vLLM gemma DFlash = **107** (85ms/step, tau 9.21). Our tau **13.33** = 1.45x moat → **ceiling ~157 tok/s** if we hit vLLM's step time.

## 1. HARDWARE — Jetson Thor (VERIFIED, some corrected the hard way)
- **sm_110a** (Blackwell). CUDA 13.0; renumbered from sm_101 (12.8/12.9). Build: `nvcc -arch=sm_110a`. CUTLASS: `-DCUTLASS_NVCC_ARCHS=110a` (no `Sm110` C++ tag in `arch/arch.h` — stock SM100 kernels need arch-guard patching).
- **20 SMs** (NOT 96 — the "96" is max-config *tensor cores*). 2560 CUDA cores. **228 KB smem/SM** (datacenter-class, CC 11.0 Table 31). Has **TMEM + tcgen05 + TMA** (like B200, unlike DGX Spark/RTX50).
- **~273 GB/s LPDDR5X unified** — but GPU only gets **~73% achievable** (~200 GB/s memcpy measured); shared arbiter. LPDDR5X latency ~500-800ns (unmeasured; do a pointer-chase before sizing cp.async STAGES).
- **THERMAL THROTTLING IS REAL** after long sessions: absolute tok/s drifts 94↔108 for the SAME code. **ALWAYS measure back-to-back A/B (stash baseline, rebuild, compare)** — never trust absolute numbers across time. Rebooting kills the Claude session — do NOT reboot.
- LD_PRELOAD for vLLM docker: `/usr/lib/aarch64-linux-gnu/nvidia/libcuda.so.1`. Stop vLLM with `docker kill` NEVER `docker stop` (page-cache leak). Drop caches: `sudo sysctl -w vm.drop_caches=3`.

## 2. MODEL + DFLASH structure (constants in src/forward.cu / src/draft.cu)
- gemma-4: H=2816, NLAYER=30, **NHEAD=16**, sliding layers HD_S=256/NKV_S=8 (G=2), FULL layers (L=5,11,17,23,29) HD_F=512/NKV_F=2 (G=8), MoE 128 experts top-8, MOE_INT=704, VOCAB=262144. NVFP4 = E2M1 codes + E4M3 per-16 block scale + FP32 per-tensor global. gemma double-norm (~10 RMSNorm/layer).
- DFlash draft: 5-layer qwen3-style block-diffusion, BLK=16, k=14 spec tokens, shares frozen embed+lmhead (FP4). Draft weights MUST stay bf16 (see §4). Verify = M=15 forward.
- Profiled DFlash step (steady-state): **draft lm_head ~22% (biggest!)** + draft linears ~12% + verify dense-TC ~13% + MoE gateup ~16% + down ~11% + sdpa ~8-9%. lm_head verify (k_lmhead_batched bf16) is <0.5% (negligible).

## 3. CHAMPION KERNEL STACK (what's in place, all bit-exact)
- **NVFP4 lm_head** (the original breakthrough, +35%): tied bf16 embed quantized to E2M1+E4m3 (`g_ewp/g_ews/g_egs`, separate cudaMalloc → 256B aligned). base M=1 = `fp4_gemv`; **M=15 draft+verify lm_heads now route through the Marlin `tc_w4a16_gemm`** (kernels/tc_verify_gemm.cu) — this was +9.7%, the single biggest win.
- **Marlin-class TC verify GEMM** `tc_w4a16_gemm` (handles ALL M≤16 dense verify GEMMs N≤8192 AND the lm_heads): raw `mma.sync.m16n8k16.f32.f16.f16.f32`, 1 warp = 8 N-cols, **A fragment read direct from L2-cached global x** (no shared/sync), **B in-register FP4→fp16 dequant**, **offline weight REPACK** → 16-byte `int4` coalesced loads via `__ldcs` (evict-first). Cached by src-ptr, lazy repack on warm-up (graph-safe). WARPS=1 = max grid fill.
- **MoE grouped**: `k_moe_invert` (expert→token map) + `k_moe_gateup_grouped` + `k_moe_down_bw` (weight-resident, no atomics, per-assignment partials + `k_moe_finalize`). Both have U=4 K-unroll prefetch. W4A16 (fp16 act, NO activation quant → dodges vLLM's ~75µs FP4-quant-at-bs=1 trap).
- **Attention head-pack** `sdpa_cache_kernel`: one block per (query, kv_head), all G sibling heads, **KV read once** (was 4x GQA-redundant); qs/acc in registers, only `red[G*hd]` in shared (16KB even for hd=512/G=8).
- CUDA graphs on base + verify decode. HW FP4 decode `__nv_cvt_fp4x2_to_halfraw2`, HW e4m3 `__nv_cvt_fp8_to_halfraw` (register-only, replaced divergent constant C_LUT).

## 4. MEASURED DECISION SURFACE (what WON / LOST / NEUTRAL — do not re-litigate)
**WON (banked):** NVFP4 lm_head (+35%); bandwidth-down MoE (+2.8%); MoE C_LUT→HW cvt (+0.5%); **route lm_heads through tc (+9.7%)**; **16-byte int4 + __ldcs evict-first (+3.5%)**; tc weight prefetch U=8 (+1.9%); tc grid-fill WARPS=1 + no-shared-A (+3%); tc offline repack (+1.6%); MoE gateup+down U=4 prefetch (+5%); dense TC raw-mma in-register dequant (+3.3%); head-pack attn (+0.6% short-ctx).
**LOST (reverted, don't retry):** draft→FP4 AND draft→FP8 (both collapse acceptance 13.33→11.14 — the bf16 draft IS the moat); cp.async on the *base M=1 gateup* (SMEM overhead > register gain at M=1); fp4_gemv/gateup 128-bit widen (per-EXPERT FP4 ptrs not 16B-aligned — but the lm_head EMBED buffer IS, that's why 16B works there); grid-cooperative megakernel (grid-barrier too expensive for M=1 tiny ops); TC grouped MoE (128-tile padding waste at ~2 tok/expert); full-step megakernel per-unit (barrier overhead); FP4 verify lm_head vs bf16 k_lmhead_batched (neutral — bf16 already efficient block-per-vocab).
**NEUTRAL:** lm_head via TC when replacing already-efficient bf16 (memory-bound); base gateup micro-tuning (U=8, launch_bounds — register wall).
**LOST — cp.async pipeline on tc GEMM (2026-07-01, STAGES=2/3/4/6 ALL ~107 vs 108.3 __ldcs):** SoL shows 90.9% L1TEX scoreboard stall (mma waits for weight) — LOOKS like cp.async territory, but our MAX-GRID-FILL structure (1 warp/block, 32768 blocks for lm_head) hides the per-warp stall via BLOCK-level parallelism (SM switches to another resident block). Async win < shared-hop + pipeline-fill overhead (kg8=22 groups). Marlin cp.async is for FEW LARGE register-heavy blocks (warp-level hiding); wrong tool for many-tiny-blocks. Register-prefetch of int4 (UG) also regressed (register pressure). **__ldcs sequential int4 + max-grid-fill is champion; tc GEMM at structural limit ~108, 57% mem.** NOTE: pushing tc 57→90% mem = only ~+2% (tc weight ~1.1GB/step is small vs 123ms block); the +9.7% lm_head win was KERNEL EFFICIENCY (CUDA-core→tc), NOT memory %.
**KEY META-LESSON:** several "LOST/NEUTRAL" verdicts were SHALLOW — the tc kernel was neutral on lm_head UNTIL it had repack+prefetch+16B, then routing the (w4a16 CUDA-core) draft lm_head through it was +9.7%. **Re-test dismissed levers after the kernel they'd use improves.**

## 5. BEST KERNEL PATTERNS (reusable, proven on Thor)
1. **SoL profiling drives everything.** `sudo -E env DFLASH=1 GEN=20 ncu --launch-skip N --launch-count K --kernel-name regex:... --section SpeedOfLight --csv ./build/forward`. Read Memory% vs Compute%: both <60% = LATENCY-bound → **prefetch (raise MLP)**; "grid too small" → **more blocks (fewer warps/block, WARPS=1)**; Memory>60% Compute low = bandwidth → **16B int4 loads + evict-first + coalesce (repack)**.
2. **Marlin memory recipe (the 60→90% path):** (a) weights `__ldcs`/`cp.async.cg.L2::evict_first` (read once, don't thrash L2 that activations reuse); activations plain (evict-normal). (b) **16-byte int4 loads only** — sub-16B can't use cp.async.cg + wastes MSHR/LSU. (c) offline REPACK weight into `[n_block][k_group8][lane*16]` so a warp reads 512 contiguous B/group, feeding mma register-order directly (no ldmatrix for B). (d) cp.async 4-6 shared stages, `cp.async.wait_group stages-2`, prefetch stage `pipe+stages-1` two k-steps early (Thor has 228KB smem → go deep). (e) XOR swizzle `^(row%8)` for bank-conflict-free A.
3. **mma.sync m16n8k16 fragment layout** (verified): A row-major 8 halves/thread (a0=As[gid][2t4..], a1=[gid+8], a2=[gid][2t4+8..], a3=[gid+8]); B col-major 4 halves/thread (b0=W[n0+gid][2t4,2t4+1], b1=[2t4+8,2t4+9]); C 4 f32 (c0=D[gid][2t4], c1=[gid][2t4+1], c2=[gid+8][2t4], c3=[gid+8][2t4+1]). gid=lane/4, t4=lane%4.
4. **K-unroll register prefetch** (latency-bound M=1..16): issue U independent weight loads into `w[U]` before decode/FMA. U=4 for MoE, U=8 for tc GEMM.
5. **In-register FP4 dequant** (no shared round-trip): `__hmul2(tcv_fp4x2(byte), sc2)` straight into mma B fragment. Pre-fold `global·block_scale` into one `__hmul2`.
6. **Head-pack GQA**: one block per (q, kv_head), G sibling heads share the one KV read; qs/acc in registers, batched reduction (red[G*hd]) to keep syncs low.
7. **NVFP4 dequant alt (not yet used):** bit-splice E2M1→fp16 via `&0x70007000 >>3 | sign` + one `__hmul2` (cheaper than cvt; vLLM marlin/dequant.h:397).

## 6. vLLM / SGLang — what we learned & stole (code-level, cited in git log)
- vLLM gemma DFlash = **107 tok/s / 85ms step / tau 9.21 / 55% accept** (measured on THIS box, docker `vllm/vllm-openai:gemma-aarch64-cu130`, transformers 5.10.2, `--speculative-config method=dflash --attention-backend triton_attn`). We exceed it (108) with tau 13.33.
- **tcgen05 is NOT the answer** (research-confirmed): block-scaled NVFP4 locked to MMA_M∈{128,256}; at M=15 we're 60× under compute roof (crossover M≈900). Padding M→128 = zero extra HBM (weight streamed once) but adds tcgen05.alloc/cp/ld latency. `mma.sync` is correct.
- vLLM verify attention = FlashInfer `BatchPrefillWithPagedKVCache` with **GQA head-packing** (packed_qo_len = M×group, CTA_TILE_Q=64, grid=8 CTAs/kv-head, KV read once). Non-causal = just `causal=False` (drop diagonal test). We reproduced the head-pack.
- vLLM greedy verify = full M×VOCAB logits + dense argmax (`rejection_greedy_sample_kernel` = pure equality `draft!=target_argmax`). **Opportunity we have and they don't:** fused chunked-vocab argmax (never materialize [M,256k]) since we don't do temp>0 — but measured NEUTRAL for us (lm_head not bandwidth-bound; our bottleneck was the GEMM efficiency, now fixed by tc).
- SGLang 1.78x vLLM MoE at bs=1 via `M<=E → BLOCK_SIZE_M=16/GROUP_SIZE_M=1` + single-CTA smem `moe_align` (no global atomics) + trtllm tile_tokens_dim=8. MoE glue (router+topk+align+quant) ≈ expert matmul cost at bs=1.
- Files to model on: `IST-DASLab/marlin/marlin_cuda_kernel.cu`, vLLM `csrc/.../marlin/{dequant.h,gptq_marlin_repack.cu,marlin_template.h}`, flashinfer `attention/{prefill,scheduler}.cuh`, sglang `fused_moe_triton_*`, `moe_align_kernel.cu`.

## 7. METHODOLOGY (non-negotiable)
- **Back-to-back A/B for every change** (thermal). `run5(){ ...median NR==3...}; A=$(run5); git stash; rebuild; B=$(run5); stash pop; rebuild; A2=$(run5)`.
- **Bit-exact gate every change**: GEN=8 tokens must == the 8-token reference AND `gate_self.sh` PASS.
- Commit each champion atomically with before/after in the message. Revert regressions/neutrals immediately.
- Weight-repack caches keyed by src ptr; lazy on warm-up so CUDA-graph capture never sees a cudaMalloc.

## 8. ROADMAP TO 157 (ordered by expected impact) — REVISED after cp.async dead-end
0. **cp.async tc pipeline: DONE, DEAD-END** (see §4 LOST). tc GEMM at structural limit ~108. Don't retry.
1. **[NEXT] Draft bf16 TC GEMM** — draft is ~35% of step; draft linears `k_linear_bf16` (12%) are CUDA-core + LATENCY-BOUND (SoL 41% mem/21% compute, like the w4a16 lm_head was before tc). Build a `tc_bf16_gemm` (mma.sync.m16n8k16.f32.**bf16.bf16**.f32, NO dequant, repack weight) — same win pattern that gave the lm_head +9.7%. SAFE: draft numerics only affect PROPOSALS/tau, not output (verify corrects → bit-exact output guaranteed); but MUST check tau doesn't drop. Est +3-5%.
2. **MoE small-batch** — gateup/down are ~27%; repack per-expert weights into aligned buffers (per-expert ptrs NOT 16B-aligned → the tc-style repack fixes alignment AND enables 16B int4). SoL-profile first (is it latency or the invert/router glue?).
3. **Draft attention / draft FC** — the rest of the 35% draft cost.
4. **Whole-step CUDA graph** coverage (~20% launch overhead in general; we're pure-CUDA so partial).
5. **FP8 KV cache** (long-context; head-pack already cut KV 4x).
6. Re-profile after each — the bottleneck moves (it was lm_head all along, not what earlier filtered profiles showed).

## 9. GOTCHAS
- Per-expert / per-layer FP4 weight ptrs are NOT 16B-aligned (safetensors packing) → uint4 loads CRASH. Only the standalone-cudaMalloc'd embed (g_ewp) and REPACKED buffers are 16B-safe.
- N must be divisible by 8 for the tc (true for qkv/o/VOCAB); K divisible by 128 for the 16B-int4 repack (2816, 4096 OK).
- Full layers hd=512 → naive shared attention exceeds 48KB → keep qs/acc in registers.
- `k_lmhead_batched` (bf16) is ALREADY efficient (block-per-vocab + shared reuse) — don't "optimize" to FP4 (neutral).
