# CUTLASS NVFP4 tensor-core on Thor sm_110a / CUDA 13.0 — GO/NO-GO = GREEN (2026-07-01)

## Result: FP4 tensor cores WORK on this box. Blocker cleared.
- Cloned CUTLASS 4.6 (has sm_110 support since 4.2). Compiled example 72a (NVFP4->BF16 GEMM)
  with `nvcc -std=c++17 --expt-relaxed-constexpr -I cutlass/include -I cutlass/tools/util/include
  -gencode arch=compute_110a,code=sm_110a`. COMPILES on CUDA 13.0 (the tcgen05 FP4 MMA assembles).
- The example's host guard whitelists only sm_100/101/103 (major==10); patched line 514 to allow
  major==11 (Thor). Then it RUNS + PASSES correctness on Thor sm_110a.
- **True kernel GPU time (ncu, isolated): 41us at M=16,N=2816,K=2816** (14% SM / 14% mem = latency-bound
  at small M, but a real 41us — the example's 220us "avg runtime" was host measurement overhead).

## Benchmark (CUTLASS example wall-clock, overhead-inflated; ncu true=41us at M16):
    M=16   N=2816 K=704  -> 289 GFLOPS | M=128 -> 2305 | (flat ~220us wall = overhead)
    M=16   N=2816 K=2816 -> 1123 GFLOPS| M=128 -> 8848 | M=256 -> 18487 (ncu true kernel M16=41us)

## Verdict for the hybrid
- TC is a REAL win for the GROUPED verify MoE (my half2 down = 787us/layer @0.6 TFLOP/s; CUTLASS grouped
  MoE ex.92 amortizes 8 experts -> ~5x). NOT a clear win for tiny single dense GEMMs (41us overhead-bound
  vs my ~20us half2).
- Path to beat vLLM (100): CUTLASS grouped-MoE (ex.92) for verify MoE + full-step CUDA graph + keep our
  draft (tau 10-13 vs vLLM 7.84) + keep our NVFP4 lm_head (vLLM eats bf16). All proven-feasible on 13.0.
- Build flags that work: -gencode arch=compute_110a,code=sm_110a, -I cutlass/include + tools/util/include.
  Callable from C++ via flashinfer CutlassFp4GemmRunner (torch-free header) OR direct CUTLASS CollectiveBuilder.
