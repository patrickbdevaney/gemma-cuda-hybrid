# cutlass_moe.cu — CUTLASS NVFP4 TC GEMM wrapper (STEP 1 DONE, validated 2026-07-01)
Callable API (extern "C"):
  cutlass_nvfp4_gemm(D_bf16, A_fp4, B_fp4, SFA_swz, SFB_swz, alpha, M,N,K, workspace, stream) -> rc(0=ok)
  cutlass_sfa_bytes/cutlass_sfb_bytes/cutlass_workspace_bytes(M,N,K)
VALIDATED: all-1.0 input (E2M1 code2=1.0 byte 0x22, ue4m3 1.0 = byte 0x38) -> D = K = 2816.0 EXACT.
  scale 0x40(=2.0) -> D=K*4. Numerically correct on Thor sm_110a / CUDA 13.0. Tile 128x128x256 1SM.
Build test: nvcc -std=c++17 --expt-relaxed-constexpr -DCUTLASS_MOE_TEST -I ~/cutlass/include
  -I ~/cutlass/tools/util/include -gencode arch=compute_110a,code=sm_110a kernels/cutlass_moe.cu -o test
Link into server: compile WITHOUT -DCUTLASS_MOE_TEST -> object; link with forward.o.
NEXT (step 2): convert our E2M1 + LINEAR e4m3 group-16 scales -> CUTLASS swizzled SFA/SFB layout
  (Sm1xxBlkScaledConfig::tile_atom_to_shape_SF*). Validated so far only with UNIFORM scales (swizzle-agnostic).
