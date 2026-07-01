// cutlass_moe.cu — CUTLASS NVFP4 tensor-core GEMM wrapper for the hybrid server (Thor sm_110a).
// D[M,N] (bf16) = alpha * dequant(A[M,K] E2M1 x SFA e4m3) @ dequant(B[N,K] E2M1 x SFB e4m3)^T
// A row-major [M,K], B column-major i.e. weight [N,K] row-major (each output row's K contraction).
// Build: nvcc -std=c++17 --expt-relaxed-constexpr -I ~/cutlass/include -I ~/cutlass/tools/util/include
//        -gencode arch=compute_110a,code=sm_110a
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cstdio>
#include <cstdint>

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;   // FP4 codes + e4m3 block scale
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using LayoutATag = cutlass::layout::RowMajor;      // A[M,K] row-major
using LayoutBTag = cutlass::layout::ColumnMajor;   // B[N,K] -> col-major means weight rows contiguous in K
using LayoutCTag = cutlass::layout::RowMajor;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32, AlignmentB = 32;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
using ElementAccumulator = float;
using ArchTag       = cutlass::arch::Sm100;                     // sm_100 tag; SASS built for sm_110a via -gencode
using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
using MmaTileShape  = Shape<_128,_128,_256>;                    // 1SM tile — best for small verify M
using ClusterShape  = Shape<_1,_1,_1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass, MmaTileShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator, MmaTileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int,int,int,int>, CollectiveMainloop, CollectiveEpilogue, void>;
using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
using StrideA = typename Gemm::GemmKernel::StrideA;
using StrideB = typename Gemm::GemmKernel::StrideB;
using StrideC = typename Gemm::GemmKernel::StrideC;
using StrideD = typename Gemm::GemmKernel::StrideD;
using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ---- callable API for the server ----
// element counts of the swizzled scale-factor buffers the caller must provide (e4m3 bytes each)
extern "C" long cutlass_sfa_bytes(int M,int N,int K){ auto l=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1)); return (long)cute::cosize(l); }
extern "C" long cutlass_sfb_bytes(int M,int N,int K){ auto l=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1)); return (long)cute::cosize(l); }
extern "C" long cutlass_workspace_bytes(int M,int N,int K){
    typename Gemm::Arguments a{ cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
        { nullptr, StrideA{}, nullptr, StrideB{}, nullptr, {}, nullptr, {} },
        { {1.f,0.f}, nullptr, StrideC{}, nullptr, StrideD{} } };
    return (long)Gemm::get_workspace_size(a);
}

// D = alpha * A@B^T.  A_fp4/B_fp4 = packed E2M1 (2 codes/byte). SFA/SFB = swizzled e4m3. workspace pre-alloc'd.
extern "C" int cutlass_nvfp4_gemm(void* D, const void* A_fp4, const void* B_fp4,
                                  const void* SFA, const void* SFB, float alpha,
                                  int M,int N,int K, void* workspace, cudaStream_t stream){
    StrideA sA = cutlass::make_cute_packed_stride(StrideA{}, {M,K,1});
    StrideB sB = cutlass::make_cute_packed_stride(StrideB{}, {N,K,1});
    StrideC sC = cutlass::make_cute_packed_stride(StrideC{}, {M,N,1});
    StrideD sD = cutlass::make_cute_packed_stride(StrideD{}, {M,N,1});
    auto lSFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1));
    auto lSFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1));
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm, {M,N,K,1},
        { (const cutlass::float_e2m1_t*)A_fp4, sA, (const cutlass::float_e2m1_t*)B_fp4, sB,
          (const cutlass::float_ue4m3_t*)SFA, lSFA, (const cutlass::float_ue4m3_t*)SFB, lSFB },
        { {alpha, 0.f}, (const ElementC*)nullptr, sC, (ElementD*)D, sD } };
    Gemm gemm;
    if(gemm.can_implement(args) != cutlass::Status::kSuccess) return 1;
    if(gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) return 2;
    if(gemm.run(stream) != cutlass::Status::kSuccess) return 3;
    return 0;
}

#ifdef CUTLASS_MOE_TEST
#include <cuda_runtime.h>
__global__ void fill(uint8_t* p,long n,int v){ long i=blockIdx.x*256L+threadIdx.x; if(i<n)p[i]=(uint8_t)((i*131+v)&0xff); }
__global__ void bf16_to_f(const uint16_t* b,float* f,int n){int i=blockIdx.x*256+threadIdx.x; if(i<n){unsigned u=(unsigned)b[i]<<16; float x; memcpy(&x,&u,4); f[i]=x;}}
int main(int argc,char**argv){
    int M=16,N=2816,K=2816;
    // KNOWN input: every value = code2(=1.0) * scale. code byte 0x22 = two 1.0 codes. sweep scale byte via argv[1].
    int scb = argc>1?atoi(argv[1]):0x38;   // e4m3 1.0 = 0x38 (guess); D should = K=2816 when scale=1.0
    {
    long aB=(long)M*K/2, bB=(long)N*K/2, sfaB=cutlass_sfa_bytes(M,N,K), sfbB=cutlass_sfb_bytes(M,N,K), wsB=cutlass_workspace_bytes(M,N,K);
    void *A,*B,*SFA,*SFB,*D,*WS; float* Df;
    cudaMalloc(&A,aB);cudaMalloc(&B,bB);cudaMalloc(&SFA,sfaB);cudaMalloc(&SFB,sfbB);cudaMalloc(&D,(long)M*N*2);cudaMalloc(&WS,wsB>0?wsB:16);cudaMalloc(&Df,(long)M*N*4);
    cudaMemset(A,0x22,aB);cudaMemset(B,0x22,bB);cudaMemset(SFA,scb,sfaB);cudaMemset(SFB,scb,sfbB);cudaDeviceSynchronize();
    int rc=cutlass_nvfp4_gemm(D,A,B,SFA,SFB,1.0f,M,N,K,WS,0); cudaError_t e=cudaDeviceSynchronize();
    bf16_to_f<<<(M*N+255)/256,256>>>((uint16_t*)D,Df,M*N); cudaDeviceSynchronize();
    float h[4]; cudaMemcpy(h,Df,16,cudaMemcpyDeviceToHost);
    printf("scale_byte=0x%02x rc=%d cuda=%s  D[0..3]= %.1f %.1f %.1f %.1f  (expect %d if scale=1.0)\n",scb,rc,cudaGetErrorString(e),h[0],h[1],h[2],h[3],K);
    return 0; }
    int M2=16,N2=2816,K2=2816; (void)M2;(void)N2;(void)K2;
    long aB=(long)M*K/2, bB=(long)N*K/2, sfaB=cutlass_sfa_bytes(M,N,K), sfbB=cutlass_sfb_bytes(M,N,K), wsB=cutlass_workspace_bytes(M,N,K);
    printf("sizes: A_fp4=%ld B_fp4=%ld SFA=%ld SFB=%ld ws=%ld bytes\n",aB,bB,sfaB,sfbB,wsB);
    void *A,*B,*SFA,*SFB,*D,*WS;
    cudaMalloc(&A,aB);cudaMalloc(&B,bB);cudaMalloc(&SFA,sfaB);cudaMalloc(&SFB,sfbB);cudaMalloc(&D,(long)M*N*2);cudaMalloc(&WS,wsB>0?wsB:16);
    fill<<<(aB+255)/256,256>>>((uint8_t*)A,aB,1); fill<<<(bB+255)/256,256>>>((uint8_t*)B,bB,7);
    fill<<<(sfaB+255)/256,256>>>((uint8_t*)SFA,sfaB,60); fill<<<(sfbB+255)/256,256>>>((uint8_t*)SFB,sfbB,60);
    cudaDeviceSynchronize();
    int rc=cutlass_nvfp4_gemm(D,A,B,SFA,SFB,1.0f,M,N,K,WS,0);
    cudaError_t e=cudaDeviceSynchronize();
    printf("cutlass_nvfp4_gemm rc=%d cuda=%s\n",rc,cudaGetErrorString(e));
    // check output finite
    uint16_t h[8]; cudaMemcpy(h,D,16,cudaMemcpyDeviceToHost);
    printf("D[0..3] raw bf16: %04x %04x %04x %04x\n",h[0],h[1],h[2],h[3]);
    printf(rc==0 && e==cudaSuccess ? "SMOKE TEST: WRAPPER RUNS OK\n":"SMOKE TEST: FAILED\n");
    return rc;
}
#endif
