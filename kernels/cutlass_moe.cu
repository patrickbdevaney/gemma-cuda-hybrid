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

// ---- scale swizzle: our LINEAR e4m3 scales [rows][K/16] -> CUTLASS blocked SFA/SFB layout ----
template<class LSF>
__global__ void k_swizzle(uint8_t* out, const uint8_t* in_lin, LSF lay, int rows, int Kg){
    long idx=(long)blockIdx.x*256+threadIdx.x; if(idx>=(long)rows*Kg)return;
    int r=idx/Kg, g=idx%Kg;
    long off = lay(r, g*16, 0);          // 3-arg cute layout: (row, k-element, L) -> swizzled offset
    out[off] = in_lin[(long)r*Kg+g];
}
extern "C" void cutlass_swizzle_sfa(uint8_t* out,const uint8_t* in,int M,int N,int K,cudaStream_t s){
    auto lay=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M,N,K,1)); int Kg=K/16;
    k_swizzle<<<(long)(M*Kg+255)/256,256,0,s>>>(out,in,lay,M,Kg);
}
extern "C" void cutlass_swizzle_sfb(uint8_t* out,const uint8_t* in,int M,int N,int K,cudaStream_t s){
    auto lay=Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M,N,K,1)); int Kg=K/16;
    k_swizzle<<<(long)(N*Kg+255)/256,256,0,s>>>(out,in,lay,N,Kg);
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
#include <cuda_fp8.h>
__device__ float e2m1d(int c){ const float v[8]={0,.5f,1,1.5f,2,3,4,6}; float m=v[c&7]; return (c&8)?-m:m; }
__device__ float e4m3d(uint8_t b){ __nv_fp8_e4m3 x; *(uint8_t*)&x=b; return (float)x; }
// reference: D[i][j] = alpha * sum_k e2m1(A code)*e4m3(As grp) * e2m1(B code)*e4m3(Bs grp)
__global__ void refgemm(float* D,const uint8_t* A,const uint8_t* B,const uint8_t* As,const uint8_t* Bs,int M,int N,int K,float alpha){
    int i=blockIdx.x, j=blockIdx.y*256+threadIdx.x; if(i>=M||j>=N)return; int Kg=K/16; float acc=0;
    for(int k=0;k<K;k++){ int ac=(A[(long)i*K/2+k/2]>>((k&1)*4))&0xf; int bc=(B[(long)j*K/2+k/2]>>((k&1)*4))&0xf;
        float av=e2m1d(ac)*e4m3d(As[(long)i*Kg+k/16]); float bv=e2m1d(bc)*e4m3d(Bs[(long)j*Kg+k/16]); acc+=av*bv; }
    D[(long)i*N+j]=alpha*acc;
}
__global__ void bf2ff(const uint16_t* b,float* f,int n){int i=blockIdx.x*256+threadIdx.x; if(i<n){unsigned u=(unsigned)b[i]<<16;float x;memcpy(&x,&u,4);f[i]=x;}}
__global__ void frand(uint8_t* p,long n,int s){long i=(long)blockIdx.x*256+threadIdx.x; if(i<n)p[i]=(uint8_t)((i*2654435761u+s*40503u)>>13);}
__global__ void srand8(uint8_t* p,long n,int s){long i=(long)blockIdx.x*256+threadIdx.x; if(i<n)p[i]=(uint8_t)(0x34+(((i*97+s)>>2)&7));} // e4m3 ~0.75..1.5
int validate_random(int M,int N,int K){
    long aB=(long)M*K/2,bB=(long)N*K/2,asB=(long)M*K/16,bsB=(long)N*K/16;
    long sfaB=cutlass_sfa_bytes(M,N,K),sfbB=cutlass_sfb_bytes(M,N,K),wsB=cutlass_workspace_bytes(M,N,K);
    uint8_t *A,*B,*As,*Bs,*SFA,*SFB; void *D,*WS; float *Dw,*Dr;
    cudaMalloc(&A,aB);cudaMalloc(&B,bB);cudaMalloc(&As,asB);cudaMalloc(&Bs,bsB);
    cudaMalloc(&SFA,sfaB);cudaMalloc(&SFB,sfbB);cudaMalloc(&D,(long)M*N*2);cudaMalloc(&WS,wsB>0?wsB:16);
    cudaMalloc(&Dw,(long)M*N*4);cudaMalloc(&Dr,(long)M*N*4);
    frand<<<(aB+255)/256,256>>>(A,aB,1);frand<<<(bB+255)/256,256>>>(B,bB,2);
    srand8<<<(asB+255)/256,256>>>(As,asB,3);srand8<<<(bsB+255)/256,256>>>(Bs,bsB,4);
    cudaMemset(SFA,0,sfaB);cudaMemset(SFB,0,sfbB);cudaDeviceSynchronize();
    cutlass_swizzle_sfa(SFA,As,M,N,K,0); cutlass_swizzle_sfb(SFB,Bs,M,N,K,0); cudaDeviceSynchronize();
    int rc=cutlass_nvfp4_gemm(D,A,B,SFA,SFB,1.0f,M,N,K,WS,0);
    bf2ff<<<(M*N+255)/256,256>>>((uint16_t*)D,Dw,M*N);
    dim3 gb(M,(N+255)/256); refgemm<<<gb,256>>>(Dr,A,B,As,Bs,M,N,K,1.0f); cudaDeviceSynchronize();
    float *hw=(float*)malloc(M*N*4),*hr=(float*)malloc(M*N*4);
    cudaMemcpy(hw,Dw,(long)M*N*4,cudaMemcpyDeviceToHost);cudaMemcpy(hr,Dr,(long)M*N*4,cudaMemcpyDeviceToHost);
    float maxrel=0; int bad=0;
    for(long t=0;t<(long)M*N;t++){ float d=fabsf(hw[t]-hr[t]),r=d/(fabsf(hr[t])+1e-3f); if(r>maxrel)maxrel=r; if(r>0.05f)bad++; }
    printf("VALIDATE M=%d N=%d K=%d: rc=%d  wrapper[0]=%.2f ref[0]=%.2f  maxrel=%.4f bad(>5%%)=%d/%ld -> %s\n",
           M,N,K,rc,hw[0],hr[0],maxrel,bad,(long)M*N, (rc==0&&maxrel<0.06f)?"PASS ✅":"FAIL ❌");
    free(hw);free(hr); return (rc==0&&maxrel<0.06f)?0:1;
}

__global__ void fill(uint8_t* p,long n,int v){ long i=blockIdx.x*256L+threadIdx.x; if(i<n)p[i]=(uint8_t)((i*131+v)&0xff); }
__global__ void bf16_to_f(const uint16_t* b,float* f,int n){int i=blockIdx.x*256+threadIdx.x; if(i<n){unsigned u=(unsigned)b[i]<<16; float x; memcpy(&x,&u,4); f[i]=x;}}
int main(int,char**){
    printf("=== end-to-end validation (random FP4 + swizzle vs fp32 reference) ===\n");
    int r=0;
    r|=validate_random(16,2816,704);    // down shape (per verify-MoE)
    r|=validate_random(16,704,2816);    // gateup shape
    r|=validate_random(128,2816,2816);  // larger square
    printf(r==0? "\nALL VALIDATIONS PASS -- swizzle + wrapper correct.\n":"\nSOME VALIDATION FAILED.\n");
    return r;
}
#endif
