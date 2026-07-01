// tc_verify_gemm.cu — Marlin-class tensor-core W4A16 GEMM for the M<=16 verify forward (lever A).
// out[M,N] = dequant(W[N,K]) @ x[M,K]^T, FP4 weight x fp16 act, fp32 accumulate. Raw mma.sync.m16n8k16 with
// IN-REGISTER FP4->fp16 dequant of B (weight straight to fragment regs, no shared round-trip) -> tiny shared
// (only the shared A tile) -> high occupancy. Each warp owns a [16 M, 8 N] tile; A staged once per block.
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>

__device__ __forceinline__ float tcv_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
__device__ __forceinline__ __half2 tcv_fp4x2(unsigned char b){
    __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }

__device__ __forceinline__ void mma_m16n8k16(float* c, const unsigned* a, const unsigned* b){
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]), "r"(b[0]),"r"(b[1]));
}

#define WARPS 8   // 8 warps -> 64 N cols/block (8 N/warp)
__global__ void tc_w4a16_kernel(float* out, const uint8_t* wp, const uint8_t* ws, float wg_inv,
                                const __half* x16, int M, int N, int K){
    int warp = threadIdx.x>>5, lane = threadIdx.x&31, gid = lane>>2, t4 = lane&3;
    int n0 = blockIdx.x*64 + warp*8;
    const int KT=16;
    __shared__ __half As[16*KT];             // activation tile [16 M, KT] (block-shared)
    float c[4] = {0.f,0.f,0.f,0.f};
    int Kg = K/16, Kh = K/2;
    const uint8_t* wr = wp + (size_t)(n0+gid)*Kh;
    const uint8_t* sr = ws + (size_t)(n0+gid)*Kg;
    for(int k0=0; k0<K; k0+=KT){
        // stage A[16][KT]: 256 threads x (16*KT/256) each
        #pragma unroll
        for(int r=0;r<(16*KT)/256;++r){ int t=threadIdx.x+r*256, am=t/KT, ak=t%KT;
            As[t] = (am<M) ? x16[(size_t)am*K + k0+ak] : __float2half(0.f); }
        __syncthreads();
        #pragma unroll
        for(int ks=0; ks<KT; ks+=16){
            unsigned a[4];
            a[0]=*(const unsigned*)&As[gid*KT + ks + 2*t4];
            a[1]=*(const unsigned*)&As[(gid+8)*KT + ks + 2*t4];
            a[2]=*(const unsigned*)&As[gid*KT + ks + 2*t4+8];
            a[3]=*(const unsigned*)&As[(gid+8)*KT + ks + 2*t4+8];
            int kk=k0+ks;
            __half2 sc2 = __float2half2_rn(tcv_e4m3(sr[kk/16]) * wg_inv);
            __half2 b0 = __hmul2(tcv_fp4x2(wr[kk/2 + t4]),   sc2);
            __half2 b1 = __hmul2(tcv_fp4x2(wr[kk/2 + t4+4]), sc2);
            unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
            mma_m16n8k16(c, a, bb);
        }
        __syncthreads();
    }
    // store: c0=D[gid][2t4] c1=D[gid][2t4+1] c2=D[gid+8][2t4] c3=D[gid+8][2t4+1]; col n=n0+{2t4,2t4+1}
    int cn=2*t4;
    if(gid<M   && n0+cn  <N) out[(size_t)gid*N   + n0+cn  ]=c[0];
    if(gid<M   && n0+cn+1<N) out[(size_t)gid*N   + n0+cn+1]=c[1];
    if(gid+8<M && n0+cn  <N) out[(size_t)(gid+8)*N + n0+cn ]=c[2];
    if(gid+8<M && n0+cn+1<N) out[(size_t)(gid+8)*N + n0+cn+1]=c[3];
}
extern "C" void tc_w4a16_gemm(float* out, const uint8_t* wp, const uint8_t* ws, float w_gscale,
                              const void* x16, int M, int N, int K, cudaStream_t s){
    dim3 grid((N+63)/64); tc_w4a16_kernel<<<grid, WARPS*32, 0, s>>>(out, wp, ws, 1.f/w_gscale, (const __half*)x16, M, N, K);
}
