// tc_verify_gemm.cu — tensor-core W4A16 GEMM for the M<=16 verify forward (lever A, Marlin-class build).
// out[M,N] = dequant(W[N,K]) @ x[M,K]^T, FP4 weight x fp16 activation, fp32 accumulate (accuracy preserved).
// wmma 16x16x16: M pads to 16 (minimal waste). Coalesced uint weight loads (8 codes/load), batched FP4->fp16
// dequant into shared per K-16 group. Each warp owns one 16-col N-tile; A-tile shared across warps.
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <mma.h>
#include <cstdint>
using namespace nvcuda;

__device__ __forceinline__ float tcv_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }
__device__ __forceinline__ __half2 tcv_fp4x2(unsigned char b){
    __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }

#define WARPS 8
#define KT 32                                  // K-tile: 2 MMAs/load, moderate shared (occupancy-friendly)
__global__ void tc_w4a16_kernel(float* out, const uint8_t* wp, const uint8_t* ws, float wg_inv,
                                const __half* x16, int M, int N, int K){
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n0 = blockIdx.x*128 + warp*16;
    __shared__ __half As[16*KT];               // [16 M, KT] shared by all warps
    __shared__ __half Bs[WARPS][16*KT];        // per-warp [16 N, KT] col-major (Bs[n*KT+k]=B[k,n])
    wmma::fragment<wmma::accumulator,16,16,16,float> cf; wmma::fill_fragment(cf,0.f);
    int Kg=K/16, upr=KT/8;                      // uints per row for the B-tile
    const uint8_t* wpb = wp + (size_t)n0*(K/2);
    const uint8_t* wsb = ws + (size_t)n0*(K/16);
    for(int k0=0; k0<K; k0+=KT){
        // A: 16*KT halves, coalesced. 256 threads x (16*KT/256) each.
        #pragma unroll
        for(int r=0; r<(16*KT)/256; ++r){ int t=threadIdx.x + r*256; int am=t/KT, ak=t%KT;
            As[t] = (am<M) ? x16[(size_t)am*K + k0+ak] : __float2half(0.f); }
        // B: coalesced uint loads. 16*upr uints, 32 lanes x (16*upr/32) each; 1 uint -> 8 codes.
        #pragma unroll
        for(int j=0; j<(16*upr)/32; ++j){ int u=lane + j*32; int bn=u/upr, kc=(u%upr)*8;
            unsigned w=*(const unsigned*)(wpb + (size_t)bn*(K/2) + (k0+kc)/2);
            const unsigned char* ub=(const unsigned char*)&w;
            __half sch=__float2half(tcv_e4m3(wsb[(size_t)bn*Kg + (k0+kc)/16]) * wg_inv);
            #pragma unroll
            for(int b=0;b<4;++b){ __half2 d=tcv_fp4x2(ub[b]);
                Bs[warp][bn*KT + kc+2*b]   = __hmul(__low2half(d), sch);
                Bs[warp][bn*KT + kc+2*b+1] = __hmul(__high2half(d), sch); } }
        __syncthreads();
        #pragma unroll
        for(int ks=0; ks<KT; ks+=16){
            wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
            wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::col_major> bf;
            wmma::load_matrix_sync(af, As + ks, KT);
            wmma::load_matrix_sync(bf, Bs[warp] + ks, KT);
            wmma::mma_sync(cf, af, bf, cf);
        }
        __syncthreads();
    }
    __shared__ float Cs[WARPS][16*16];
    wmma::store_matrix_sync(Cs[warp], cf, 16, wmma::mem_row_major);
    for(int e=lane; e<16*16; e+=32){ int cm=e>>4, cn=e&15; int gn=n0+cn;
        if(cm<M && gn<N) out[(size_t)cm*N + gn] = Cs[warp][e]; }
}
extern "C" void tc_w4a16_gemm(float* out, const uint8_t* wp, const uint8_t* ws, float w_gscale,
                              const void* x16, int M, int N, int K, cudaStream_t s){
    dim3 grid((N+127)/128); tc_w4a16_kernel<<<grid, WARPS*32, 0, s>>>(out, wp, ws, 1.f/w_gscale, (const __half*)x16, M, N, K);
}
