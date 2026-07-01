// tc_verify_gemm.cu — tensor-core W4A16 GEMM for the M<=16 verify forward (lever A).
// out[M,N] = dequant(W[N,K]) @ x[M,K]^T, FP4 weight x fp16 activation, fp32 accumulate (accuracy preserved).
// Uses wmma 16x16x16: M=15 pads to 16 (minimal waste, vs the 128-tile grouped path). FP4 dequant->fp16 into
// shared per K-16 group (one e4m3 scale/group), then MMA. Each warp owns one 16-col N-tile; A-tile shared across warps.
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

#define WARPS 8   // 8 warps/block -> 8 N-tiles (128 N) per block
// block.x = ceil(N/128). blockDim = WARPS*32 = 256.
__global__ void tc_w4a16_kernel(float* out, const uint8_t* wp, const uint8_t* ws, float wg_inv,
                                const __half* x16, int M, int N, int K){
    int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int n0 = blockIdx.x*128 + warp*16;           // this warp's N-tile start
    const int KT=64;                             // K-tile: 4 MMAs per shared-load -> 4x fewer syncs
    __shared__ __half As[16*KT];                 // [16 M, KT] shared by all warps
    __shared__ __half Bs[WARPS][16*KT];          // per-warp [16 N, KT] col-major (Bs[n*KT+k]=B[k,n])
    wmma::fragment<wmma::accumulator,16,16,16,float> cf; wmma::fill_fragment(cf,0.f);
    int Kg = K/16;
    const uint8_t* wpb = wp + (size_t)n0*(K/2);
    const uint8_t* wsb = ws + (size_t)n0*(K/16);
    for(int k0=0; k0<K; k0+=KT){
        int idx=threadIdx.x;                     // A: 256 threads x 4 = 16*64 elements
        #pragma unroll
        for(int r=0;r<4;++r){ int t=idx+r*256; int am=t/KT, ak=t%KT;
            As[t] = (am<M) ? x16[(size_t)am*K + k0+ak] : __float2half(0.f); }
        #pragma unroll                           // B: 32 lanes x 32 = 16*64 elements/warp (dequant)
        for(int e=0;e<32;++e){ int t=lane*32+e; int bn=t/KT, bk=t%KT;
            float sc = tcv_e4m3(wsb[(size_t)bn*Kg + (k0+bk)/16]) * wg_inv;
            uint8_t byte = wpb[(size_t)bn*(K/2) + (k0+bk)/2];
            uint8_t nib = ((k0+bk)&1) ? (byte>>4) : (byte&0xF);
            Bs[warp][bn*KT+bk] = __hmul(__low2half(tcv_fp4x2(nib)), __float2half(sc)); }
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
    // store [16 M, 16 N] tile -> out[m, n0+n], only m<M, n0+n<N
    __shared__ float Cs[WARPS][16*16];
    wmma::store_matrix_sync(Cs[warp], cf, 16, wmma::mem_row_major);
    for(int e=lane; e<16*16; e+=32){ int cm=e>>4, cn=e&15; int gn=n0+cn;
        if(cm<M && gn<N) out[(size_t)cm*N + gn] = Cs[warp][e]; }
}
extern "C" void tc_w4a16_gemm(float* out, const uint8_t* wp, const uint8_t* ws, float w_gscale,
                              const void* x16, int M, int N, int K, cudaStream_t s){
    dim3 grid((N+127)/128); tc_w4a16_kernel<<<grid, WARPS*32, 0, s>>>(out, wp, ws, 1.f/w_gscale, (const __half*)x16, M, N, K);
}
