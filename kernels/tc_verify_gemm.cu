// tc_verify_gemm.cu — Marlin-class TC W4A16 verify GEMM (lever A) with OFFLINE WEIGHT REPACK.
// out[M,N] = dequant(W[N,K]) @ x[M,K]^T, FP4 weight x fp16 act, fp32 accum. Raw mma.sync.m16n8k16, 1 warp = 8 N-cols
// (max grid fill), A fragment direct from L2 global x (no shared/sync), B in-register FP4->fp16 dequant.
// REPACK: weight/scale rearranged once (cached by src ptr) into [n_block][k_tile][lane] order so a warp's per-k-tile
// reads are 64 CONTIGUOUS bytes (coalesced) instead of 8 strided row-reads -> pushes the memory-bound kernel higher.
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdio>
#include <unordered_map>

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
// ---- repack: wp[N][K/2] -> wpr[N/8][K/16][64] (lane order [blo,bhi], lane=gid*4+t4); ws[N][K/16] -> wsr[N/8][K/16][8]
__global__ void k_tc_repack_w(uint8_t* wpr, const uint8_t* wp, int N, int K){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; int kt=K/16; long tot=(long)(N/8)*kt*32; if(idx>=tot)return;
    int lane=idx&31; long r=idx>>5; int k_tile=r%kt, n_block=r/kt, gid=lane>>2, t4=lane&3;
    long src=(long)(n_block*8+gid)*(K/2) + (long)k_tile*8;
    long dst=((long)n_block*kt + k_tile)*64 + lane*2;
    wpr[dst]=wp[src+t4]; wpr[dst+1]=wp[src+t4+4];
}
__global__ void k_tc_repack_s(uint8_t* wsr, const uint8_t* ws, int N, int K){
    long idx=(long)blockIdx.x*blockDim.x+threadIdx.x; int kt=K/16; long tot=(long)(N/8)*kt*8; if(idx>=tot)return;
    int g=idx&7; long r=idx>>3; int k_tile=r%kt, n_block=r/kt;
    wsr[((long)n_block*kt + k_tile)*8 + g] = ws[(long)(n_block*8+g)*(K/16) + k_tile];
}
// ---- GEMM over repacked weight (coalesced 64B/warp/k-tile) ----
#ifndef WARPS
#define WARPS 1
#endif
__global__ void tc_w4a16_kernel(float* out, const uint8_t* wpr, const uint8_t* wsr, float wg_inv,
                                const __half* x16, int M, int N, int K){
    int lane=threadIdx.x&31, gid=lane>>2, t4=lane&3;
    int warp=threadIdx.x>>5; int n_block=blockIdx.x*WARPS+warp; if((long)n_block*8>=N) return; int n0=n_block*8;
    float c[4]={0.f,0.f,0.f,0.f}; int kt=K/16;
    const uint8_t* wb = wpr + (long)n_block*kt*64;
    const uint8_t* sb = wsr + (long)n_block*kt*8;
    const __half* xg0 = x16 + (size_t)gid*K;
    const __half* xg8 = x16 + (size_t)(gid+8)*K;
    bool m0=gid<M, m8=(gid+8)<M;
    for(int k_tile=0; k_tile<kt; ++k_tile){
        int k0=k_tile*16;
        unsigned a[4];
        a[0]=m0? *(const unsigned*)(xg0+k0+2*t4)   : 0u;
        a[1]=m8? *(const unsigned*)(xg8+k0+2*t4)   : 0u;
        a[2]=m0? *(const unsigned*)(xg0+k0+2*t4+8) : 0u;
        a[3]=m8? *(const unsigned*)(xg8+k0+2*t4+8) : 0u;
        unsigned short w2 = *(const unsigned short*)(wb + (long)k_tile*64 + lane*2);   // coalesced
        __half2 sc2 = __float2half2_rn(tcv_e4m3(sb[(long)k_tile*8 + gid]) * wg_inv);
        __half2 b0 = __hmul2(tcv_fp4x2((uint8_t)(w2&0xFF)), sc2);
        __half2 b1 = __hmul2(tcv_fp4x2((uint8_t)(w2>>8)),   sc2);
        unsigned bb[2]; bb[0]=*(unsigned*)&b0; bb[1]=*(unsigned*)&b1;
        mma_m16n8k16(c, a, bb);
    }
    int cn=2*t4;
    if(gid<M   && n0+cn  <N) out[(size_t)gid*N   + n0+cn  ]=c[0];
    if(gid<M   && n0+cn+1<N) out[(size_t)gid*N   + n0+cn+1]=c[1];
    if(gid+8<M && n0+cn  <N) out[(size_t)(gid+8)*N + n0+cn ]=c[2];
    if(gid+8<M && n0+cn+1<N) out[(size_t)(gid+8)*N + n0+cn+1]=c[3];
}
static std::unordered_map<const void*, std::pair<uint8_t*,uint8_t*>> g_tc_cache;   // repacked weights by src ptr
extern "C" void tc_w4a16_gemm(float* out, const uint8_t* wp, const uint8_t* ws, float w_gscale,
                              const void* x16, int M, int N, int K, cudaStream_t s){
    auto it=g_tc_cache.find((const void*)wp); uint8_t *wpr,*wsr;
    if(it==g_tc_cache.end()){                                    // repack once (first call = warm-up, pre-graph-capture)
        int kt=K/16; long nb=N/8;
        cudaMalloc(&wpr,(size_t)nb*kt*64); cudaMalloc(&wsr,(size_t)nb*kt*8);
        k_tc_repack_w<<<(unsigned)(((long)nb*kt*32+255)/256),256>>>(wpr,wp,N,K);
        k_tc_repack_s<<<(unsigned)(((long)nb*kt*8+255)/256),256>>>(wsr,ws,N,K);
        cudaDeviceSynchronize(); g_tc_cache[(const void*)wp]={wpr,wsr};
    } else { wpr=it->second.first; wsr=it->second.second; }
    tc_w4a16_kernel<<<(unsigned)((N/8+WARPS-1)/WARPS), WARPS*32, 0, s>>>(out, wpr, wsr, 1.f/w_gscale, (const __half*)x16, M, N, K);
}
