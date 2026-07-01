// megakernel.cu — persistent counter-synced fused kernels (milestone M1).
// The champion per-kernel path stays default; this runs only under MEGAKERNEL=1.
// M1 goal: prove the persistent-grid + global-counter-sync pattern by fusing
// input_rmsnorm + Q/K/V projection into ONE kernel, bit-close to the current path.
#include "megakernel.h"
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <cstdio>
#include <cstring>

#define MCU(x) do{cudaError_t e=(x); if(e){fprintf(stderr,"mega %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));}}while(0)

__device__ __forceinline__ float mbf2f(uint16_t h){ unsigned u=(unsigned)h<<16; float f; memcpy(&f,&u,4); return f; }
__device__ __forceinline__ __half2 mdec_fp4x2(unsigned char b){
    __half2_raw r=__nv_cvt_fp4x2_to_halfraw2((__nv_fp4x2_storage_t)b,__NV_E2M1); return *reinterpret_cast<__half2*>(&r); }
__device__ __forceinline__ float mhw_e4m3(uint8_t b){
    __half_raw r=__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)b,__NV_E4M3); return __half2float(*reinterpret_cast<__half*>(&r)); }

// warp GEMV dot: y[n] = sum_k dec(Wp[n][k]) * e4m3(Ws[n][k/16]) * wg_inv * x16[k]. Result valid on lane 0.
__device__ __forceinline__ float mega_dot(const uint8_t* Wp,const uint8_t* Ws,float wg_inv,const __half* x16,int n,int K,int lane){
    const unsigned* wpn=(const unsigned*)(Wp+(size_t)n*(K/2)); const uint8_t* wsn=Ws+(size_t)n*(K/16);
    float acc=0.f; int nu=K/8;
    for(int vi=lane; vi<nu; vi+=32){
        unsigned w=__ldcs(&wpn[vi]); int k=vi*8; float sc=mhw_e4m3(__ldcs(&wsn[k>>4]))*wg_inv;
        uint4 xpk=*(const uint4*)(x16+k); const __half2* xh2=(const __half2*)&xpk; const unsigned char* wb=(const unsigned char*)&w;
        __half2 a2=__float2half2_rn(0.f);
        #pragma unroll
        for(int b=0;b<4;++b) a2=__hfma2(mdec_fp4x2(wb[b]),xh2[b],a2);
        acc += sc*(__half2float(__low2half(a2))+__half2float(__high2half(a2)));
    }
    #pragma unroll
    for(int o=16;o>0;o>>=1) acc+=__shfl_down_sync(0xffffffffu,acc,o);
    return acc;
}

// Persistent M1 kernel. Phase A (block 0): RMSNorm h -> hnorm16, signal counter. All blocks spin, then Phase B: GEMV.
__global__ void mega_qkv_kernel(float* q,float* k,float* v,
    const float* h,const uint16_t* g,__half* hnorm16,int* counter,
    const uint8_t* qp,const uint8_t* qs,float qwi,
    const uint8_t* kp,const uint8_t* ks,float kwi,
    const uint8_t* vp,const uint8_t* vs,float vwi,
    int H,int qd,int kd,float eps){
    int tid=threadIdx.x, lane=tid&31, warp=tid>>5, wpb=blockDim.x>>5;
    if(blockIdx.x==0){                                   // Phase A: produce hnorm16 (fp16), FlashNorm-style not yet (kept simple/correct)
        __shared__ float ss[256]; float loc=0.f;
        for(int i=tid;i<H;i+=blockDim.x){ float x=h[i]; loc+=x*x; }
        ss[tid]=loc; __syncthreads();
        for(int s=blockDim.x/2;s>0;s>>=1){ if(tid<s)ss[tid]+=ss[tid+s]; __syncthreads(); }
        float rinv=rsqrtf(ss[0]/H+eps);
        for(int i=tid;i<H;i+=blockDim.x) hnorm16[i]=__float2half(h[i]*rinv*mbf2f(g[i]));
        __threadfence();
        if(tid==0) atomicExch(counter,1);
    }
    if(tid==0){ while(*(volatile int*)counter==0){} }    // sentinel spin (all blocks resident -> no deadlock)
    __syncthreads(); __threadfence();
    int nv_out = vp? kd : 0;                              // full layer: vp null, v=k copied by host
    long total = (long)qd + kd + nv_out;
    for(long o=(long)blockIdx.x*wpb+warp; o<total; o+=(long)gridDim.x*wpb){
        if(o<qd){ float r=mega_dot(qp,qs,qwi,hnorm16,(int)o,H,lane); if(lane==0)q[o]=r; }
        else if(o<qd+kd){ int n=o-qd; float r=mega_dot(kp,ks,kwi,hnorm16,n,H,lane); if(lane==0)k[n]=r; }
        else { int n=o-qd-kd; float r=mega_dot(vp,vs,vwi,hnorm16,n,H,lane); if(lane==0)v[n]=r; }
    }
}

extern "C" void mega_qkv(float* q,float* k,float* v,const float* h,const uint16_t* g,
    const uint8_t* qp,const uint8_t* qs,float qwg,
    const uint8_t* kp,const uint8_t* ks,float kwg,
    const uint8_t* vp,const uint8_t* vs,float vwg,
    int H,int qd,int kd,float eps,cudaStream_t s){
    static __half* hnorm16=nullptr; static int* counter=nullptr;
    if(!hnorm16){ MCU(cudaMalloc(&hnorm16,(size_t)16384*sizeof(__half))); MCU(cudaMalloc(&counter,sizeof(int))); }
    MCU(cudaMemsetAsync(counter,0,sizeof(int),s));
    mega_qkv_kernel<<<64,256,0,s>>>(q,k,v,h,g,hnorm16,counter,
        qp,qs,1.f/qwg, kp,ks,1.f/kwg, vp,vp?vs:nullptr,vp?1.f/vwg:0.f, H,qd,kd,eps);
    if(!vp) MCU(cudaMemcpyAsync(v,k,(size_t)kd*sizeof(float),cudaMemcpyDeviceToDevice,s));  // full layer: v=k
}
