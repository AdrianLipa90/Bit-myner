#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include "sha256_common.h"

typedef __m512i V;
#define VSET(x) _mm512_set1_epi32((int)(x))
#define VZERO() _mm512_setzero_si512()
#define VADD(a,b) _mm512_add_epi32((a),(b))
#define VXOR(a,b) _mm512_xor_si512((a),(b))
#define VAND(a,b) _mm512_and_si512((a),(b))
#define VANDNOT(a,b) _mm512_andnot_si512((a),(b))
#define VOR(a,b) _mm512_or_si512((a),(b))
#define VROR(x,n) VOR(_mm512_srli_epi32((x),(n)),_mm512_slli_epi32((x),32-(n)))
static inline V vs0(V x){return VXOR(VXOR(VROR(x,7),VROR(x,18)),_mm512_srli_epi32(x,3));}
static inline V vs1(V x){return VXOR(VXOR(VROR(x,17),VROR(x,19)),_mm512_srli_epi32(x,10));}
static inline V vS0(V x){return VXOR(VXOR(VROR(x,2),VROR(x,13)),VROR(x,22));}
static inline V vS1(V x){return VXOR(VXOR(VROR(x,6),VROR(x,11)),VROR(x,25));}
static inline V vch(V e,V f,V g){return VXOR(VAND(e,f),VANDNOT(e,g));}
static inline V vmaj(V a,V b,V c){return VXOR(VXOR(VAND(a,b),VAND(a,c)),VAND(b,c));}
static inline void expand(V w[64]){for(int i=16;i<64;i++)w[i]=VADD(VADD(VADD(w[i-16],vs0(w[i-15])),w[i-7]),vs1(w[i-2]));}
static inline void rounds(V st[8], V w[64]){V a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];for(int i=0;i<64;i++){V t1=VADD(VADD(VADD(VADD(h,vS1(e)),vch(e,f,g)),VSET(BM_K256[i])),w[i]);V t2=VADD(vS0(a),vmaj(a,b,c));h=g;g=f;f=e;e=VADD(d,t1);d=c;c=b;b=a;a=VADD(t1,t2);}st[0]=VADD(st[0],a);st[1]=VADD(st[1],b);st[2]=VADD(st[2],c);st[3]=VADD(st[3],d);st[4]=VADD(st[4],e);st[5]=VADD(st[5],f);st[6]=VADD(st[6],g);st[7]=VADD(st[7],h);}
static inline void vcompress(V st[8],V w[64]){expand(w);rounds(st,w);}
static inline void hash16(const uint32_t mid[8],uint32_t t0,uint32_t t1,uint32_t t2,uint32_t base,V out[8]){V w[64];for(int i=0;i<64;i++)w[i]=VZERO();w[0]=VSET(t0);w[1]=VSET(t1);w[2]=VSET(t2);uint32_t n[16];for(int i=0;i<16;i++)n[i]=bm_bswap32(base+(uint32_t)i);w[3]=_mm512_loadu_si512((const void*)n);w[4]=VSET(0x80000000u);w[15]=VSET(640u);V s[8];for(int i=0;i<8;i++)s[i]=VSET(mid[i]);vcompress(s,w);for(int i=0;i<64;i++)w[i]=VZERO();for(int i=0;i<8;i++)w[i]=s[i];w[8]=VSET(0x80000000u);w[15]=VSET(256u);for(int i=0;i<8;i++)out[i]=VSET(BM_IV256[i]);vcompress(out,w);}
static inline uint64_t fold16(V x){uint32_t a[16];_mm512_storeu_si512((void*)a,x);uint64_t z=0;for(int i=0;i<16;i++)z^=((uint64_t)a[i]<<((i&1)*32));return z;}
uint64_t bm_avx512_scan_bench(const uint8_t *header76,uint32_t start,uint64_t count,int threads){uint32_t mid[8];bm_header_midstate(header76,mid);uint32_t t0=bm_be32(header76+64),t1=bm_be32(header76+68),t2=bm_be32(header76+72);uint64_t chunks=count/16,chk=0;
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1) reduction(^:chk)
for(uint64_t j=0;j<chunks;j++){V o[8];hash16(mid,t0,t1,t2,start+(uint32_t)(j*16),o);chk^=fold16(o[0]);chk^=fold16(o[7]);}return chk^count;}
int bm_avx512_scan_target(const uint8_t *header76,uint32_t start,uint64_t count,const uint8_t *target_le,int threads,uint32_t *found_nonce,uint8_t *found_hash){uint32_t mid[8];bm_header_midstate(header76,mid);uint32_t t0=bm_be32(header76+64),t1=bm_be32(header76+68),t2=bm_be32(header76+72);uint64_t chunks=count/16,best=UINT64_MAX;
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1)
for(uint64_t j=0;j<chunks;j++){if(j*16>=best)continue;V o[8];hash16(mid,t0,t1,t2,start+(uint32_t)(j*16),o);uint32_t words[8][16];for(int k=0;k<8;k++)_mm512_storeu_si512((void*)words[k],o[k]);for(int l=0;l<16;l++){uint8_t h[32];for(int k=0;k<8;k++)bm_putbe32(h+4*k,words[k][l]);if(bm_hash_le_target(h,target_le)){uint64_t off=j*16+(uint64_t)l;
#pragma omp critical(bitmyner_best_avx512)
{if(off<best)best=off;}}}}
if(best==UINT64_MAX)return 0;*found_nonce=start+(uint32_t)best;bm_sha256d_one(header76,*found_nonce,found_hash);return 1;}
