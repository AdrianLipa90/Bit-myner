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
static inline V s0(V x){return VXOR(VXOR(VROR(x,7),VROR(x,18)),_mm512_srli_epi32(x,3));}
static inline V s1(V x){return VXOR(VXOR(VROR(x,17),VROR(x,19)),_mm512_srli_epi32(x,10));}
static inline V S0(V x){return VXOR(VXOR(VROR(x,2),VROR(x,13)),VROR(x,22));}
static inline V S1(V x){return VXOR(VXOR(VROR(x,6),VROR(x,11)),VROR(x,25));}
static inline V Ch(V e,V f,V g){return VXOR(VAND(e,f),VANDNOT(e,g));}
static inline V Maj(V a,V b,V c){return VXOR(VXOR(VAND(a,b),VAND(a,c)),VAND(b,c));}
static inline void compress2_roll(V sa[8],V wa[16],V sb[8],V wb[16]){
 V aa=sa[0],ba=sa[1],ca=sa[2],da=sa[3],ea=sa[4],fa=sa[5],ga=sa[6],ha=sa[7];
 V ab=sb[0],bb=sb[1],cb=sb[2],db=sb[3],eb=sb[4],fb=sb[5],gb=sb[6],hb=sb[7];
 for(int i=0;i<64;i++){
  int j=i&15;
  if(i>=16){wa[j]=VADD(VADD(VADD(wa[j],s0(wa[(j+1)&15])),wa[(j+9)&15]),s1(wa[(j+14)&15]));wb[j]=VADD(VADD(VADD(wb[j],s0(wb[(j+1)&15])),wb[(j+9)&15]),s1(wb[(j+14)&15]));}
  V t1a=VADD(VADD(VADD(VADD(ha,S1(ea)),Ch(ea,fa,ga)),VSET(BM_K256[i])),wa[j]);V t2a=VADD(S0(aa),Maj(aa,ba,ca));
  V t1b=VADD(VADD(VADD(VADD(hb,S1(eb)),Ch(eb,fb,gb)),VSET(BM_K256[i])),wb[j]);V t2b=VADD(S0(ab),Maj(ab,bb,cb));
  ha=ga;ga=fa;fa=ea;ea=VADD(da,t1a);da=ca;ca=ba;ba=aa;aa=VADD(t1a,t2a);
  hb=gb;gb=fb;fb=eb;eb=VADD(db,t1b);db=cb;cb=bb;bb=ab;ab=VADD(t1b,t2b);
 }
 sa[0]=VADD(sa[0],aa);sa[1]=VADD(sa[1],ba);sa[2]=VADD(sa[2],ca);sa[3]=VADD(sa[3],da);sa[4]=VADD(sa[4],ea);sa[5]=VADD(sa[5],fa);sa[6]=VADD(sa[6],ga);sa[7]=VADD(sa[7],ha);
 sb[0]=VADD(sb[0],ab);sb[1]=VADD(sb[1],bb);sb[2]=VADD(sb[2],cb);sb[3]=VADD(sb[3],db);sb[4]=VADD(sb[4],eb);sb[5]=VADD(sb[5],fb);sb[6]=VADD(sb[6],gb);sb[7]=VADD(sb[7],hb);
}
static inline void hash32(const uint32_t mid[8],uint32_t t0,uint32_t t1,uint32_t t2,uint32_t base,V oa[8],V ob[8]){
 V wa[16],wb[16];for(int i=0;i<16;i++)wa[i]=wb[i]=VZERO();wa[0]=wb[0]=VSET(t0);wa[1]=wb[1]=VSET(t1);wa[2]=wb[2]=VSET(t2);uint32_t na[16],nb[16];for(int i=0;i<16;i++){na[i]=bm_bswap32(base+i);nb[i]=bm_bswap32(base+16u+i);}wa[3]=_mm512_loadu_si512((void*)na);wb[3]=_mm512_loadu_si512((void*)nb);wa[4]=wb[4]=VSET(0x80000000u);wa[15]=wb[15]=VSET(640u);V sa[8],sb[8];for(int i=0;i<8;i++)sa[i]=sb[i]=VSET(mid[i]);compress2_roll(sa,wa,sb,wb);
 for(int i=0;i<16;i++){wa[i]=VZERO();wb[i]=VZERO();}
 for(int i=0;i<8;i++){wa[i]=sa[i];wb[i]=sb[i];}
 wa[8]=wb[8]=VSET(0x80000000u); wa[15]=wb[15]=VSET(256u);
 for(int i=0;i<8;i++){oa[i]=VSET(BM_IV256[i]);ob[i]=VSET(BM_IV256[i]);}
 compress2_roll(oa,wa,ob,wb);
}
static inline uint64_t fold(V x){uint32_t a[16];_mm512_storeu_si512((void*)a,x);uint64_t z=0;for(int i=0;i<16;i++)z^=((uint64_t)a[i]<<((i&1)*32));return z;}
uint64_t bm_avx512x2_scan_bench(const uint8_t*h,uint32_t start,uint64_t count,int threads){uint32_t mid[8];bm_header_midstate(h,mid);uint32_t t0=bm_be32(h+64),t1=bm_be32(h+68),t2=bm_be32(h+72);uint64_t chunks=count/32,chk=0;
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1) reduction(^:chk)
for(uint64_t j=0;j<chunks;j++){V a[8],b[8];hash32(mid,t0,t1,t2,start+(uint32_t)(j*32),a,b);chk^=fold(a[0])^fold(a[7])^fold(b[0])^fold(b[7]);}return chk^count;}
int bm_avx512x2_scan_target(const uint8_t*h,uint32_t start,uint64_t count,const uint8_t*t,int threads,uint32_t*fn,uint8_t*fh){uint32_t mid[8];bm_header_midstate(h,mid);uint32_t t0=bm_be32(h+64),t1=bm_be32(h+68),t2=bm_be32(h+72);uint64_t chunks=count/32,best=UINT64_MAX;
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1)
for(uint64_t j=0;j<chunks;j++){V a[8],b[8];hash32(mid,t0,t1,t2,start+(uint32_t)(j*32),a,b);uint32_t wa[8][16],wb[8][16];for(int k=0;k<8;k++){_mm512_storeu_si512(wa[k],a[k]);_mm512_storeu_si512(wb[k],b[k]);}for(int l=0;l<32;l++){uint8_t hh[32];uint32_t(*w)[16]=l<16?wa:wb;int q=l<16?l:l-16;for(int k=0;k<8;k++)bm_putbe32(hh+4*k,w[k][q]);if(bm_hash_le_target(hh,t)){uint64_t off=j*32+l;
#pragma omp critical(bitmyner_best_x2r)
{if(off<best)best=off;}}}}
if(best==UINT64_MAX){return 0;}
 *fn=start+(uint32_t)best;
 bm_sha256d_one(h,*fn,fh);
 return 1;
}
