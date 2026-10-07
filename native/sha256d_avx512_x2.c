#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <stdlib.h>
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
#define VROR(x,n) _mm512_ror_epi32((x),(n))
#define VXOR3(a,b,c) _mm512_ternarylogic_epi32((a),(b),(c),0x96)
static inline V s0(V x){return VXOR3(VROR(x,7),VROR(x,18),_mm512_srli_epi32(x,3));}
static inline V s1(V x){return VXOR3(VROR(x,17),VROR(x,19),_mm512_srli_epi32(x,10));}
static inline V S0(V x){return VXOR3(VROR(x,2),VROR(x,13),VROR(x,22));}
static inline V S1(V x){return VXOR3(VROR(x,6),VROR(x,11),VROR(x,25));}
static inline V Ch(V e,V f,V g){return _mm512_ternarylogic_epi32(e,f,g,0xCA);}
static inline V Maj(V a,V b,V c){return _mm512_ternarylogic_epi32(a,b,c,0xE8);}
static inline uint32_t rS0(uint32_t x){return bm_rotr32(x,2)^bm_rotr32(x,13)^bm_rotr32(x,22);}
static inline uint32_t rS1(uint32_t x){return bm_rotr32(x,6)^bm_rotr32(x,11)^bm_rotr32(x,25);}
static inline uint32_t rs0(uint32_t x){return bm_rotr32(x,7)^bm_rotr32(x,18)^(x>>3);}
static inline uint32_t rs1(uint32_t x){return bm_rotr32(x,17)^bm_rotr32(x,19)^(x>>10);}
static inline uint32_t rCh(uint32_t e,uint32_t f,uint32_t g){return (e&f)^((~e)&g);}
static inline uint32_t rMaj(uint32_t a,uint32_t b,uint32_t c){return (a&b)^(a&c)^(b&c);}

static inline void prehash3(const uint32_t mid[8],uint32_t t0,uint32_t t1,uint32_t t2,uint32_t p[8]){
 uint32_t a=mid[0],b=mid[1],c=mid[2],d=mid[3],e=mid[4],f=mid[5],g=mid[6],h=mid[7];
 uint32_t ws[3]={t0,t1,t2};
 for(int i=0;i<3;i++){
  uint32_t t1v=h+rS1(e)+rCh(e,f,g)+BM_K256[i]+ws[i];
  uint32_t t2v=rS0(a)+rMaj(a,b,c);
  h=g;g=f;f=e;e=d+t1v;d=c;c=b;b=a;a=t1v+t2v;
 }
 p[0]=a;p[1]=b;p[2]=c;p[3]=d;p[4]=e;p[5]=f;p[6]=g;p[7]=h;
}

static inline void round_pair(V *a,V *b,V *c,V *d,V *e,V *f,V *g,V *h,V w,int i){
 V t1=VADD(VADD(VADD(VADD(*h,S1(*e)),Ch(*e,*f,*g)),VSET(BM_K256[i])),w);
 V t2=VADD(S0(*a),Maj(*a,*b,*c));
 *h=*g;*g=*f;*f=*e;*e=VADD(*d,t1);*d=*c;*c=*b;*b=*a;*a=VADD(t1,t2);
}

static inline void firsthash32(const uint32_t mid[8],uint32_t t0,uint32_t t1,uint32_t t2,uint32_t base,V oa[8],V ob[8]){
 V wa[16],wb[16];
 for(int i=0;i<16;i++)wa[i]=wb[i]=VZERO();
 wa[0]=wb[0]=VSET(t0);wa[1]=wb[1]=VSET(t1);wa[2]=wb[2]=VSET(t2);
 const V offs=_mm512_setr_epi32(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15);
 static const uint8_t bswap_raw[64] __attribute__((aligned(64)))={3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12,3,2,1,0,7,6,5,4,11,10,9,8,15,14,13,12};
 const V bswap=_mm512_load_si512((const void*)bswap_raw);
 wa[3]=_mm512_shuffle_epi8(VADD(VSET(base),offs),bswap);
 wb[3]=_mm512_shuffle_epi8(VADD(VSET(base+16u),offs),bswap);
 wa[4]=wb[4]=VSET(0x80000000u);wa[15]=wb[15]=VSET(640u);
 uint32_t p[8];prehash3(mid,t0,t1,t2,p);
 V aa=VSET(p[0]),ba=VSET(p[1]),ca=VSET(p[2]),da=VSET(p[3]),ea=VSET(p[4]),fa=VSET(p[5]),ga=VSET(p[6]),ha=VSET(p[7]);
 V ab=aa,bb=ba,cb=ca,db=da,eb=ea,fb=fa,gb=ga,hb=ha;
 for(int i=3;i<16;i++){round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[i],i);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[i],i);}
 const uint32_t c16=t0+rs0(t1),c17=t1+rs0(t2)+rs1(640u);
 wa[0]=wb[0]=VSET(c16); round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[0],16);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[0],16);
 wa[1]=wb[1]=VSET(c17); round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[1],17);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[1],17);
 wa[2]=VADD(VSET(t2+rs1(c16)),s0(wa[3]));wb[2]=VADD(VSET(t2+rs1(c16)),s0(wb[3]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[2],18);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[2],18);
 wa[3]=VADD(wa[3],VSET(rs0(0x80000000u)+rs1(c17)));wb[3]=VADD(wb[3],VSET(rs0(0x80000000u)+rs1(c17)));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[3],19);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[3],19);
 wa[4]=VADD(VSET(0x80000000u),s1(wa[2]));wb[4]=VADD(VSET(0x80000000u),s1(wb[2]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[4],20);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[4],20);
 wa[5]=s1(wa[3]);wb[5]=s1(wb[3]);round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[5],21);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[5],21);
 wa[6]=VADD(VSET(640u),s1(wa[4]));wb[6]=VADD(VSET(640u),s1(wb[4]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[6],22);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[6],22);
 wa[7]=VADD(wa[0],s1(wa[5]));wb[7]=VADD(wb[0],s1(wb[5]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[7],23);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[7],23);
 wa[8]=VADD(wa[1],s1(wa[6]));wb[8]=VADD(wb[1],s1(wb[6]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[8],24);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[8],24);
 wa[9]=VADD(wa[2],s1(wa[7]));wb[9]=VADD(wb[2],s1(wb[7]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[9],25);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[9],25);
 wa[10]=VADD(wa[3],s1(wa[8]));wb[10]=VADD(wb[3],s1(wb[8]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[10],26);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[10],26);
 wa[11]=VADD(wa[4],s1(wa[9]));wb[11]=VADD(wb[4],s1(wb[9]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[11],27);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[11],27);
 wa[12]=VADD(wa[5],s1(wa[10]));wb[12]=VADD(wb[5],s1(wb[10]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[12],28);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[12],28);
 wa[13]=VADD(wa[6],s1(wa[11]));wb[13]=VADD(wb[6],s1(wb[11]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[13],29);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[13],29);
 wa[14]=VADD(VADD(VSET(rs0(640u)),wa[7]),s1(wa[12]));wb[14]=VADD(VADD(VSET(rs0(640u)),wb[7]),s1(wb[12]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[14],30);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[14],30);
 wa[15]=VADD(VADD(VADD(VSET(640u),s0(wa[0])),wa[8]),s1(wa[13]));wb[15]=VADD(VADD(VADD(VSET(640u),s0(wb[0])),wb[8]),s1(wb[13]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[15],31);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[15],31);
 for(int i=32;i<64;i++){int j=i&15;wa[j]=VADD(VADD(VADD(wa[j],s0(wa[(j+1)&15])),wa[(j+9)&15]),s1(wa[(j+14)&15]));wb[j]=VADD(VADD(VADD(wb[j],s0(wb[(j+1)&15])),wb[(j+9)&15]),s1(wb[(j+14)&15]));round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[j],i);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[j],i);}
 oa[0]=VADD(VSET(mid[0]),aa);oa[1]=VADD(VSET(mid[1]),ba);oa[2]=VADD(VSET(mid[2]),ca);oa[3]=VADD(VSET(mid[3]),da);oa[4]=VADD(VSET(mid[4]),ea);oa[5]=VADD(VSET(mid[5]),fa);oa[6]=VADD(VSET(mid[6]),ga);oa[7]=VADD(VSET(mid[7]),ha);
 ob[0]=VADD(VSET(mid[0]),ab);ob[1]=VADD(VSET(mid[1]),bb);ob[2]=VADD(VSET(mid[2]),cb);ob[3]=VADD(VSET(mid[3]),db);ob[4]=VADD(VSET(mid[4]),eb);ob[5]=VADD(VSET(mid[5]),fb);ob[6]=VADD(VSET(mid[6]),gb);ob[7]=VADD(VSET(mid[7]),hb);
}

static inline void second_full_pair(V inA[8],V inB[8],V outA[8],V outB[8]){
 V wa[16],wb[16];for(int i=0;i<16;i++)wa[i]=wb[i]=VZERO();
 for(int i=0;i<8;i++){wa[i]=inA[i];wb[i]=inB[i];}
 wa[8]=wb[8]=VSET(0x80000000u);wa[15]=wb[15]=VSET(256u);
 V aa=VSET(BM_IV256[0]),ba=VSET(BM_IV256[1]),ca=VSET(BM_IV256[2]),da=VSET(BM_IV256[3]),ea=VSET(BM_IV256[4]),fa=VSET(BM_IV256[5]),ga=VSET(BM_IV256[6]),ha=VSET(BM_IV256[7]);
 V ab=aa,bb=ba,cb=ca,db=da,eb=ea,fb=fa,gb=ga,hb=ha;
 for(int i=0;i<64;i++){
  int j=i&15;if(i>=16){wa[j]=VADD(VADD(VADD(wa[j],s0(wa[(j+1)&15])),wa[(j+9)&15]),s1(wa[(j+14)&15]));wb[j]=VADD(VADD(VADD(wb[j],s0(wb[(j+1)&15])),wb[(j+9)&15]),s1(wb[(j+14)&15]));}
  round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[j],i);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[j],i);
 }
 outA[0]=VADD(VSET(BM_IV256[0]),aa);outA[1]=VADD(VSET(BM_IV256[1]),ba);outA[2]=VADD(VSET(BM_IV256[2]),ca);outA[3]=VADD(VSET(BM_IV256[3]),da);outA[4]=VADD(VSET(BM_IV256[4]),ea);outA[5]=VADD(VSET(BM_IV256[5]),fa);outA[6]=VADD(VSET(BM_IV256[6]),ga);outA[7]=VADD(VSET(BM_IV256[7]),ha);
 outB[0]=VADD(VSET(BM_IV256[0]),ab);outB[1]=VADD(VSET(BM_IV256[1]),bb);outB[2]=VADD(VSET(BM_IV256[2]),cb);outB[3]=VADD(VSET(BM_IV256[3]),db);outB[4]=VADD(VSET(BM_IV256[4]),eb);outB[5]=VADD(VSET(BM_IV256[5]),fb);outB[6]=VADD(VSET(BM_IV256[6]),gb);outB[7]=VADD(VSET(BM_IV256[7]),hb);
}

/* Compute the final SHA256 output word 7 exactly after only rounds 0..60.
   SHA256 round shifting implies h_after_round_63 == e_after_round_60, so rounds
   61..63 cannot change final word 7. This is an exact Bitcoin target precheck. */
static inline void second_word7_pair(V inA[8],V inB[8],V *word7A,V *word7B){
 V wa[16],wb[16];for(int i=0;i<16;i++)wa[i]=wb[i]=VZERO();
 for(int i=0;i<8;i++){wa[i]=inA[i];wb[i]=inB[i];}
 wa[8]=wb[8]=VSET(0x80000000u);wa[15]=wb[15]=VSET(256u);
 V aa=VSET(BM_IV256[0]),ba=VSET(BM_IV256[1]),ca=VSET(BM_IV256[2]),da=VSET(BM_IV256[3]),ea=VSET(BM_IV256[4]),fa=VSET(BM_IV256[5]),ga=VSET(BM_IV256[6]),ha=VSET(BM_IV256[7]);
 V ab=aa,bb=ba,cb=ca,db=da,eb=ea,fb=fa,gb=ga,hb=ha;
 for(int i=0;i<=60;i++){
  int j=i&15;if(i>=16){wa[j]=VADD(VADD(VADD(wa[j],s0(wa[(j+1)&15])),wa[(j+9)&15]),s1(wa[(j+14)&15]));wb[j]=VADD(VADD(VADD(wb[j],s0(wb[(j+1)&15])),wb[(j+9)&15]),s1(wb[(j+14)&15]));}
  round_pair(&aa,&ba,&ca,&da,&ea,&fa,&ga,&ha,wa[j],i);round_pair(&ab,&bb,&cb,&db,&eb,&fb,&gb,&hb,wb[j],i);
 }
 *word7A=VADD(VSET(BM_IV256[7]),ea);*word7B=VADD(VSET(BM_IV256[7]),eb);
}

static inline uint64_t fold(V x){uint32_t a[16];_mm512_storeu_si512((void*)a,x);uint64_t z=0;for(int i=0;i<16;i++)z^=((uint64_t)a[i]<<((i&1)*32));return z;}

uint64_t bm_avx512x2_scan_bench(const uint8_t*h,uint32_t start,uint64_t count,int threads){
 uint32_t mid[8];bm_header_midstate(h,mid);uint32_t t0=bm_be32(h+64),t1=bm_be32(h+68),t2=bm_be32(h+72);uint64_t chunks=count/32,chk=0;
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1) reduction(^:chk)
 for(uint64_t j=0;j<chunks;j++){V a[8],b[8],oa[8],ob[8];firsthash32(mid,t0,t1,t2,start+(uint32_t)(j*32),a,b);second_full_pair(a,b,oa,ob);chk^=fold(oa[0])^fold(oa[7])^fold(ob[0])^fold(ob[7]);}
 return chk^count;
}

int bm_avx512x2_scan_target(const uint8_t*h,uint32_t start,uint64_t count,const uint8_t*t,int threads,uint32_t*fn,uint8_t*fh){
 uint32_t mid[8];bm_header_midstate(h,mid);uint32_t t0=bm_be32(h+64),t1=bm_be32(h+68),t2=bm_be32(h+72);uint64_t chunks=count/32,best=UINT64_MAX;
 uint32_t target_hi=(uint32_t)t[28]|((uint32_t)t[29]<<8)|((uint32_t)t[30]<<16)|((uint32_t)t[31]<<24);
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1)
 for(uint64_t j=0;j<chunks;j++){
  V a[8],b[8];firsthash32(mid,t0,t1,t2,start+(uint32_t)(j*32),a,b);
  uint64_t off=UINT64_MAX;
  if(target_hi==0){
   V w7a,w7b;second_word7_pair(a,b,&w7a,&w7b);
   __mmask16 eqA=_mm512_cmpeq_epi32_mask(w7a,VZERO());
   __mmask16 eqB=_mm512_cmpeq_epi32_mask(w7b,VZERO());
   if(!(eqA|eqB)) continue;
   V oa[8],ob[8];second_full_pair(a,b,oa,ob);uint32_t wa[8][16],wb[8][16];
   if(eqA)for(int k=0;k<8;k++)_mm512_storeu_si512((void*)wa[k],oa[k]);
   if(eqB)for(int k=0;k<8;k++)_mm512_storeu_si512((void*)wb[k],ob[k]);
   for(int l=0;l<16 && off==UINT64_MAX;l++)if(eqA&(1u<<l)){uint8_t hh[32];for(int k=0;k<8;k++)bm_putbe32(hh+4*k,wa[k][l]);if(bm_hash_le_target(hh,t))off=j*32+(uint64_t)l;}
   for(int l=0;l<16 && off==UINT64_MAX;l++)if(eqB&(1u<<l)){uint8_t hh[32];for(int k=0;k<8;k++)bm_putbe32(hh+4*k,wb[k][l]);if(bm_hash_le_target(hh,t))off=j*32+16u+(uint64_t)l;}
  }else{
   /* General-target fallback: full exact comparison preserves scan semantics. */
   V oa[8],ob[8];second_full_pair(a,b,oa,ob);uint32_t wa[8][16],wb[8][16];
   for(int k=0;k<8;k++){_mm512_storeu_si512((void*)wa[k],oa[k]);_mm512_storeu_si512((void*)wb[k],ob[k]);}
   for(int l=0;l<16 && off==UINT64_MAX;l++){uint8_t hh[32];for(int k=0;k<8;k++)bm_putbe32(hh+4*k,wa[k][l]);if(bm_hash_le_target(hh,t))off=j*32+(uint64_t)l;}
   for(int l=0;l<16 && off==UINT64_MAX;l++){uint8_t hh[32];for(int k=0;k<8;k++)bm_putbe32(hh+4*k,wb[k][l]);if(bm_hash_le_target(hh,t))off=j*32+16u+(uint64_t)l;}
  }
  if(off!=UINT64_MAX){
#pragma omp critical(bitmyner_best_x2r)
   {if(off<best)best=off;}
  }
 }
 if(best==UINT64_MAX)return 0;*fn=start+(uint32_t)best;bm_sha256d_one(h,*fn,fh);return 1;
}


typedef struct { uint64_t off; uint8_t hash[32]; } bm_event_row;
static int bm_event_cmp(const void *a,const void *b){
 const bm_event_row *x=(const bm_event_row*)a,*y=(const bm_event_row*)b;
 return x->off<y->off?-1:(x->off>y->off?1:0);
}
static inline int bm_word7_lz(uint32_t h7){
 uint32_t display_prefix=bm_bswap32(h7);
 return display_prefix?__builtin_clz(display_prefix):32;
}

uint64_t bm_avx512x2_scan_events(
 const uint8_t*h,uint32_t start,uint64_t count,int threshold_bits,int threads,
 uint32_t capacity,uint32_t*out_count,uint32_t*out_nonces,uint8_t*out_hashes){
 if(!out_count||threshold_bits<1||threshold_bits>32)return UINT64_MAX;
 const int nt=threads>0?threads:1;
 const uint64_t chunks=count/32;
 uint32_t mid[8];bm_header_midstate(h,mid);
 uint32_t t0=bm_be32(h+64),t1=bm_be32(h+68),t2=bm_be32(h+72);
 uint32_t *local_counts=(uint32_t*)calloc((size_t)nt,sizeof(uint32_t));
 uint64_t *local_hits=(uint64_t*)calloc((size_t)nt,sizeof(uint64_t));
 size_t slots=(size_t)nt*(size_t)capacity;
 bm_event_row *local_rows=slots?(bm_event_row*)malloc(slots*sizeof(bm_event_row)):NULL;
 if(!local_counts||!local_hits||(slots&&!local_rows)){free(local_counts);free(local_hits);free(local_rows);return UINT64_MAX;}
#pragma omp parallel num_threads(nt)
 {
  int tid=omp_get_thread_num();
  uint32_t kept=0;uint64_t hits=0;
#pragma omp for schedule(static)
  for(uint64_t j=0;j<chunks;j++){
   V a[8],b[8],oa[8],ob[8];
   firsthash32(mid,t0,t1,t2,start+(uint32_t)(j*32),a,b);
   second_full_pair(a,b,oa,ob);
   uint32_t a7[16],b7[16];_mm512_storeu_si512((void*)a7,oa[7]);_mm512_storeu_si512((void*)b7,ob[7]);
   uint32_t mask=0;
   for(int l=0;l<16;l++){if(bm_word7_lz(a7[l])>=threshold_bits)mask|=(1u<<l);}
   for(int l=0;l<16;l++){if(bm_word7_lz(b7[l])>=threshold_bits)mask|=(1u<<(l+16));}
   if(!mask)continue;
   uint32_t wa[8][16],wb[8][16];int words_ready=0;
   for(int l=0;l<32;l++){
    if(!(mask&(1u<<l)))continue;
    hits++;
    if(kept>=capacity)continue;
    if(!words_ready){for(int k=0;k<8;k++){_mm512_storeu_si512((void*)wa[k],oa[k]);_mm512_storeu_si512((void*)wb[k],ob[k]);}words_ready=1;}
    bm_event_row *row=&local_rows[(size_t)tid*capacity+kept++];
    row->off=j*32u+(uint64_t)l;
    uint32_t(*w)[16]=l<16?wa:wb;int q=l<16?l:l-16;
    for(int k=0;k<8;k++)bm_putbe32(row->hash+4*k,w[k][q]);
   }
  }
  local_counts[tid]=kept;local_hits[tid]=hits;
 }
 uint64_t total_hits=0;size_t merged_n=0;
 for(int ti=0;ti<nt;ti++){total_hits+=local_hits[ti];merged_n+=local_counts[ti];}
 bm_event_row *merged=merged_n?(bm_event_row*)malloc(merged_n*sizeof(bm_event_row)):NULL;
 if(merged_n&&!merged){free(local_counts);free(local_hits);free(local_rows);return UINT64_MAX;}
 size_t pos=0;
 for(int ti=0;ti<nt;ti++)for(uint32_t i=0;i<local_counts[ti];i++)merged[pos++]=local_rows[(size_t)ti*capacity+i];
 if(merged_n>1)qsort(merged,merged_n,sizeof(bm_event_row),bm_event_cmp);
 uint32_t emit=(uint32_t)(merged_n<(size_t)capacity?merged_n:(size_t)capacity);
 for(uint32_t i=0;i<emit;i++){
  if(out_nonces)out_nonces[i]=start+(uint32_t)merged[i].off;
  if(out_hashes)memcpy(out_hashes+(size_t)i*32,merged[i].hash,32);
 }
 *out_count=emit;
 free(merged);free(local_rows);free(local_hits);free(local_counts);
 return total_hits;
}
