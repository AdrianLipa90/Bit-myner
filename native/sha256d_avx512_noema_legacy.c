#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <omp.h>
#include <limits.h>

static const uint32_t K256[64] = {
0x428a2f98U,0x71374491U,0xb5c0fbcfU,0xe9b5dba5U,0x3956c25bU,0x59f111f1U,0x923f82a4U,0xab1c5ed5U,
0xd807aa98U,0x12835b01U,0x243185beU,0x550c7dc3U,0x72be5d74U,0x80deb1feU,0x9bdc06a7U,0xc19bf174U,
0xe49b69c1U,0xefbe4786U,0x0fc19dc6U,0x240ca1ccU,0x2de92c6fU,0x4a7484aaU,0x5cb0a9dcU,0x76f988daU,
0x983e5152U,0xa831c66dU,0xb00327c8U,0xbf597fc7U,0xc6e00bf3U,0xd5a79147U,0x06ca6351U,0x14292967U,
0x27b70a85U,0x2e1b2138U,0x4d2c6dfcU,0x53380d13U,0x650a7354U,0x766a0abbU,0x81c2c92eU,0x92722c85U,
0xa2bfe8a1U,0xa81a664bU,0xc24b8b70U,0xc76c51a3U,0xd192e819U,0xd6990624U,0xf40e3585U,0x106aa070U,
0x19a4c116U,0x1e376c08U,0x2748774cU,0x34b0bcb5U,0x391c0cb3U,0x4ed8aa4aU,0x5b9cca4fU,0x682e6ff3U,
0x748f82eeU,0x78a5636fU,0x84c87814U,0x8cc70208U,0x90befffaU,0xa4506cebU,0xbef9a3f7U,0xc67178f2U};

static const uint32_t IV256[8] = {0x6a09e667U,0xbb67ae85U,0x3c6ef372U,0xa54ff53aU,0x510e527fU,0x9b05688cU,0x1f83d9abU,0x5be0cd19U};

static inline uint32_t rotr32s(uint32_t x, unsigned n){ return (x>>n)|(x<<(32-n)); }
static inline uint32_t be32(const uint8_t *p){ return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|p[3]; }
static inline uint32_t bswap32s(uint32_t x){ return __builtin_bswap32(x); }

static void compress_scalar(uint32_t st[8], const uint32_t in16[16]){
    uint32_t w[64];
    for(int i=0;i<16;i++) w[i]=in16[i];
    for(int i=16;i<64;i++){
        uint32_t s0=rotr32s(w[i-15],7)^rotr32s(w[i-15],18)^(w[i-15]>>3);
        uint32_t s1=rotr32s(w[i-2],17)^rotr32s(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    uint32_t a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for(int i=0;i<64;i++){
        uint32_t S1=rotr32s(e,6)^rotr32s(e,11)^rotr32s(e,25);
        uint32_t ch=(e&f)^((~e)&g);
        uint32_t t1=h+S1+ch+K256[i]+w[i];
        uint32_t S0=rotr32s(a,2)^rotr32s(a,13)^rotr32s(a,22);
        uint32_t maj=(a&b)^(a&c)^(b&c);
        uint32_t t2=S0+maj;
        h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    st[0]+=a; st[1]+=b; st[2]+=c; st[3]+=d; st[4]+=e; st[5]+=f; st[6]+=g; st[7]+=h;
}

static void midstate_header64(const uint8_t header76[76], uint32_t mid[8]){
    uint32_t w[16];
    for(int i=0;i<16;i++) w[i]=be32(header76+4*i);
    memcpy(mid,IV256,sizeof(IV256));
    compress_scalar(mid,w);
}

#define VADD(a,b) _mm512_add_epi32((a),(b))
#define VXOR(a,b) _mm512_xor_si512((a),(b))
#define VAND(a,b) _mm512_and_si512((a),(b))
#define VANDN(a,b) _mm512_andnot_si512((a),(b))
#define VSHR(a,n) _mm512_srli_epi32((a),(n))
#define VROR(a,n) _mm512_ror_epi32((a),(n))

static inline __m512i bswap32v(__m512i x){
    __m512i m0=_mm512_set1_epi32(0x000000ff);
    __m512i m1=_mm512_set1_epi32(0x0000ff00);
    __m512i m2=_mm512_set1_epi32(0x00ff0000);
    __m512i m3=_mm512_set1_epi32((int)0xff000000U);
    __m512i a=_mm512_slli_epi32(_mm512_and_si512(x,m0),24);
    __m512i b=_mm512_slli_epi32(_mm512_and_si512(x,m1),8);
    __m512i c=_mm512_srli_epi32(_mm512_and_si512(x,m2),8);
    __m512i d=_mm512_srli_epi32(_mm512_and_si512(x,m3),24);
    return _mm512_or_si512(_mm512_or_si512(a,b),_mm512_or_si512(c,d));
}

static inline void compress16(__m512i st[8], __m512i w[64]){
    for(int i=16;i<64;i++){
        __m512i s0=VXOR(VXOR(VROR(w[i-15],7),VROR(w[i-15],18)),VSHR(w[i-15],3));
        __m512i s1=VXOR(VXOR(VROR(w[i-2],17),VROR(w[i-2],19)),VSHR(w[i-2],10));
        w[i]=VADD(VADD(VADD(w[i-16],s0),w[i-7]),s1);
    }
    __m512i a=st[0],b=st[1],c=st[2],d=st[3],e=st[4],f=st[5],g=st[6],h=st[7];
    for(int i=0;i<64;i++){
        __m512i S1=VXOR(VXOR(VROR(e,6),VROR(e,11)),VROR(e,25));
        __m512i ch=VXOR(VAND(e,f),VANDN(e,g));
        __m512i t1=VADD(VADD(VADD(VADD(h,S1),ch),_mm512_set1_epi32((int)K256[i])),w[i]);
        __m512i S0=VXOR(VXOR(VROR(a,2),VROR(a,13)),VROR(a,22));
        __m512i maj=VXOR(VXOR(VAND(a,b),VAND(a,c)),VAND(b,c));
        __m512i t2=VADD(S0,maj);
        h=g;g=f;f=e;e=VADD(d,t1);d=c;c=b;b=a;a=VADD(t1,t2);
    }
    st[0]=VADD(st[0],a); st[1]=VADD(st[1],b); st[2]=VADD(st[2],c); st[3]=VADD(st[3],d);
    st[4]=VADD(st[4],e); st[5]=VADD(st[5],f); st[6]=VADD(st[6],g); st[7]=VADD(st[7],h);
}

static inline void hash16(const uint32_t mid[8], const uint8_t tail12[12], uint32_t base_nonce, __m512i outst[8]){
    __m512i w[64];
    const uint32_t t0=be32(tail12),t1=be32(tail12+4),t2=be32(tail12+8);
    w[0]=_mm512_set1_epi32((int)t0); w[1]=_mm512_set1_epi32((int)t1); w[2]=_mm512_set1_epi32((int)t2);
    __m512i n=_mm512_setr_epi32(
        (int)(base_nonce+0),(int)(base_nonce+1),(int)(base_nonce+2),(int)(base_nonce+3),
        (int)(base_nonce+4),(int)(base_nonce+5),(int)(base_nonce+6),(int)(base_nonce+7),
        (int)(base_nonce+8),(int)(base_nonce+9),(int)(base_nonce+10),(int)(base_nonce+11),
        (int)(base_nonce+12),(int)(base_nonce+13),(int)(base_nonce+14),(int)(base_nonce+15));
    w[3]=bswap32v(n); w[4]=_mm512_set1_epi32((int)0x80000000U);
    for(int i=5;i<15;i++) w[i]=_mm512_setzero_si512();
    w[15]=_mm512_set1_epi32(640);
    __m512i st[8]; for(int i=0;i<8;i++) st[i]=_mm512_set1_epi32((int)mid[i]);
    compress16(st,w);
    for(int i=0;i<8;i++) w[i]=st[i];
    w[8]=_mm512_set1_epi32((int)0x80000000U);
    for(int i=9;i<15;i++) w[i]=_mm512_setzero_si512();
    w[15]=_mm512_set1_epi32(256);
    for(int i=0;i<8;i++) outst[i]=_mm512_set1_epi32((int)IV256[i]);
    compress16(outst,w);
}

int bitmyner_hash_one(const uint8_t header76[76], uint32_t nonce, uint8_t digest32[32]){
    uint32_t mid[8]; midstate_header64(header76,mid);
    __m512i st[8]; hash16(mid,header76+64,nonce,st);
    uint32_t lanes[16];
    for(int i=0;i<8;i++){
        _mm512_storeu_si512((void*)lanes,st[i]);
        uint32_t x=lanes[0];
        digest32[4*i+0]=(uint8_t)(x>>24); digest32[4*i+1]=(uint8_t)(x>>16); digest32[4*i+2]=(uint8_t)(x>>8); digest32[4*i+3]=(uint8_t)x;
    }
    return 0;
}

int bitmyner_scan_mt_mask(const uint8_t header76[76], uint32_t start_nonce, uint32_t count, uint64_t target_hi, int threads, uint32_t *found_nonce, uint32_t *checked){
    uint32_t mid[8]; midstate_header64(header76,mid);
    uint64_t groups=((uint64_t)count+15ULL)/16ULL;
    unsigned long long found64=ULLONG_MAX;
    uint32_t thi=(uint32_t)(target_hi>>32), tlo=(uint32_t)target_hi;
    __m512i vthi=_mm512_set1_epi32((int)thi), vtlo=_mm512_set1_epi32((int)tlo);
    if(threads<1) threads=1;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:found64)
    for(uint64_t g=0;g<groups;g++){
        uint32_t base=start_nonce+(uint32_t)(g*16ULL);
        uint32_t valid=16; uint64_t pos=g*16ULL;
        if(pos+16ULL>(uint64_t)count) valid=(uint32_t)((uint64_t)count-pos);
        __m512i st[8]; hash16(mid,header76+64,base,st);
        __m512i x6=bswap32v(st[6]), x7=bswap32v(st[7]);
        __mmask16 lt7=_mm512_cmplt_epu32_mask(x7,vthi);
        __mmask16 eq7=_mm512_cmpeq_epi32_mask(x7,vthi);
        __mmask16 le6=_mm512_cmple_epu32_mask(x6,vtlo);
        __mmask16 mask=(__mmask16)(lt7 | (eq7 & le6));
        if(valid<16) mask &= (__mmask16)((1u<<valid)-1u);
        if(mask){ unsigned lane=(unsigned)__builtin_ctz((unsigned)mask); uint64_t n=(uint64_t)base+lane; if(n<found64) found64=n; }
    }
    *checked=count;
    if(found64!=ULLONG_MAX){ *found_nonce=(uint32_t)found64; return 1; }
    return 0;
}

static inline void compress16_pair(__m512i A[8], __m512i wa[64], __m512i B[8], __m512i wb[64]){
    for(int i=16;i<64;i++){
        __m512i a0=VXOR(VXOR(VROR(wa[i-15],7),VROR(wa[i-15],18)),VSHR(wa[i-15],3));
        __m512i a1=VXOR(VXOR(VROR(wa[i-2],17),VROR(wa[i-2],19)),VSHR(wa[i-2],10));
        wa[i]=VADD(VADD(VADD(wa[i-16],a0),wa[i-7]),a1);
        __m512i b0=VXOR(VXOR(VROR(wb[i-15],7),VROR(wb[i-15],18)),VSHR(wb[i-15],3));
        __m512i b1=VXOR(VXOR(VROR(wb[i-2],17),VROR(wb[i-2],19)),VSHR(wb[i-2],10));
        wb[i]=VADD(VADD(VADD(wb[i-16],b0),wb[i-7]),b1);
    }
    __m512i aa=A[0],ab=A[1],ac=A[2],ad=A[3],ae=A[4],af=A[5],ag=A[6],ah=A[7];
    __m512i ba=B[0],bb=B[1],bc=B[2],bd=B[3],be=B[4],bf=B[5],bg=B[6],bh=B[7];
    for(int i=0;i<64;i++){
        __m512i k=_mm512_set1_epi32((int)K256[i]);
        __m512i aS1=VXOR(VXOR(VROR(ae,6),VROR(ae,11)),VROR(ae,25));
        __m512i ach=VXOR(VAND(ae,af),VANDN(ae,ag));
        __m512i at1=VADD(VADD(VADD(VADD(ah,aS1),ach),k),wa[i]);
        __m512i aS0=VXOR(VXOR(VROR(aa,2),VROR(aa,13)),VROR(aa,22));
        __m512i amaj=VXOR(VXOR(VAND(aa,ab),VAND(aa,ac)),VAND(ab,ac));
        __m512i at2=VADD(aS0,amaj);
        ah=ag;ag=af;af=ae;ae=VADD(ad,at1);ad=ac;ac=ab;ab=aa;aa=VADD(at1,at2);
        __m512i bS1=VXOR(VXOR(VROR(be,6),VROR(be,11)),VROR(be,25));
        __m512i bch=VXOR(VAND(be,bf),VANDN(be,bg));
        __m512i bt1=VADD(VADD(VADD(VADD(bh,bS1),bch),k),wb[i]);
        __m512i bS0=VXOR(VXOR(VROR(ba,2),VROR(ba,13)),VROR(ba,22));
        __m512i bmaj=VXOR(VXOR(VAND(ba,bb),VAND(ba,bc)),VAND(bb,bc));
        __m512i bt2=VADD(bS0,bmaj);
        bh=bg;bg=bf;bf=be;be=VADD(bd,bt1);bd=bc;bc=bb;bb=ba;ba=VADD(bt1,bt2);
    }
    A[0]=VADD(A[0],aa); A[1]=VADD(A[1],ab); A[2]=VADD(A[2],ac); A[3]=VADD(A[3],ad); A[4]=VADD(A[4],ae); A[5]=VADD(A[5],af); A[6]=VADD(A[6],ag); A[7]=VADD(A[7],ah);
    B[0]=VADD(B[0],ba); B[1]=VADD(B[1],bb); B[2]=VADD(B[2],bc); B[3]=VADD(B[3],bd); B[4]=VADD(B[4],be); B[5]=VADD(B[5],bf); B[6]=VADD(B[6],bg); B[7]=VADD(B[7],bh);
}

static inline __m512i nonce_vec(uint32_t base){
 return _mm512_setr_epi32((int)(base+0),(int)(base+1),(int)(base+2),(int)(base+3),(int)(base+4),(int)(base+5),(int)(base+6),(int)(base+7),(int)(base+8),(int)(base+9),(int)(base+10),(int)(base+11),(int)(base+12),(int)(base+13),(int)(base+14),(int)(base+15));
}

static inline void hash32(const uint32_t mid[8], const uint8_t tail12[12], uint32_t base, __m512i OA[8], __m512i OB[8]){
    __m512i wa[64], wb[64];
    uint32_t t0=be32(tail12),t1=be32(tail12+4),t2=be32(tail12+8);
    wa[0]=wb[0]=_mm512_set1_epi32((int)t0); wa[1]=wb[1]=_mm512_set1_epi32((int)t1); wa[2]=wb[2]=_mm512_set1_epi32((int)t2);
    wa[3]=bswap32v(nonce_vec(base)); wb[3]=bswap32v(nonce_vec(base+16));
    wa[4]=wb[4]=_mm512_set1_epi32((int)0x80000000U);
    for(int i=5;i<15;i++) wa[i]=wb[i]=_mm512_setzero_si512();
    wa[15]=wb[15]=_mm512_set1_epi32(640);
    __m512i A[8],B[8]; for(int i=0;i<8;i++) A[i]=B[i]=_mm512_set1_epi32((int)mid[i]);
    compress16_pair(A,wa,B,wb);
    for(int i=0;i<8;i++){ wa[i]=A[i]; wb[i]=B[i]; }
    wa[8]=wb[8]=_mm512_set1_epi32((int)0x80000000U);
    for(int i=9;i<15;i++) wa[i]=wb[i]=_mm512_setzero_si512();
    wa[15]=wb[15]=_mm512_set1_epi32(256);
    for(int i=0;i<8;i++) OA[i]=OB[i]=_mm512_set1_epi32((int)IV256[i]);
    compress16_pair(OA,wa,OB,wb);
}

int bitmyner_scan_mt_mask32(const uint8_t header76[76], uint32_t start_nonce, uint32_t count, uint64_t target_hi, int threads, uint32_t *found_nonce, uint32_t *checked){
    uint32_t mid[8]; midstate_header64(header76,mid);
    uint64_t groups=((uint64_t)count+31ULL)/32ULL;
    unsigned long long found64=ULLONG_MAX;
    uint32_t thi=(uint32_t)(target_hi>>32), tlo=(uint32_t)target_hi;
    __m512i vthi=_mm512_set1_epi32((int)thi), vtlo=_mm512_set1_epi32((int)tlo);
    if(threads<1) threads=1;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:found64)
    for(uint64_t g=0;g<groups;g++){
        uint32_t base=start_nonce+(uint32_t)(g*32ULL); uint64_t pos=g*32ULL;
        uint32_t valid=(pos+32ULL<=(uint64_t)count)?32:(uint32_t)((uint64_t)count-pos);
        __m512i A[8],B[8]; hash32(mid,header76+64,base,A,B);
        __m512i x6=bswap32v(A[6]),x7=bswap32v(A[7]);
        __mmask16 ma=(__mmask16)(_mm512_cmplt_epu32_mask(x7,vthi) | (_mm512_cmpeq_epi32_mask(x7,vthi)&_mm512_cmple_epu32_mask(x6,vtlo)));
        if(valid<16) ma &= (__mmask16)((1u<<valid)-1u);
        if(ma){ unsigned lane=(unsigned)__builtin_ctz((unsigned)ma); uint64_t n=(uint64_t)base+lane; if(n<found64) found64=n; }
        if(valid>16){
            x6=bswap32v(B[6]);x7=bswap32v(B[7]);
            __mmask16 mb=(__mmask16)(_mm512_cmplt_epu32_mask(x7,vthi) | (_mm512_cmpeq_epi32_mask(x7,vthi)&_mm512_cmple_epu32_mask(x6,vtlo)));
            if(valid<32) mb &= (__mmask16)((1u<<(valid-16))-1u);
            if(mb){ unsigned lane=(unsigned)__builtin_ctz((unsigned)mb); uint64_t n=(uint64_t)base+16+lane; if(n<found64) found64=n; }
        }
    }
    *checked=count;
    if(found64!=ULLONG_MAX){ *found_nonce=(uint32_t)found64; return 1; }
    return 0;
}
