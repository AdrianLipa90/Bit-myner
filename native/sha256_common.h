#ifndef BITMYNER_SHA256_COMMON_H
#define BITMYNER_SHA256_COMMON_H
#include <stdint.h>
#include <stddef.h>
#include <string.h>

static const uint32_t BM_K256[64] = {
0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
static const uint32_t BM_IV256[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
static inline uint32_t bm_rotr32(uint32_t x,unsigned n){return (x>>n)|(x<<(32-n));}
static inline uint32_t bm_be32(const uint8_t*p){return ((uint32_t)p[0]<<24)|((uint32_t)p[1]<<16)|((uint32_t)p[2]<<8)|(uint32_t)p[3];}
static inline uint32_t bm_bswap32(uint32_t x){return __builtin_bswap32(x);}
static inline void bm_putbe32(uint8_t*p,uint32_t x){p[0]=(uint8_t)(x>>24);p[1]=(uint8_t)(x>>16);p[2]=(uint8_t)(x>>8);p[3]=(uint8_t)x;}
static inline void bm_compress_scalar(uint32_t s[8], const uint8_t block[64]){
 uint32_t w[64]; for(int i=0;i<16;i++) w[i]=bm_be32(block+4*i);
 for(int i=16;i<64;i++){uint32_t a=w[i-15],b=w[i-2]; uint32_t s0=bm_rotr32(a,7)^bm_rotr32(a,18)^(a>>3); uint32_t s1=bm_rotr32(b,17)^bm_rotr32(b,19)^(b>>10); w[i]=w[i-16]+s0+w[i-7]+s1;}
 uint32_t a=s[0],b=s[1],c=s[2],d=s[3],e=s[4],f=s[5],g=s[6],h=s[7];
 for(int i=0;i<64;i++){uint32_t S1=bm_rotr32(e,6)^bm_rotr32(e,11)^bm_rotr32(e,25);uint32_t ch=(e&f)^((~e)&g);uint32_t t1=h+S1+ch+BM_K256[i]+w[i];uint32_t S0=bm_rotr32(a,2)^bm_rotr32(a,13)^bm_rotr32(a,22);uint32_t maj=(a&b)^(a&c)^(b&c);uint32_t t2=S0+maj;h=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;}
 s[0]+=a;s[1]+=b;s[2]+=c;s[3]+=d;s[4]+=e;s[5]+=f;s[6]+=g;s[7]+=h;
}
static inline void bm_header_midstate(const uint8_t header76[76], uint32_t out[8]){memcpy(out,BM_IV256,32);bm_compress_scalar(out,header76);}
static inline void bm_second_block(const uint8_t header76[76],uint32_t nonce,uint8_t out[64]){memset(out,0,64);memcpy(out,header76+64,12);out[12]=(uint8_t)nonce;out[13]=(uint8_t)(nonce>>8);out[14]=(uint8_t)(nonce>>16);out[15]=(uint8_t)(nonce>>24);out[16]=0x80;out[62]=0x02;out[63]=0x80;}
static inline void bm_digest_block_from_state(const uint32_t h1[8],uint8_t out[64]){memset(out,0,64);for(int i=0;i<8;i++)bm_putbe32(out+4*i,h1[i]);out[32]=0x80;out[62]=0x01;out[63]=0x00;}
static inline void bm_sha256d_one(const uint8_t header76[76],uint32_t nonce,uint8_t out[32]){uint32_t h1[8],h2[8];uint8_t b[64];bm_header_midstate(header76,h1);bm_second_block(header76,nonce,b);bm_compress_scalar(h1,b);memcpy(h2,BM_IV256,32);bm_digest_block_from_state(h1,b);bm_compress_scalar(h2,b);for(int i=0;i<8;i++)bm_putbe32(out+4*i,h2[i]);}
static inline int bm_hash_le_target(const uint8_t h[32],const uint8_t target_le[32]){for(int i=31;i>=0;i--){if(h[i]<target_le[i])return 1;if(h[i]>target_le[i])return 0;}return 1;}
#endif
