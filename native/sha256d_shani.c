#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <omp.h>
#include <openssl/sha.h>
#include "sha256_common.h"

static inline void ossl_midstate(const uint8_t header76[76], SHA256_CTX *pre){SHA256_Init(pre);SHA256_Transform(pre,header76);}
static inline void ossl_hash_one(const SHA256_CTX *pre,const uint8_t header76[76],uint32_t nonce,uint8_t out[32]){
 uint8_t b[64]; SHA256_CTX c=*pre; bm_second_block(header76,nonce,b); SHA256_Transform(&c,b);
 memset(b,0,64); for(int i=0;i<8;i++)bm_putbe32(b+4*i,c.h[i]); b[32]=0x80; b[62]=0x01; b[63]=0x00;
 SHA256_CTX d; SHA256_Init(&d); SHA256_Transform(&d,b); for(int i=0;i<8;i++)bm_putbe32(out+4*i,d.h[i]);
}
uint64_t bm_scan_bench(const uint8_t *header76,uint32_t start,uint64_t count,int threads){SHA256_CTX pre;ossl_midstate(header76,&pre);uint64_t chk=0;
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1) reduction(^:chk)
 for(uint64_t j=0;j<count;j++){uint8_t h[32];ossl_hash_one(&pre,header76,start+(uint32_t)j,h);uint64_t a,b;memcpy(&a,h,8);memcpy(&b,h+24,8);chk^=a^b;}return chk^count;}
int bm_scan_target(const uint8_t *header76,uint32_t start,uint64_t count,const uint8_t *target_le,int threads,uint32_t *found_nonce,uint8_t *found_hash){SHA256_CTX pre;ossl_midstate(header76,&pre);uint64_t best=UINT64_MAX;
#pragma omp parallel for schedule(static) num_threads(threads>0?threads:1)
 for(uint64_t j=0;j<count;j++){if(j>=best)continue;uint8_t h[32];ossl_hash_one(&pre,header76,start+(uint32_t)j,h);if(bm_hash_le_target(h,target_le)){
#pragma omp critical(bitmyner_best_shani)
 {if(j<best)best=j;}}}
 if(best==UINT64_MAX)return 0;*found_nonce=start+(uint32_t)best;ossl_hash_one(&pre,header76,*found_nonce,found_hash);return 1;}
