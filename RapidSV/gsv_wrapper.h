#ifndef _GSV_WRAPPER_H_
#define _GSV_WRAPPER_H_

#include <cstdint>

#define GSV_TPI 16

// #define GSV_256BIT

#ifdef GSV_256BIT  // use 256-bit integer instead of 512-bit
#define GSV_BITS 256
#else
#define GSV_BITS 512
#endif

// #define GSV_KNOWN_PKEY

typedef struct {
    uint32_t _limbs[(GSV_BITS + 31) / 32];
} gsv_mem_t;

typedef struct {
    gsv_mem_t r;  // sig->r
    gsv_mem_t s;  // sig->s
    gsv_mem_t e;  // digest
#ifndef GSV_KNOWN_PKEY
    gsv_mem_t key_x;  // public key
    gsv_mem_t key_y;  // public key
#endif
} sig_t;

void GSV_init(int device_id = 0);

void GSV_verify(int count, sig_t *sig, int *results);

void GSV_close();

#endif  // _GSV_WRAPPER_H_
