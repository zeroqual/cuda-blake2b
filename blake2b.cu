#include "constants/blake2b.h"

__device__ __constant__ uint64_t blake2b_IV[8] = {
    0x6a09e667f3bcc908ULL, 0xbb67ae8584caa73bULL,
    0x3c6ef372fe94f82bULL, 0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL, 0x9b05688c2b3e6c1fULL,
    0x1f83d9abfb41bd6bULL, 0x5be0cd19137e2179ULL};

__device__ __constant__ uint8_t blake2b_sigma[12][16] = {
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
    {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4},
    {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
    {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13},
    {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
    {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11},
    {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
    {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5},
    {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0},
    {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
    {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3}};

__device__ __forceinline__ uint64_t rotr64(uint64_t x, unsigned r)
{
    return (x >> r) | (x << (64 - r));
}

__device__ static inline void G(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t x, uint64_t y)
{
    a = a + b + x;
    d = rotr64(d ^ a, 32);
    c = c + d;
    b = rotr64(b ^ c, 24);
    a = a + b + y;
    d = rotr64(d ^ a, 16);
    c = c + d;
    b = rotr64(b ^ c, 63);
}

__device__ void blake2b_compress(uint64_t h[8], const uint8_t block[128], uint64_t t_low, uint64_t t_high, bool last)
{
    uint64_t m[16];

    for (int i = 0; i < 16; ++i)
    {
        const uint8_t *p = block + i * 8;
        m[i] = (uint64_t)p[0] | ((uint64_t)p[1] << 8) | ((uint64_t)p[2] << 16) | ((uint64_t)p[3] << 24) | ((uint64_t)p[4] << 32) | ((uint64_t)p[5] << 40) | ((uint64_t)p[6] << 48) | ((uint64_t)p[7] << 56);
    }

    uint64_t v[16];

    for (int i = 0; i < 8; ++i)
        v[i] = h[i];
    for (int i = 0; i < 8; ++i)
        v[i + 8] = blake2b_IV[i];

    v[12] ^= t_low;
    v[13] ^= t_high;
    if (last)
        v[14] ^= 0xFFFFFFFFFFFFFFFFULL;

    for (int r = 0; r < 12; ++r)
    {
        const uint8_t *s = blake2b_sigma[r];

        G(v[0], v[4], v[8], v[12], m[s[0]], m[s[1]]);
        G(v[1], v[5], v[9], v[13], m[s[2]], m[s[3]]);
        G(v[2], v[6], v[10], v[14], m[s[4]], m[s[5]]);
        G(v[3], v[7], v[11], v[15], m[s[6]], m[s[7]]);

        G(v[0], v[5], v[10], v[15], m[s[8]], m[s[9]]);
        G(v[1], v[6], v[11], v[12], m[s[10]], m[s[11]]);
        G(v[2], v[7], v[8], v[13], m[s[12]], m[s[13]]);
        G(v[3], v[4], v[9], v[14], m[s[14]], m[s[15]]);
    }

    for (int i = 0; i < 8; ++i)
    {
        h[i] = h[i] ^ v[i] ^ v[i + 8];
    }
}

__device__ static inline void blake2b_init_param(uint64_t outlen, uint64_t h[8])
{

    for (int i = 0; i < 8; ++i)
        h[i] = blake2b_IV[i];

    uint64_t param0 = (uint64_t)outlen | ((uint64_t)0 << 8) | ((uint64_t)1 << 16) | ((uint64_t)1 << 24) | ((uint64_t)0 << 32);
    h[0] ^= param0;
}

__device__ static inline void store64_le(uint8_t out[8], uint64_t w)
{
    out[0] = (uint8_t)(w & 0xFF);
    out[1] = (uint8_t)((w >> 8) & 0xFF);
    out[2] = (uint8_t)((w >> 16) & 0xFF);
    out[3] = (uint8_t)((w >> 24) & 0xFF);
    out[4] = (uint8_t)((w >> 32) & 0xFF);
    out[5] = (uint8_t)((w >> 40) & 0xFF);
    out[6] = (uint8_t)((w >> 48) & 0xFF);
    out[7] = (uint8_t)((w >> 56) & 0xFF);
}

__device__ void blake2b256_device_single(const uint8_t *in, size_t inlen, uint8_t out[32])
{
    uint64_t h[8];
    blake2b_init_param(32, h);

    uint64_t t_low = 0;
    uint64_t t_high = 0;
    uint8_t block[128];

    while (inlen > 128)
    {
        for (int i = 0; i < 128; ++i)
            block[i] = in[i];

        uint64_t prev = t_low;
        t_low += 128;
        if (t_low < prev)
            t_high++;

        blake2b_compress(h, block, t_low, t_high, false);
        in += 128;
        inlen -= 128;
    }

    for (int i = 0; i < 128; ++i)
        block[i] = 0;
    if (inlen > 0)
    {
        for (size_t i = 0; i < inlen; ++i)
            block[i] = in[i];
    }

    uint64_t prev = t_low;
    t_low += (uint64_t)inlen;
    if (t_low < prev)
        t_high++;

    blake2b_compress(h, block, t_low, t_high, true);

    for (int i = 0; i < 4; ++i)
    {
        store64_le(out + i * 8, h[i]);
    }
}