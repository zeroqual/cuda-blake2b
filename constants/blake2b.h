#ifdef __cplusplus
extern "C"
{
#endif

#include <stdint.h>
#include <stddef.h>

    __device__ uint64_t rotr64(uint64_t x, unsigned r);
    __device__ void G(uint64_t &a, uint64_t &b, uint64_t &c, uint64_t &d, uint64_t x, uint64_t y);
    __device__ void blake2b_compress(uint64_t h[8], const uint8_t block[128], uint64_t t_low, uint64_t t_high, bool last);
    __device__ void blake2b_init_param(uint64_t outlen, uint64_t h[8]);
    __device__ void store64_le(uint8_t out[8], uint64_t w);
    __device__ void blake2b256_device_single(const uint8_t *in, size_t inlen, uint8_t out[32]);

#ifdef __cplusplus
}
#endif
