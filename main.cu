#include <iostream>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

#include "blake2b.cu"

const int BLOCKS = 128;
const int THREADS = 256;

__global__ void test_blake2b_kernel(uint8_t *results, const uint8_t *test_inputs, int input_len)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t hash_result[32];

    blake2b256_device_single(test_inputs, input_len, hash_result);

    for (int i = 0; i < 32; i++)
    {
        results[idx * 32 + i] = hash_result[i];
    }
}

__global__ void test_blake2b_multiple_kernel(uint8_t *results)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    uint8_t test_data[64];
    uint8_t hash_result[32];

    for (int i = 0; i < 64; i++)
    {
        test_data[i] = (uint8_t)((idx + i) & 0xFF);
    }

    blake2b256_device_single(test_data, 64, hash_result);

    for (int i = 0; i < 32; i++)
    {
        results[idx * 32 + i] = hash_result[i];
    }
}

void print_hex(const uint8_t *data, size_t len)
{
    for (size_t i = 0; i < len; i++)
    {
        printf("%02x", data[i]);
    }
    printf("\n");
}

void test_with_known_vectors()
{
    std::cout << "\nTesting with known test vectors on GPU...\n";

    uint8_t empty_input[] = "";
    uint8_t *d_empty_input, *d_empty_result;
    uint8_t h_empty_result[32];

    cudaMalloc(&d_empty_input, 1);
    cudaMalloc(&d_empty_result, 32);

    cudaMemcpy(d_empty_input, empty_input, 1, cudaMemcpyHostToDevice);

    test_blake2b_kernel<<<1, 1>>>(d_empty_result, d_empty_input, 0);
    cudaDeviceSynchronize();

    cudaMemcpy(h_empty_result, d_empty_result, 32, cudaMemcpyDeviceToHost);

    std::cout << "Empty input hash: ";
    print_hex(h_empty_result, 32);

    uint8_t abc_input[] = "abc";
    uint8_t *d_abc_input, *d_abc_result;
    uint8_t h_abc_result[32];

    cudaMalloc(&d_abc_input, 3);
    cudaMalloc(&d_abc_result, 32);

    cudaMemcpy(d_abc_input, abc_input, 3, cudaMemcpyHostToDevice);

    test_blake2b_kernel<<<1, 1>>>(d_abc_result, d_abc_input, 3);
    cudaDeviceSynchronize();

    cudaMemcpy(h_abc_result, d_abc_result, 32, cudaMemcpyDeviceToHost);

    std::cout << "\"abc\" hash: ";
    print_hex(h_abc_result, 32);

    cudaFree(d_empty_input);
    cudaFree(d_empty_result);
    cudaFree(d_abc_input);
    cudaFree(d_abc_result);
}

int main()
{
    std::cout << "Testing BLAKE2B on GPU...\n";

    const int total_threads = BLOCKS * THREADS;
    const size_t results_size = total_threads * 32;

    uint8_t *d_results = nullptr;
    cudaMalloc(&d_results, results_size);

    uint8_t *h_results = new uint8_t[results_size];

    std::cout << "Running kernel with " << BLOCKS << " blocks, " << THREADS << " threads\n";
    std::cout << "Total threads: " << total_threads << "\n";

    test_blake2b_multiple_kernel<<<BLOCKS, THREADS>>>(d_results);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess)
    {
        std::cerr << "CUDA kernel error: " << cudaGetErrorString(err) << "\n";
        return 1;
    }

    cudaMemcpy(h_results, d_results, results_size, cudaMemcpyDeviceToHost);

    std::cout << "\nFirst 5 hash results:\n";
    for (int i = 0; i < 5 && i < total_threads; i++)
    {
        std::cout << "Thread " << i << ": ";
        print_hex(h_results + i * 32, 32);
    }

    bool all_different = true;
    for (int i = 1; i < total_threads && i < 100; i++)
    {
        if (memcmp(h_results, h_results + i * 32, 32) == 0)
        {
            all_different = false;
            std::cout << "Found duplicate at index " << i << "\n";
            break;
        }
    }

    std::cout << "\nAll hashes are different: " << (all_different ? "YES" : "NO") << "\n";

    test_with_known_vectors();

    delete[] h_results;
    cudaFree(d_results);

    std::cout << "Test completed successfully!\n";
    return 0;
}