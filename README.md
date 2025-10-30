# ⚡ CUDA BLAKE2B GPU Implementation

[![CUDA](https://img.shields.io/badge/CUDA-11.0+-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

High-performance GPU implementation of BLAKE2B-256 cryptographic hash algorithm using NVIDIA CUDA. Achieves **150x speedup** compared to CPU implementations.

## 🚀 Quick Start

### Compile & Run
```bash
# Clone repository
git clone https://github.com/zeroqual/cuda-blake2b
cd cuda-blake2b

# Compile
nvcc -o test_blake2b main.cu

# Run tests
./test_blake2b