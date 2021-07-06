# Crypto-Accelerator-HKUST

## RapidSV

GPU-accelerated elliptic curve signature verification library.

### Dependencies

- NVIDIA CUDA Toolkit
- GMP (GNU Multiple Precision Library)

### Build the example (main.cpp)

1. Build with CMake
```
mkdir build && cd build
cmake -DCMAKE_CUDA_FLAGS="-arch=sm_70" ..
make
```

2. Build manually
```
mkdir -p bin
nvcc -O3 -gencode arch=compute_70,code=sm_70 -o bin/gsv.o -c gsv.cu
g++ -O3 -I/usr/local/cuda-11/include -o bin/main.o -c main.cpp
g++ -O3 -o bin/main bin/gsv.o bin/main.o -L/usr/local/cuda-11/lib64 -lcuda -lcudart -lgmp
```

### API description

```
#include "RapidSV/gsv_wrapper.h"
void GSV_init(int device_id = 0);
void GSV_verify(int count, sig_t *sig, int *results);
void GSV_close();
```

`GSV_init()` initializes the GPU device, allocates the GPU memory pool, and copies the precomputed table to constant memory.

`GSV_verify()` does the signature verification.
- The first parameter is the number of signatures.
- The second parameter is the array of signatures. Each signature is a `sig_t` struct that contains 5 big integers, namely signature `(r, s)`, message digest `e`, and public key `(key_x, key_y)`. Each big integer is represented by an array of unsigned 32-bit integers.
- The third parameter is the array of verification results. Correct verification returns 0, otherwise 1.

`GSV_close()` frees GPU memory.

## TASSL-acc

A fork of [TASSL-1.1.1b](https://github.com/jntass/TASSL-1.1.1b) with optimizations for the SM2 curve.

### Usage

Drop-in replacement of TASSL.
