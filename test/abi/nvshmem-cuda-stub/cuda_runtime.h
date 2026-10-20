/*
 * SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* What includesProgramModels/mynvshmem/include needs from CUDA's runtime
 * header, so that //test/abi:nvshmem_abi_test compiles it without a CUDA
 * install: the execution-space qualifiers, which CUDA 13.1's
 * crt/host_defines.h defines to nothing for a compiler that is not compiling
 * CUDA, and the two types nvshmemx.h's launch routines take, declared as its
 * driver_types.h and vector_types.h declare them. */

#ifndef SPMD_TEST_NVSHMEM_CUDA_STUB_CUDA_RUNTIME_H
#define SPMD_TEST_NVSHMEM_CUDA_STUB_CUDA_RUNTIME_H

#define __host__
#define __device__

typedef struct CUstream_st *cudaStream_t;

struct dim3 {
  unsigned int x, y, z;
};

#endif
