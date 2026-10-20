/*
 * SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

/* What includesProgramModels/nccl/include/nccl.h needs from CUDA's runtime
 * header, so that //test/abi:nccl_abi_test compiles it without a CUDA
 * install: the stream type, declared as CUDA 13.1's driver_types.h declares
 * it.  CUDART_VERSION stays undefined, which keeps nccl.h from including the
 * bf16 and fp8 headers it does not need for its declarations. */

#ifndef SPMD_TEST_NCCL_CUDA_STUB_CUDA_RUNTIME_H
#define SPMD_TEST_NCCL_CUDA_STUB_CUDA_RUNTIME_H

typedef struct CUstream_st *cudaStream_t;

#endif
