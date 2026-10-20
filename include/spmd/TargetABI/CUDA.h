// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CUDA.h - Predefined constants of the CUDA runtime ABI ----*- C++ -*-===//
//
// The constants of the CUDA runtime API the lowering of the dialect to CUDA
// rests on: the success code every routine returns, and the directions of
// `enum cudaMemcpyKind`.  The values are those of CUDA 13.1.0, whose
// driver_types.h each one quotes.
//
// Unlike MPICH.h, NCCL.h and OpenSHMEM.h, this table has no test in //test/abi
// against the header it copies.  includesProgramModels/cuda/ holds only the
// USIS API description; there is no vendored cuda_runtime_api.h, and the one
// on this machine comes from an environment module outside the build, so a
// cc_test could only be compiled non-hermetically -- and one guarded by
// __has_include would pass silently in the Bazel sandbox, which never sees it.
// Until CUDA's headers are vendored or a CUDA toolchain is wired into the
// build, these five numbers are unchecked.
//
// -lower-spmd-to-cuda does not emit a direction of its own: spmd.memcpy's
// $kind is the i32 the program itself computed and it passes through.  The
// table is what checkSPMDToCUDA reads to report a *constant* $kind that is no
// direction at all, which cudaMemcpy would reject at run time.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_TARGETABI_CUDA_H
#define SPMD_TARGETABI_CUDA_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd::abi::cuda {

/// cudaSuccess (driver_types.h:199), what every runtime routine returns when
/// it succeeds.  cudaError_t is a C enum, so an !spmd.error of CUDA is an i32.
inline constexpr int32_t kSuccess = 0;

/// One direction of a copy, with the name driver_types.h gives it.
struct NamedKind {
  llvm::StringLiteral name;
  int32_t value;
};

/// enum cudaMemcpyKind (driver_types.h:1263), the fourth argument of
/// cudaMemcpy.  cudaMemcpyDefault infers the direction from the pointers and
/// needs unified virtual addressing.
inline constexpr NamedKind kMemcpyKinds[] = {
    {"cudaMemcpyHostToHost", 0},   {"cudaMemcpyHostToDevice", 1},
    {"cudaMemcpyDeviceToHost", 2}, {"cudaMemcpyDeviceToDevice", 3},
    {"cudaMemcpyDefault", 4},
};

/// The name of the direction \p value, or nullopt when cudaMemcpyKind has no
/// enumerator with that value.
inline std::optional<llvm::StringRef> lookupMemcpyKind(int64_t value) {
  for (const NamedKind &kind : kMemcpyKinds)
    if (kind.value == value)
      return kind.name;
  return std::nullopt;
}

} // namespace mlir::spmd::abi::cuda

#endif // SPMD_TARGETABI_CUDA_H
