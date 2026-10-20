// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- NVSHMEM.h - Predefined handles of the NVSHMEM ABI --------*- C++ -*-===//
//
// The predefined handles and constants of NVSHMEM, as the headers in
// includesProgramModels/mynvshmem/include declare them.  Those headers are not
// NVIDIA's: they are a stand-in written for the verification, which declares
// every host routine as an ordinary function (-nvshmem-to-host-abi says what
// that means at link time).  The values here are NVIDIA's all the same --
// NVSHMEM 3.8's nvshmem_common.cuh, nvshmem_constants.h and
// nvshmem_common_transport.h give each of them the value below -- so a program
// lowered to them runs against the real library.
//
// Unlike OpenSHMEM's ABI, NVSHMEM passes a team as an int32_t and its
// predefined teams are enumerators, so a call takes NVSHMEM_TEAM_WORLD as the
// constant 0.  -convert-apiCalls-to-spmd recognises these in the calls it
// raises and -lower-spmd-to-nvshmem emits them in the calls it lowers to, so
// the two read them from this one place.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_TARGETABI_NVSHMEM_H
#define SPMD_TARGETABI_NVSHMEM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd::abi::nvshmem {

/// A handle together with the name the dialect gives it: the NVSHMEM name
/// without its prefix, as spmd.constThreadLevel and spmd.constSignalOpType
/// spell it.
struct NamedHandle {
  llvm::StringLiteral name;
  int32_t handle;
};

// Thread support levels (nvshmem_constants.h:17).  The encoding is MPI's and
// OpenSHMEM's, which is why a translated program keeps its
// spmd.constThreadLevel unchanged.
inline constexpr NamedHandle kThreadLevels[] = {
    {"SINGLE", 0},
    {"FUNNELED", 1},
    {"SERIALIZED", 2},
    {"MULTIPLE", 3},
};

// What nvshmem_<T>_put_signal's signal operation can be
// (nvshmem_common_transport.h:39).  Not OpenSHMEM's 0 and 1: NVSHMEM numbers
// them after its internal atomic operations.
inline constexpr NamedHandle kSignalOps[] = {
    {"SET", 9},
    {"ADD", 10},
};

/// A team is an int32_t (nvshmem_types.h:6), and the predefined teams are
/// enumerators (nvshmem_common.cuh:68): NVSHMEM_TEAM_WORLD, and
/// NVSHMEMX_TEAM_NODE, the PEs on this PE's node, which is what spmd.commNode
/// stands for.  NVSHMEM_TEAM_SHARED -- the PEs whose memory this PE can reach
/// directly -- has no operation of its own.
inline constexpr int32_t kTeamWorld = 0;
inline constexpr int32_t kTeamShared = 1;
inline constexpr int32_t kTeamNode = 2;
inline constexpr int32_t kTeamInvalid = -1;

/// What the routines that return an int return when they succeed.  NVSHMEM
/// gives the value no name of its own -- "zero on success, non-zero otherwise"
/// -- and the routines that return void always succeed, so a lowering fills
/// their spmd.error with this.
inline constexpr int32_t kSuccess = 0;

/// nvshmem_team_split_strided takes an optional nvshmem_team_config_t by
/// address and a mask saying which of its fields to read.  A lowering that
/// passes no configuration passes a null pointer and an empty mask.
inline constexpr int64_t kTeamConfigMaskNone = 0;

inline std::optional<int32_t> lookupHandle(llvm::ArrayRef<NamedHandle> table,
                                           llvm::StringRef name) {
  for (const NamedHandle &entry : table)
    if (entry.name == name)
      return entry.handle;
  return std::nullopt;
}

} // namespace mlir::spmd::abi::nvshmem

#endif // SPMD_TARGETABI_NVSHMEM_H
