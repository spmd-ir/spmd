// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- OpenSHMEM.h - Predefined handles of the OpenSHMEM ABI ----*- C++ -*-===//
//
// The predefined handles and constants of Sandia OpenSHMEM 1.5, the
// implementation whose headers includesProgramModels/shmem/include carries
// (shmem-def.h:60).  Unlike MPICH, this ABI passes a team as a pointer to an
// opaque struct, and its predefined teams are extern variables rather than
// constants: a call takes SHMEM_TEAM_WORLD by loading that variable, not by
// passing a literal.  -convert-apiCalls-to-spmd recognises these in the calls
// it raises and -lower-spmd-to-shmem emits them in the calls it lowers to, so
// the two read them from this one place.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_TARGETABI_OPENSHMEM_H
#define SPMD_TARGETABI_OPENSHMEM_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd::abi::openshmem {

/// A handle together with the name the dialect gives it: the OpenSHMEM name
/// without its prefix, as spmd.constThreadLevel and spmd.constSignalOpType
/// spell it.
struct NamedHandle {
  llvm::StringLiteral name;
  int32_t handle;
};

// Thread support levels (shmem-def.h:70).  The encoding is MPI's, which is why
// a translated program keeps its spmd.constThreadLevel unchanged.
inline constexpr NamedHandle kThreadLevels[] = {
    {"SINGLE", 0},
    {"FUNNELED", 1},
    {"SERIALIZED", 2},
    {"MULTIPLE", 3},
};

// What shmem_put_signal's signal operation can be (shmem-def.h:114).
inline constexpr NamedHandle kSignalOps[] = {
    {"SET", 0},
    {"ADD", 1},
};

/// The predefined teams (shmem-def.h:103), which are extern variables of type
/// shmem_team_t rather than constants.  A lowering declares the variable and
/// loads it; there is no value to write down.
inline constexpr llvm::StringLiteral kTeamWorld = "SHMEM_TEAM_WORLD";
inline constexpr llvm::StringLiteral kTeamShared = "SHMEM_TEAM_SHARED";

/// SHMEM_TEAM_INVALID is NULL (shmem-def.h:110): a team handle is a pointer,
/// so the null team is the null pointer.
inline constexpr int64_t kTeamInvalid = 0;

/// What the routines that return an int return when they succeed.  OpenSHMEM
/// gives the value no name of its own -- the specification says "zero on
/// success, non-zero otherwise" -- and the routines that return void always
/// succeed, so a lowering fills their spmd.error with this.
inline constexpr int32_t kSuccess = 0;

/// The bits of shmem_team_split_strided's config_mask (shmem-def.h:117).  A
/// lowering that passes no shmem_team_config_t passes a null pointer and an
/// empty mask.
inline constexpr int64_t kTeamNumContexts = 1;
inline constexpr int64_t kTeamConfigMaskNone = 0;

/// A routine whose return type in this implementation differs from the one
/// the OpenSHMEM specification gives it -- which is what USIS describes, and so
/// what the lowering would otherwise declare.  A call declared with the other
/// type collides with the declaration a program compiled against shmem.h
/// already has.
struct ReturnTypeOverride {
  llvm::StringLiteral callee;
  llvm::StringLiteral cType;
};

// OpenSHMEM 1.5 declares shmem_team_destroy void; Sandia's returns an int
// status (shmem.h:2106).
inline constexpr ReturnTypeOverride kReturnTypeOverrides[] = {
    {"shmem_team_destroy", "int"},
};

inline std::optional<llvm::StringRef>
lookupReturnTypeOverride(llvm::StringRef callee) {
  for (const ReturnTypeOverride &entry : kReturnTypeOverrides)
    if (entry.callee == callee)
      return llvm::StringRef(entry.cType);
  return std::nullopt;
}

inline std::optional<int32_t> lookupHandle(llvm::ArrayRef<NamedHandle> table,
                                           llvm::StringRef name) {
  for (const NamedHandle &entry : table)
    if (entry.name == name)
      return entry.handle;
  return std::nullopt;
}

} // namespace mlir::spmd::abi::openshmem

#endif // SPMD_TARGETABI_OPENSHMEM_H
