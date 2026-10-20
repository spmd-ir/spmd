// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToSHMEMFamily.h - Lower to OpenSHMEM or NVSHMEM -*- C++ -*-===//
//
// One lowering for the two models of the OpenSHMEM family.  NVSHMEM's host API
// is OpenSHMEM's with an nvshmem_ prefix, and both share what makes the
// lowering what it is: a window is the symmetric address it is over, no call
// hands out a request and the model's quiet completes everything, and the
// element type and the reduction operation are spelled in the callee's name.
// What differs is the ABI, which an SHMEMTarget describes;
// LowerSPMDToSHMEM.cpp and LowerSPMDToNVSHMEM.cpp each hold one and the pass
// that uses it.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDTOSHMEMFAMILY_H
#define SPMD_CONVERSION_LOWERSPMDTOSHMEMFAMILY_H

#include "LowerSPMDCommon.h"
#include "SHMEMLoweringTable.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir::spmd {

/// What one model of the family is, to the lowering: its names, and the ABI
/// of the handles the dialect keeps.
struct SHMEMTarget {
  ProgrammingModel model;
  SHMEMFlavour flavour;
  /// What the diagnostics call the pass and the model, and the prefix of the
  /// model's routines: the lowering names shmem_quiet or nvshmem_quiet as
  /// prefix + kQuiet.
  StringLiteral passName;
  StringLiteral displayName;
  StringLiteral prefix;
  /// What to set, and to what, when the API description is missing.
  StringLiteral jsonVariable;
  StringLiteral jsonFile;
  /// How the ABI passes a team.  OpenSHMEM's is a pointer, and its predefined
  /// teams are extern variables the lowering loads (\p worldTeamVariable,
  /// \p nodeTeamVariable); NVSHMEM's is an int32_t, and its predefined teams
  /// are constants (\p worldTeam, \p nodeTeam).
  bool teamIsPointer;
  StringLiteral worldTeamVariable;
  StringLiteral nodeTeamVariable;
  int32_t worldTeam;
  int32_t nodeTeam;
  /// The model's value of a thread level and a signal operation, by the name
  /// spmd.constThreadLevel and spmd.constSignalOpType give it.
  std::optional<int32_t> (*threadLevel)(StringRef name);
  std::optional<int32_t> (*signalOp)(StringRef name);
  /// What a routine that succeeded returns.
  int32_t success;
  /// The C return type the implementation gives \p callee, where it differs
  /// from the one USIS describes.
  std::optional<StringRef> (*returnTypeOverride)(StringRef callee);
  /// Whether another PE can reach the program's global variables.  OpenSHMEM
  /// makes them symmetric; NVSHMEM reaches only the symmetric heap
  /// nvshmem_malloc allocates on the GPU.
  bool globalsAreSymmetric;
};

/// Reports every operation tagged for \p target's model that the model cannot
/// express: a rooted reduction, a communicator split by colour and key, a
/// non-blocking collective, two-sided messaging, a window provably outside the
/// memory another PE can reach, a free of a window whose origin is not in
/// sight, an operation inside a GPU kernel, and an operand no call has a
/// parameter for.  A driver lowering several models runs this for each of
/// them before it rewrites anything, so that no diagnostic arrives after half
/// the module has changed.
LogicalResult checkSPMDToSHMEMFamily(ModuleOp module,
                                     const SHMEMTarget &target);

/// Lowers every operation in \p module tagged for \p target's model to the
/// call it models, and leaves the operations of other models alone.  Every
/// operation the model cannot express is reported before anything is
/// rewritten; on failure the module may be partly lowered.
LogicalResult lowerSPMDToSHMEMFamily(ModuleOp module, const SHMEMTarget &target,
                                     StructureLowering structure);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDTOSHMEMFAMILY_H
