// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- SHMEMLoweringTable.h - The OpenSHMEM lowerings, read off the raiser ===//
//
// What LoweringTable's inversion of the CACTS map and the USIS API description
// cannot say about the OpenSHMEM family -- OpenSHMEM itself and NVSHMEM, whose
// API is OpenSHMEM's with an nvshmem_ prefix: how the ABI passes each argument
// (SHMEMArgKind), which of the many spellings of one call to pick, and that
// "SUM" and "sum" are the same reduction.  -lower-spmd-to-shmem and
// -lower-spmd-to-nvshmem walk the derived slots and feed SHMEMCallArgs.
//
// The typed callee names are the reason this lowering is derived rather than
// written: both models put the element type and the reduction operation in the
// function name, and the CACTS map already says, for each of the ~1300 names,
// which datatype and which operation the raiser reads out of it.  Inverted,
// that is exactly the rule for picking the name back.
//
// The two models differ in the spellings they have -- NVSHMEM has half and
// bfloat16 and no long double or complex type -- and in their ABI, which the
// lowering rather than the table deals with (LowerSPMDToSHMEMFamily.h).
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_SHMEMLOWERINGTABLE_H
#define SPMD_CONVERSION_SHMEMLOWERINGTABLE_H

#include "LoweringTable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::spmd {

/// A model of the OpenSHMEM family, whose table this is.
enum class SHMEMFlavour { OpenSHMEM, NVSHMEM };

/// How a model of the family passes one argument of a call: the value of
/// ArgSlot::abiKind for a call of its table.
enum class SHMEMArgKind {
  /// The operation's operand at ArgSlot::operand, as the lowering converted
  /// it.  Every handle of either ABI is either a pointer or an integer the
  /// dialect already holds in the right width, so nothing is rewritten on the
  /// way in.
  In,
  /// An out-parameter holding a team handle: a pointer in OpenSHMEM, an
  /// int32_t in NVSHMEM.
  OutTeam,
  /// An out-parameter holding an int, such as the granted thread level.
  OutInt,
  /// A null pointer, for a shmem_team_config_t no operand feeds.
  NullPointer,
  /// A zero, for an integer parameter no operand feeds, such as a config mask
  /// that selects nothing.
  ZeroI32,
  ZeroI64,
};

/// The routines the lowering emits from patterns of its own rather than from
/// the table, without the model's prefix: the operations they lower carry a
/// window or a request the call has no parameter for, and the routines are
/// process-wide anyway.  shmem_barrier_all is spmd.fence, shmem_quiet
/// spmd.flushAll and every completion, shmem_fence spmd.fenceOrdering.
inline constexpr StringLiteral kBarrierAll = "barrier_all";
inline constexpr StringLiteral kQuiet = "quiet";
inline constexpr StringLiteral kFence = "fence";

/// Everything the lowering of \p flavour adds to the inversion.
const TableSpec &getSHMEMTableSpec(SHMEMFlavour flavour);

/// Every call of \p flavour's table, in a stable order.
ArrayRef<CallDescription> getSHMEMLoweringTable(SHMEMFlavour flavour);

/// The call of \p flavour \p op lowers to, or nullptr; see selectCall.
const CallDescription *selectSHMEMCall(SHMEMFlavour flavour, Operation *op,
                                       OpBuilder &builder,
                                       SmallVectorImpl<std::string> &reasons);

/// Prints the derived table, one line per call, for
/// -test-shmem-lowering-table and -test-nvshmem-lowering-table.
void printSHMEMLoweringTable(SHMEMFlavour flavour, raw_ostream &os,
                             OpBuilder &builder, bool canonicalOnly = false,
                             StringRef filter = "");

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_SHMEMLOWERINGTABLE_H
