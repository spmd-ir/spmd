// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- MPILoweringTable.h - The MPI lowering, read off the raiser -*- C++ -*-=//
//
// What LoweringTable's inversion of the CACTS map and the USIS API description
// cannot say about MPI: how MPICH passes each argument (MPIArgKind), which
// callee to pick when several raise to the same operation, and how a
// diagnostic reads.  -lower-spmd-to-mpi walks the derived slots and feeds
// MPICallArgs.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_MPILOWERINGTABLE_H
#define SPMD_CONVERSION_MPILOWERINGTABLE_H

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

/// How MPICH passes one argument of an MPI call: the value of ArgSlot::abiKind
/// for a call of the MPI table.
enum class MPIArgKind {
  /// The operation's operand at ArgSlot::operand, as the lowering converted
  /// it.
  In,
  /// The same, as an element count: the dialect holds an MPI_Count where the
  /// classic call takes an int, so the call may need its large-count form.
  Count,
  /// The same, passed by address, because MPI writes the handle back.
  ByAddress,
  /// The same, an array of MPI_Statuses -- or MPI_STATUSES_IGNORE when the
  /// operation leaves that operand out.
  StatusArray,
  /// An out-parameter holding an int: a rank, a size, a new handle, an index.
  OutInt,
  /// An out-parameter holding a pointer, such as the address MPI allocates.
  OutPtr,
  /// An out-parameter holding an int flag, read back as an i1 result.
  OutFlag,
  /// An MPI_Status out-parameter, or MPI_STATUS_IGNORE when the operation has
  /// no status result.
  OutStatus,
  /// MPI_INFO_NULL, for an MPI_Info no operand feeds.
  InfoNull,
  /// A null pointer, for the argc and argv of MPI_Init.
  NullPointer,
};

/// Everything the MPI lowering adds to the inversion.
const TableSpec &getMPITableSpec();

/// Every call of the MPI table, in a stable order.
ArrayRef<CallDescription> getMPILoweringTable();

/// The MPI call \p op lowers to, or nullptr; see selectCall.
const CallDescription *selectMPICall(Operation *op, OpBuilder &builder,
                                     SmallVectorImpl<std::string> &reasons);

/// Prints the derived table, one line per call, for -test-mpi-lowering-table.
void printMPILoweringTable(raw_ostream &os, OpBuilder &builder);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_MPILOWERINGTABLE_H
