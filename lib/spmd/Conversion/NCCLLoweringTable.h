// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- NCCLLoweringTable.h - The NCCL lowering, off the raiser --*- C++ -*-===//
//
// What LoweringTable's inversion of the CACTS map and the USIS API description
// cannot say about NCCL: how the ABI passes each argument (NCCLArgKind), and
// which of the two broadcasts to pick.  -lower-spmd-to-nccl walks the derived
// slots.
//
// NCCL's calls are the regular part of its lowering: a handle, a buffer, a
// count, an enum, a stream, in the order the prototype has them.  What is not
// regular -- that the world communicator has to be bootstrapped, and that a
// group is completed by ncclGroupEnd rather than by a request -- is in
// LowerSPMDToNCCL.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_NCCLLOWERINGTABLE_H
#define SPMD_CONVERSION_NCCLLOWERINGTABLE_H

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

/// How NCCL passes one argument of a call: the value of ArgSlot::abiKind for a
/// call of the NCCL table.
enum class NCCLArgKind {
  /// The operation's operand at ArgSlot::operand, as the lowering converted
  /// it.  A communicator and a stream are pointers, a datatype and a reduction
  /// enums the dialect's constants lower to, and a count a size_t the dialect
  /// already holds as an i64, so nothing is rewritten on the way in.
  In,
  /// An out-parameter holding a communicator, which is a pointer.
  OutComm,
  /// An out-parameter holding an int: a rank, a size, a device.
  OutInt,
  /// An out-parameter holding an address, which ncclMemAlloc allocates.
  OutPtr,
  /// A null pointer, for an ncclConfig_t no operand feeds.
  NullPointer,
};

/// Everything the NCCL lowering adds to the inversion.
const TableSpec &getNCCLTableSpec();

/// Every call of the NCCL table, in a stable order.
ArrayRef<CallDescription> getNCCLLoweringTable();

/// The NCCL call \p op lowers to, or nullptr; see selectCall.
const CallDescription *selectNCCLCall(Operation *op, OpBuilder &builder,
                                      SmallVectorImpl<std::string> &reasons);

/// Prints the derived table, one line per call, for -test-nccl-lowering-table.
void printNCCLLoweringTable(raw_ostream &os, OpBuilder &builder);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_NCCLLOWERINGTABLE_H
