// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CUDALoweringTable.h - The CUDA lowering, off the raiser --*- C++ -*-===//
//
// What LoweringTable's inversion of the CACTS map and the USIS API description
// cannot say about CUDA, which is almost nothing: every argument of every call
// the raiser maps is an operand the dialect already holds in the C type the
// prototype wants, so the ABI has one kind (CUDAArgKind::In) and no rule to
// pick between callees.  -lower-spmd-to-cuda walks the derived slots.
//
// The table has the two calls -convert-apiCalls-to-spmd raises
// (lib/spmd/passes/CACTS/cudaSpmdEntries.cpp): cudaMemcpy and cudaMemset.
// CUDA's runtime API has some 330 routines, and the others are left as the
// llvm.call the frontend produced -- nothing raises them, so nothing has to
// lower them.  A call added to the raiser's map appears here on its own; only
// an argument classifyCUDAArg has no rule for would need an edit.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_CUDALOWERINGTABLE_H
#define SPMD_CONVERSION_CUDALOWERINGTABLE_H

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

/// How CUDA passes one argument of a call: the value of ArgSlot::abiKind for a
/// call of the CUDA table.  There is one kind because both calls take only
/// arguments the operation carries: a buffer is a pointer, a count the size_t
/// the dialect already holds as an i64, and a direction and a fill byte the
/// ints the program computed.  Nothing is an out-parameter -- cudaMemcpy and
/// cudaMemset report through their return value alone.
enum class CUDAArgKind {
  /// The operation's operand at ArgSlot::operand, as the lowering converted
  /// it.
  In,
};

/// Everything the CUDA lowering adds to the inversion.
const TableSpec &getCUDATableSpec();

/// Every call of the CUDA table, in a stable order.
ArrayRef<CallDescription> getCUDALoweringTable();

/// The CUDA call \p op lowers to, or nullptr; see selectCall.
const CallDescription *selectCUDACall(Operation *op, OpBuilder &builder,
                                      SmallVectorImpl<std::string> &reasons);

/// Prints the derived table, one line per call, for -test-cuda-lowering-table.
void printCUDALoweringTable(raw_ostream &os, OpBuilder &builder);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_CUDALOWERINGTABLE_H
