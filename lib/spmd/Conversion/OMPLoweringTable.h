// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- OMPLoweringTable.h - The OpenMP queries, off the raiser --*- C++ -*-===//
//
// What LoweringTable's inversion of the CACTS map and the USIS API description
// cannot say about OpenMP's runtime routines.  Those the raiser maps --
// omp_get_thread_num and omp_get_num_threads -- ask the current team, which is
// the communicator the raiser injects, so they take no argument at all and
// return what the operation's result holds.
//
// The directives are not calls, and not in this table: -lower-spmd-to-omp
// builds the omp dialect operations -omp-to-spmd took them from.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_OMPLOWERINGTABLE_H
#define SPMD_CONVERSION_OMPLOWERINGTABLE_H

#include "LoweringTable.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::spmd {

/// Everything the OpenMP lowering adds to the inversion.
const TableSpec &getOMPTableSpec();

/// Every call of the OpenMP table, in a stable order.
ArrayRef<CallDescription> getOMPLoweringTable();

/// The OpenMP call \p op lowers to, or nullptr; see selectCall.
const CallDescription *selectOMPCall(Operation *op, OpBuilder &builder,
                                     SmallVectorImpl<std::string> &reasons);

/// Prints the derived table, one line per call, for -test-omp-lowering-table.
void printOMPLoweringTable(raw_ostream &os, OpBuilder &builder);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_OMPLOWERINGTABLE_H
