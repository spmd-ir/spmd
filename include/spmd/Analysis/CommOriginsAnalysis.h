/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef SPMD_ANALYSIS_COMMORIGINSANALYSIS_H
#define SPMD_ANALYSIS_COMMORIGINSANALYSIS_H

#include "spmd/Analysis/HandleFacts.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir::spmd {

/// Sparse forward analysis that tracks which concrete spmd.commWorld (or other
/// comm-creating) Values each comm-typed SSA value may originate from.
///
/// Propagates through unrealized_conversion_cast, enzymexla.pointer2memref,
/// and enzymexla.memref2pointer so that the indirect
///
///   %comm = spmd.commWorld
///   %ptr  = cast %comm : !spmd.comm to !llvm.ptr
///   %mr   = pointer2memref(%ptr)
///   %ld   = memref.load %mr[0]
///   %comm2 = cast %ld : !llvm.ptr to !spmd.comm
///
/// pattern resolves %comm2 back to the same origin atom as %comm.
class CommOriginsAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<HandleFactsLattice> {
public:
  using Base = dataflow::SparseForwardDataFlowAnalysis<HandleFactsLattice>;
  using Base::Base;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const HandleFactsLattice *> operands,
                               ArrayRef<HandleFactsLattice *> results) override;

  void
  visitExternalCall(CallOpInterface call,
                    ArrayRef<const HandleFactsLattice *> argumentLattices,
                    ArrayRef<HandleFactsLattice *> resultLattices) override;

protected:
  void setToEntryState(HandleFactsLattice *lattice) override;
};

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_COMMORIGINSANALYSIS_H