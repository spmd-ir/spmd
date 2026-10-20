/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef SPMD_ANALYSIS_HANDLEORIGINSANALYSIS_H
#define SPMD_ANALYSIS_HANDLEORIGINSANALYSIS_H

#include "spmd/Analysis/HandleFacts.h"

#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "llvm/ADT/ArrayRef.h"

namespace mlir::spmd {

/// Sparse forward analysis that tracks which concrete handle values each SSA
/// handle may originate from.
class HandleOriginsAnalysis
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

#endif // SPMD_ANALYSIS_HANDLEORIGINSANALYSIS_H