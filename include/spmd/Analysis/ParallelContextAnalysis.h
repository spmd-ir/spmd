/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- ParallelContextAnalysis.h - Parallel nesting context ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Dense forward analysis that tracks the stack of enclosing spmd::ParallelOp
// operations at every program point.  This is shared infrastructure consumed
// by both UniformityValueAnalysis (so it knows which parallel levels an SSA
// value participates in) and ExecutionContextAnalysis (to resolve the correct
// parallel level when issuing level-aware uniformity queries).
//
// The stack is outermost-first; every entry is a live spmd::ParallelOp*.
//
// Join at control-flow merge points:
//   In well-formed SPMD IR a parallel region cannot be entered on one CFG
//   path and skipped on another, so all paths reaching a merge point carry
//   the same stack.  Join is therefore the identity for equal stacks and
//   fires an assertion otherwise.
//
// Interprocedural handling:
//   visitCallControlFlowTransfer propagates the caller's stack into the
//   callee, so functions invoked from within a parallel body correctly
//   inherit that body's parallel context — something a plain parent-op walk
//   cannot do.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPMD_ANALYSIS_PARALLELCONTEXTANALYSIS_H
#define MLIR_DIALECT_SPMD_ANALYSIS_PARALLELCONTEXTANALYSIS_H

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// ParallelContextState — dense, per program point
//
// Carries the ordered stack of enclosing spmd::ParallelOp pointers,
// outermost first.  An uninitialized state (isInitialized() == false) means
// the analysis has not yet propagated to this program point; downstream
// analyses should defer processing until it becomes initialized.
//===----------------------------------------------------------------------===//

class ParallelContextState : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &other) override;

  bool isInitialized() const { return initialized; }

  /// The stack of enclosing ParallelOps, outermost first.
  ArrayRef<Operation *> getStack() const { return parallelStack; }

  /// Returns the innermost enclosing ParallelOp, or nullptr if at the top
  /// level (not inside any parallel region).
  Operation *innermostParallelOp() const {
    return parallelStack.empty() ? nullptr : parallelStack.back();
  }

  /// Overwrite this state with the given stack.
  /// Returns Change iff the state actually changed.
  ChangeResult initializeTo(ArrayRef<Operation *> stack);

  void print(raw_ostream &os) const override;

private:
  SmallVector<Operation *, 4> parallelStack;
  bool initialized = false;
};

//===----------------------------------------------------------------------===//
// ParallelContextAnalysis — dense forward
//
// Transfer rules:
//   spmd::ParallelOp — propagates stack ++ [thisOp] into the body region;
//                      propagates the unchanged pre-op stack past the op
//                      itself (all executors reconverge at the join point).
//   CallOpInterface  — propagates the caller's stack into the callee on
//                      EnterCallee and back on ExitCallee / ExternalCallee.
//   All other ops   — identity (stack is unchanged).
//===----------------------------------------------------------------------===//

class ParallelContextAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<ParallelContextState> {
public:
  using Base = dataflow::DenseForwardDataFlowAnalysis<ParallelContextState>;
  using Base::Base;

  /// Entry state: empty stack (not yet inside any parallel region).
  void setToEntryState(ParallelContextState *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               const ParallelContextState &before,
                               ParallelContextState *after) override;

  /// Propagate the caller's parallel-context stack through call edges so
  /// that callees inherit the enclosing parallel context correctly.
  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const ParallelContextState &before,
                                    ParallelContextState *after) override;
};

} // namespace mlir::spmd

#endif // MLIR_DIALECT_SPMD_ANALYSIS_PARALLELCONTEXTANALYSIS_H