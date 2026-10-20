// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- EliminateRedundantComm.cpp - SPMD Pass -------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Redundant collective elimination (docs/optimizations.md, "Available
// collectives").  A collective whose
// result is already sitting in its destination buffer - same operation, same
// communicator, same buffers, same shape, with nothing written to either buffer
// in between - is removed.
//
// Where -eliminate-dead-comm reasons about one buffer at a time and needs no
// dataflow, this one needs a real must-analysis: "already performed on every
// path, on every rank".  AvailableCollectives.h carries that analysis and the
// argument behind it.
//
// Like the dead-comm pass, every removal is textual and whole-op, so all ranks
// lose the same operation and per-rank counts stay aligned by construction.
// Unlike it, the *fact* being relied on is one about all ranks, which is why
// GEN - and only GEN - is gated on uniform execution.
//
//===----------------------------------------------------------------------===//

#include "spmd/Analysis/AvailableCollectives.h"
#include "spmd/Analysis/CommCompletions.h"
#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/Analysis/ParallelContextAnalysis.h"
#include "spmd/Analysis/UniformityAnalysis.h"
#include "spmd/SPMDPasses.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::spmd {

#define GEN_PASS_DEF_ELIMINATEREDUNDANTCOMM
#include "spmd/SPMDPasses.h.inc"

namespace {

struct EliminateRedundantComm
    : impl::EliminateRedundantCommBase<EliminateRedundantComm> {
  using EliminateRedundantCommBase::EliminateRedundantCommBase;

  void runOnOperation() override {
    // Removing one collective can expose another (a b c with b redundant given
    // a, and c redundant given b only once b is gone), and the analysis is not
    // incremental, so re-run it until nothing changes.
    for (unsigned iteration = 0; iteration < maxIterations; ++iteration)
      if (!eliminateOnce())
        break;
  }

  bool eliminateOnce() {
    ModuleOp moduleOp = getOperation();
    DataLayout dl(moduleOp);

    // Which ranks execute an op, and whether they agree on a value: the
    // analyses -check-collectives solves.
    DataFlowSolver aliasSolver;
    aliasSolver.load<enzyme::AliasAnalysis>(&getContext());
    aliasSolver.load<enzyme::PointsToPointerAnalysis>();
    aliasSolver.load<dataflow::DeadCodeAnalysis>();
    aliasSolver.load<dataflow::SparseConstantPropagation>();
    DataFlowSolver uniformity;
    uniformity.load<dataflow::DeadCodeAnalysis>();
    uniformity.load<dataflow::SparseConstantPropagation>();
    uniformity.load<ParallelContextAnalysis>();
    uniformity.load<UniformityValueAnalysis>(aliasSolver);
    uniformity.load<UniformityMemoryAnalysis>(aliasSolver);
    uniformity.load<ExecutionContextAnalysis>();
    if (failed(aliasSolver.initializeAndRun(moduleOp)) ||
        failed(uniformity.initializeAndRun(moduleOp))) {
      moduleOp.emitError("-eliminate-redundant-comm: the uniformity analyses "
                         "failed");
      signalPassFailure();
      return false;
    }

    AliasAnalysis aliasAnalysis(moduleOp);
    DataFlowSolver solver;
    solver.load<dataflow::DeadCodeAnalysis>();
    solver.load<dataflow::SparseConstantPropagation>();
    solver.load<AvailableCollectivesAnalysis>(aliasAnalysis, uniformity, dl);
    if (failed(solver.initializeAndRun(moduleOp))) {
      moduleOp.emitError("-eliminate-redundant-comm: the availability "
                         "analysis failed");
      signalPassFailure();
      return false;
    }

    SmallVector<Operation *> redundant;
    moduleOp.walk([&](Operation *op) {
      std::optional<CommKey> key = getCommKey(op, dl);
      if (!key)
        return;
      // The error code is the only result a blocking collective has; if
      // anything inspects it, the call is observable.
      if (!llvm::all_of(op->getResults(),
                        [](Value result) { return result.use_empty(); }))
        return;

      const AvailableCollectives *state =
          solver.lookupState<AvailableCollectives>(
              solver.getProgramPointBefore(op));
      // No state, or a state nothing has reached: no information, no removal.
      if (!state || state->isBottom())
        return;
      if (state->contains(*key))
        redundant.push_back(op);
    });

    // Blocking collectives have no completions, but go through
    // collectRemovals anyway: nothing in these passes may erase a
    // communication op directly.
    SetVector<Operation *> removals = collectRemovals(redundant, dl);
    unsigned removed = redundant.size();
    IRRewriter rewriter(&getContext());
    eraseOps(rewriter, std::move(removals));

    numRedundant += removed;
    return removed != 0;
  }
};

} // namespace

} // namespace mlir::spmd
