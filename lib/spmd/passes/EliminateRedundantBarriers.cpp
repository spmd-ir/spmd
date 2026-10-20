// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- EliminateRedundantBarriers.cpp - SPMD Pass ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass removes redundant adjacent spmd.barrier operations.
//
// It is a pattern applied in one walk, not a canonicalization pattern of
// spmd.barrier: spmd-verify runs -canonicalize before the checkers, which have
// to see every barrier (see "Never before the checkers" in
// docs/optimizations.md).
//
//===----------------------------------------------------------------------===//

#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/WalkPatternRewriteDriver.h"

namespace mlir::spmd {

#define GEN_PASS_DEF_ELIMINATEREDUNDANTBARRIERS
#include "spmd/SPMDPasses.h.inc"

namespace {

/// A blocking barrier whose error nobody reads.
bool isFoldable(BarrierOp barrier) {
  return barrier.getIsBlocking() && barrier.getError().use_empty();
}

/// Erases a barrier that follows an identical one: the same communicator and
/// the same attributes, so the same model and matching semantics.
struct FoldRepeatedBarrier : public OpRewritePattern<BarrierOp> {
  FoldRepeatedBarrier(MLIRContext *context, Pass::Statistic &numFolded)
      : OpRewritePattern(context), numFolded(numFolded) {}

  LogicalResult matchAndRewrite(BarrierOp barrier,
                                PatternRewriter &rewriter) const override {
    auto previous = dyn_cast_or_null<BarrierOp>(barrier->getPrevNode());
    if (!previous || !isFoldable(previous) || !isFoldable(barrier))
      return failure();
    if (previous.getComm() != barrier.getComm() ||
        previous->getAttrDictionary() != barrier->getAttrDictionary())
      return failure();
    rewriter.eraseOp(barrier);
    ++numFolded;
    return success();
  }

  Pass::Statistic &numFolded;
};

struct EliminateRedundantBarriers
    : impl::EliminateRedundantBarriersBase<EliminateRedundantBarriers> {
  using EliminateRedundantBarriersBase::EliminateRedundantBarriersBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<FoldRepeatedBarrier>(&getContext(), numFolded);
    walkAndApplyPatterns(getOperation(), std::move(patterns));
  }
};

} // namespace

} // namespace mlir::spmd
