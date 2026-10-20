// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDPasses.h"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::spmd {
// Define Base Class and Default Constructor for main Class
#define GEN_PASS_DEF_CONVERTARITHSELECTTOIF
#include "spmd/SPMDPasses.h.inc"

namespace {

struct ConvertArithSelectToIfPattern
    : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const override {
    // Convert ONE user per application.
    //
    // The previous version collected every user, created the scf.if at the
    // select's own position and cloned all of them into it. That is not
    // dominance-safe: a user may consume values defined *between* the select
    // and itself, and those definitions stay behind the newly created if — the
    // verifier then rejects the module with "operand does not dominate this
    // use". Anchoring the if at one user instead is always safe, because every
    // operand of an operation dominates that operation by construction. The
    // greedy driver re-applies the pattern until the select has no convertible
    // users left, so multi-user selects are still fully converted.
    Operation *user = nullptr;
    for (Operation *candidate : selectOp.getResult().getUsers()) {
      // A terminator cannot be sunk into a branch: its results (if any) belong
      // to the enclosing region's control flow.
      if (candidate->hasTrait<OpTrait::IsTerminator>())
        continue;
      user = candidate;
      break;
    }
    if (!user)
      return failure();

    Value condition = selectOp.getCondition();
    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();

    rewriter.setInsertionPoint(user);
    auto ifOp = scf::IfOp::create(rewriter, user->getLoc(),
                                  user->getResultTypes(), condition,
                                  /*withElseRegion=*/true);

    auto cloneInto = [&](Block *block, Value selectReplacement) {
      if (block->mightHaveTerminator())
        rewriter.eraseOp(block->getTerminator());
      rewriter.setInsertionPointToStart(block);
      IRMapping mapping;
      mapping.map(selectOp.getResult(), selectReplacement);
      Operation *cloned = rewriter.clone(*user, mapping);
      scf::YieldOp::create(rewriter, user->getLoc(), cloned->getResults());
    };

    cloneInto(ifOp.thenBlock(), trueValue);
    cloneInto(ifOp.elseBlock(), falseValue);

    rewriter.replaceOp(user, ifOp.getResults());
    return success();
  }
};

struct ConvertArithSelectToIf
    : impl::ConvertArithSelectToIfBase<ConvertArithSelectToIf> {
  using ConvertArithSelectToIfBase::ConvertArithSelectToIfBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    RewritePatternSet patterns(&getContext());
    // GreedyRewriteConfig config;
    patterns.insert<ConvertArithSelectToIfPattern>(&getContext());
    (void)applyPatternsGreedily(moduleOp, std::move(patterns) /*, config*/);
  }
};

} // namespace
} // namespace mlir::spmd
