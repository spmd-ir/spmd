/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef SPMD_ANALYSIS_PASSUTILS_H
#define SPMD_ANALYSIS_PASSUTILS_H

#include "spmd/SPMDTypes.h"

#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <string>

namespace mlir::spmd {

/// Return the integer "usedModel" attribute if present on \p op.
inline std::optional<int> getUsedModel(Operation *op) {
  if (!op)
    return std::nullopt;
  auto attr = op->getAttrOfType<IntegerAttr>("usedModel");
  if (!attr)
    return std::nullopt;
  return static_cast<int>(attr.getInt());
}

/// Print \p value as an SSA operand string.
inline std::string valAsString(Value value) {
  std::string res;
  llvm::raw_string_ostream os(res);
  value.printAsOperand(os, OpPrintingFlags());
  return res;
}

/// Return true if \p type is a handle type (ReqType or WinType).
inline bool isHandleType(Type type) { return isa<ReqType, WinType>(type); }

inline bool isSpmdOp(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && dialect->getNamespace() == "spmd";
}

/// Resolve through CallSiteLoc / FusedLoc / NameLoc wrappers to find a
/// displayable FileLineColLoc, falling back to \p loc itself.
Location findLocToShowWrapper(Location loc);

/// Return the condition operand(s) for known branch-like operations.
SmallVector<Value> getConditionOperands(Operation *op);

/// Collect all enclosing RegionBranchOpInterface ancestors of \p op.
SmallVector<Operation *> getEnclosingRegionBranches(Operation *op);

struct SwitchOpUnrealizedCasted : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern<arith::SelectOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp selectOp,
                                PatternRewriter &rewriter) const final {
    OpBuilder builder(selectOp);
    Value trueValue = selectOp.getTrueValue();
    Value falseValue = selectOp.getFalseValue();

    auto trueUCastOp = trueValue.getDefiningOp<UnrealizedConversionCastOp>();
    if (!trueUCastOp || trueUCastOp->getNumOperands() != 1) {
      return failure();
    }
    trueValue = trueUCastOp.getOperand(0);

    auto falseUCastOp = falseValue.getDefiningOp<UnrealizedConversionCastOp>();
    if (!falseUCastOp || falseUCastOp->getNumOperands() != 1) {
      return failure();
    }
    falseValue = falseUCastOp.getOperand(0);

    SmallVector<Operation *> users(selectOp.getResult().getUsers());

    bool allUsersAreCasts = llvm::all_of(
        users, [](Operation *u) { return isa<UnrealizedConversionCastOp>(u); });

    if (!allUsersAreCasts)
      return failure();

    auto newSelectOp =
        arith::SelectOp::create(rewriter, selectOp.getLoc(),
                                selectOp.getCondition(), trueValue, falseValue);

    for (auto *u : users) {
      auto castOp = cast<UnrealizedConversionCastOp>(u);
      rewriter.replaceOp(castOp, newSelectOp.getResult());
    }

    rewriter.eraseOp(selectOp);
    return success();
  }
};

struct ReplaceSPMDOpOnArithSelectWithSCFIf : public RewritePattern {
  ReplaceSPMDOpOnArithSelectWithSCFIf(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), /*benefit=*/1, context) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getDialect() != op->getContext()->getLoadedDialect("spmd"))
      return failure();

    // Collect ALL arith.select operands upfront — handle everything in one shot
    SmallVector<std::pair<unsigned, arith::SelectOp>> selects;
    for (auto [idx, operand] : llvm::enumerate(op->getOperands()))
      if (auto sel = operand.getDefiningOp<arith::SelectOp>())
        selects.emplace_back(idx, sel);

    if (selects.empty())
      return failure();

    rewriter.setInsertionPoint(op);
    SmallVector<Value> baseOperands(op->getOperands());
    scf::IfOp topIf =
        buildNestedIf(rewriter, op, op->getLoc(), selects, 0, baseOperands);

    rewriter.replaceOp(op, topIf.getResults());
    return success();
  }

private:
  // Recursively build one scf.if per select, nesting them.
  // On entry:  insertion point is where this if should be created.
  // On return: insertion point is just after the returned ifOp in its block.
  scf::IfOp
  buildNestedIf(PatternRewriter &rewriter, Operation *op, Location loc,
                ArrayRef<std::pair<unsigned, arith::SelectOp>> selects,
                unsigned depth, SmallVector<Value> operands) const {
    auto [idx, selectOp] = selects[depth];

    auto ifOp = scf::IfOp::create(rewriter, loc, op->getResultTypes(),
                                  selectOp.getCondition(), /*withElse=*/true);

    // ---- then branch ----
    {
      SmallVector<Value> thenOps = operands;
      thenOps[idx] = selectOp.getTrueValue();
      rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
      populateBranch(rewriter, op, loc, selects, depth + 1, thenOps);
    }

    // ---- else branch ----
    {
      SmallVector<Value> elseOps = operands;
      elseOps[idx] = selectOp.getFalseValue();
      rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
      populateBranch(rewriter, op, loc, selects, depth + 1, elseOps);
    }

    // Restore insertion point to after ifOp so the caller can place yield
    rewriter.setInsertionPointAfter(ifOp);
    return ifOp;
  }

  // Fill one branch block. Insertion point is at the block's start on entry.
  void populateBranch(PatternRewriter &rewriter, Operation *op, Location loc,
                      ArrayRef<std::pair<unsigned, arith::SelectOp>> selects,
                      unsigned depth, SmallVector<Value> operands) const {
    if (depth < selects.size()) {
      // More selects remain — nest another if
      scf::IfOp inner =
          buildNestedIf(rewriter, op, loc, selects, depth, operands);
      // buildNestedIf left insertion point after `inner` — yield its results
      scf::YieldOp::create(rewriter, loc, inner.getResults());
    } else {
      // Leaf: clone with all selects resolved
      IRMapping mapping;
      for (unsigned i = 0; i < op->getNumOperands(); ++i)
        if (operands[i] != op->getOperand(i))
          mapping.map(op->getOperand(i), operands[i]);

      Operation *clone = rewriter.clone(*op, mapping);
      rewriter.setInsertionPointAfter(clone);
      scf::YieldOp::create(rewriter, loc, clone->getResults());
    }
  }
};

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_PASSUTILS_H