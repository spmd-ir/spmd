// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDPasses.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "spmd-reconcile-unrealized-cast"

#define PATTERN "main"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir::spmd {
// Define Base Class and Default Constructor for main Class
#define GEN_PASS_DEF_SPMDRECONCILEUNREALIZEDCAST
#include "spmd/SPMDPasses.h.inc"

namespace {

struct HoistUnrealizedCastsFromSCFIf : public OpRewritePattern<scf::IfOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::IfOp ifOp,
                                PatternRewriter &rewriter) const override {
    LLVM_DEBUG(DBGS() << "Trying to match scf.if at " << *ifOp << "\n");

    if (ifOp.getNumResults() == 0)
      return mlir::failure();

    LLVM_DEBUG(DBGS() << "Matching scf.if at " << *ifOp << "\n");
    auto thenYield =
        llvm::dyn_cast<mlir::scf::YieldOp>(ifOp.thenBlock()->getTerminator());
    auto elseYield =
        llvm::dyn_cast<mlir::scf::YieldOp>(ifOp.elseBlock()->getTerminator());

    if (!thenYield || !elseYield)
      return mlir::failure();

    unsigned numResults = ifOp.getNumResults();

    SmallVector<Type> newResultTypes;
    SmallVector<bool> shouldHoist(numResults, false);
    SmallVector<Type> srcTypes(numResults);
    SmallVector<Type> dstTypes(numResults);

    bool anyMatch = false;

    // --- Check each result position ---
    for (unsigned i = 0; i < numResults; ++i) {
      Value thenVal = thenYield.getOperand(i);
      Value elseVal = elseYield.getOperand(i);

      auto thenCast = thenVal.getDefiningOp<mlir::UnrealizedConversionCastOp>();
      auto elseCast = elseVal.getDefiningOp<mlir::UnrealizedConversionCastOp>();

      if (!thenCast || !elseCast || thenCast.getNumOperands() != 1 ||
          elseCast.getNumOperands() != 1)
        continue;

      Type srcType = thenCast.getOperand(0).getType();
      Type dstType = thenCast.getResult(0).getType();

      if (elseCast.getOperand(0).getType() != srcType ||
          elseCast.getResult(0).getType() != dstType)
        continue;

      LLVM_DEBUG(
          DBGS() << "Found unrealized_conversion_cast pair at result index "
                 << i << ": "
                 << "srcType=" << srcType << ", dstType=" << dstType << "\n");
      shouldHoist[i] = true;
      srcTypes[i] = srcType;
      dstTypes[i] = dstType;
      anyMatch = true;
    }

    if (!anyMatch)
      return mlir::failure();

    // --- Build new result type list ---
    for (unsigned i = 0; i < numResults; ++i) {
      if (shouldHoist[i])
        newResultTypes.push_back(srcTypes[i]);
      else
        newResultTypes.push_back(ifOp.getResultTypes()[i]);
    }

    rewriter.setInsertionPoint(ifOp);

    auto newIf = rewriter.create<scf::IfOp>(ifOp.getLoc(), newResultTypes,
                                            ifOp.getCondition(),
                                            /*withElseRegion=*/true);

    // Erase auto-created empty blocks
    newIf.getThenRegion().front().erase();
    newIf.getElseRegion().front().erase();

    // Move regions
    newIf.getThenRegion().takeBody(ifOp.getThenRegion());
    newIf.getElseRegion().takeBody(ifOp.getElseRegion());

    // --- Fix terminators ---
    auto fixYield = [&](Block &block, scf::YieldOp yieldOp) {
      rewriter.setInsertionPoint(yieldOp);

      SmallVector<Value> newOperands;

      for (unsigned i = 0; i < numResults; ++i) {
        if (!shouldHoist[i]) {
          newOperands.push_back(yieldOp.getOperand(i));
          continue;
        }

        auto castOp = yieldOp.getOperand(i)
                          .getDefiningOp<mlir::UnrealizedConversionCastOp>();

        newOperands.push_back(castOp.getOperand(0));
      }

      rewriter.replaceOpWithNewOp<mlir::scf::YieldOp>(yieldOp, newOperands);
    };

    fixYield(newIf.getThenRegion().front(),
             cast<scf::YieldOp>(newIf.thenBlock()->getTerminator()));

    fixYield(newIf.getElseRegion().front(),
             cast<scf::YieldOp>(newIf.elseBlock()->getTerminator()));

    // --- Insert hoisted casts ---
    rewriter.setInsertionPointAfter(newIf);

    SmallVector<Value> finalResults;

    for (unsigned i = 0; i < numResults; ++i) {
      Value res = newIf.getResult(i);

      if (!shouldHoist[i]) {
        finalResults.push_back(res);
        continue;
      }

      auto cast = rewriter.create<mlir::UnrealizedConversionCastOp>(
          ifOp.getLoc(), dstTypes[i], res);

      finalResults.push_back(cast.getResult(0));
    }

    rewriter.replaceOp(ifOp, finalResults);

    return success();
  }
};

static bool isTargetHandleType(Type ty, Type targetTy) {
  return ty == targetTy;
}

static bool isI32ToTargetHandleCast(Operation *op, Type targetTy) {
  auto cast = dyn_cast<UnrealizedConversionCastOp>(op);
  if (!cast)
    return false;
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return false;
  if (!cast.getOperand(0).getType().isInteger(32))
    return false;
  return isTargetHandleType(cast.getResult(0).getType(), targetTy);
}

static UnrealizedConversionCastOp getTargetHandleToI32Cast(Value v,
                                                           Type targetTy) {
  auto cast = v.getDefiningOp<UnrealizedConversionCastOp>();
  if (!cast)
    return {};
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return {};
  if (!isTargetHandleType(cast.getOperand(0).getType(), targetTy))
    return {};
  if (!cast.getResult(0).getType().isInteger(32))
    return {};
  return cast;
}

template <typename HandleTy>
struct RetypeI32AllocaToHandle : public OpRewritePattern<memref::AllocaOp> {
  using OpRewritePattern<memref::AllocaOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocaOp allocaOp,
                                PatternRewriter &rewriter) const override {
    auto oldTy = allocaOp.getType();

    if (oldTy.getRank() != 1 || oldTy.getShape()[0] != 1)
      return failure();

    if (!oldTy.getElementType().isInteger(32))
      return failure();

    Type targetHandleTy = HandleTy::get(allocaOp.getContext());

    SmallVector<memref::LoadOp> loads;
    SmallVector<memref::StoreOp> stores;

    for (Operation *user : allocaOp->getUsers()) {
      if (auto load = dyn_cast<memref::LoadOp>(user)) {
        loads.push_back(load);

        // Optional extra safety: only index 0
        if (load.getIndices().size() != 1)
          return failure();

        for (Operation *loadUser : load.getResult().getUsers()) {
          if (!isI32ToTargetHandleCast(loadUser, targetHandleTy))
            return failure();
        }
        continue;
      }

      if (auto store = dyn_cast<memref::StoreOp>(user)) {
        stores.push_back(store);

        // Optional extra safety: only index 0
        if (store.getIndices().size() != 1)
          return failure();

        if (!getTargetHandleToI32Cast(store.getValueToStore(), targetHandleTy))
          return failure();

        continue;
      }

      return failure();
    }

    auto newTy = MemRefType::get(oldTy.getShape(), targetHandleTy,
                                 oldTy.getLayout(), oldTy.getMemorySpace());

    rewriter.setInsertionPoint(allocaOp);
    auto newAlloca = rewriter.create<memref::AllocaOp>(
        allocaOp.getLoc(), newTy, allocaOp.getDynamicSizes(),
        allocaOp.getSymbolOperands(), allocaOp.getAlignmentAttr());

    for (memref::LoadOp load : loads) {
      rewriter.setInsertionPoint(load);
      auto newLoad = rewriter.create<memref::LoadOp>(
          load.getLoc(), newAlloca.getResult(), load.getIndices());

      SmallVector<Operation *> castUsers(load.getResult().getUsers().begin(),
                                         load.getResult().getUsers().end());
      for (Operation *castUser : castUsers) {
        auto castOp = mlir::cast<UnrealizedConversionCastOp>(castUser);
        rewriter.replaceOp(castOp, newLoad.getResult());
      }

      rewriter.eraseOp(load);
    }

    for (memref::StoreOp store : stores) {
      auto cast =
          getTargetHandleToI32Cast(store.getValueToStore(), targetHandleTy);
      if (!cast)
        return failure();

      Value handleVal = cast.getOperand(0);

      rewriter.setInsertionPoint(store);
      rewriter.create<memref::StoreOp>(
          store.getLoc(), handleVal, newAlloca.getResult(), store.getIndices());

      rewriter.eraseOp(store);
      if (cast->use_empty())
        rewriter.eraseOp(cast);
    }

    if (!allocaOp->use_empty())
      return failure();

    rewriter.eraseOp(allocaOp);
    return success();
  }
};

struct SPMDReconcileUnrealizedCast
    : impl::SPMDReconcileUnrealizedCastBase<SPMDReconcileUnrealizedCast> {
  using SPMDReconcileUnrealizedCastBase::SPMDReconcileUnrealizedCastBase;
  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    OpBuilder builder(moduleOp.getContext());
    RewritePatternSet patterns(&getContext());
    // GreedyRewriteConfig config;
    // patterns.insert<HoistUnrealizedCastsFromSCFIf,
    //          RetypeI32AllocaToHandle<spmd::WinType>,
    //          RetypeI32AllocaToHandle<spmd::ReqType>>(&getContext());
    patterns.insert<SwitchOpUnrealizedCasted
                    //  ReplaceSPMDOpOnArithSelectWithSCFIf
                    >(&getContext());
    (void)applyPatternsGreedily(moduleOp, std::move(patterns) /*, config*/);
  }
};

} // namespace
} // namespace mlir::spmd
