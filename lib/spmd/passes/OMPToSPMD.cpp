// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/RegionUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "omp-to-spmd"

namespace mlir::spmd {

#define GEN_PASS_DEF_OMPTOSPMD
#include "spmd/SPMDPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Classifies an omp.declare_reduction combiner region into a well-known
/// SPMD ReductionKind. Only matches the trivial single-op pattern.
/// Returns nullopt for user-defined combiners — $kind is left absent.
static std::optional<spmd::ReductionKind>
matchReductionKind(Region &reductionRegion) {
  if (reductionRegion.empty())
    return std::nullopt;
  Block &block = reductionRegion.front();
  if (block.getOperations().size() != 2)
    return std::nullopt; // not a single combine-op + yield

  Operation &op = block.front();
  if (isa<arith::AddIOp, arith::AddFOp>(op))
    return spmd::ReductionKind::Add;
  if (isa<arith::MulIOp, arith::MulFOp>(op))
    return spmd::ReductionKind::Mul;
  if (isa<arith::MinSIOp>(op))
    return spmd::ReductionKind::MinS;
  if (isa<arith::MinUIOp>(op))
    return spmd::ReductionKind::MinU;
  if (isa<arith::MinimumFOp>(op))
    return spmd::ReductionKind::MinF;
  if (isa<arith::MaxSIOp>(op))
    return spmd::ReductionKind::MaxS;
  if (isa<arith::MaxUIOp>(op))
    return spmd::ReductionKind::MaxU;
  if (isa<arith::MaximumFOp>(op))
    return spmd::ReductionKind::MaxF;
  if (isa<arith::AndIOp>(op))
    return spmd::ReductionKind::And;
  if (isa<arith::OrIOp>(op))
    return spmd::ReductionKind::Or;
  if (isa<arith::XOrIOp>(op))
    return spmd::ReductionKind::Xor;
  return std::nullopt;
}

/// Returns the communicator for an operation.
/// Inside an spmd.parallel: emits spmd.getParallelComm.
/// Otherwise: falls back to spmd.commWorld for top-level constructs.
static Value getCommForOp(Operation *op, OpBuilder &builder, int modelGroup) {
  Location loc = op->getLoc();
  if (op->getParentOfType<spmd::ParallelOp>())
    return builder.create<spmd::GetParallelCommOp>(loc).getComm();
  return builder.create<spmd::CommWorldOp>(loc, modelGroup).getCommWorld();
}

/// Emits a diagnostic warning and returns failure for unsupported OpenMP ops.
static LogicalResult unsupportedOp(Operation *op, PatternRewriter &rewriter) {
  op->emitWarning() << "[OMPToSPMD] unsupported OpenMP op '" << op->getName()
                    << "' — no conversion pattern implemented yet";
  return rewriter.notifyMatchFailure(op, "no conversion pattern implemented");
}

/// Collects private vars from omp.parallel, folding firstprivate into private.
/// Both are treated identically for data race analysis — each thread gets its
/// own copy and neither can race. The initialization distinction is dropped.
static SmallVector<Value> collectPrivateVars(omp::ParallelOp op) {
  return llvm::to_vector(op.getPrivateVars());
}

struct ConvertDeclareReductionOp
    : public OpConversionPattern<omp::DeclareReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(omp::DeclareReductionOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<spmd::ReductionKind> kind =
        matchReductionKind(op.getReductionRegion());

    spmd::ReductionKindAttr kindAttr;
    if (kind)
      kindAttr = spmd::ReductionKindAttr::get(rewriter.getContext(), *kind);

    auto spmdDecl = rewriter.create<spmd::DeclareReductionOp>(
        op.getLoc(), op.getSymNameAttr(), op.getTypeAttr(),
        op.getByrefElementTypeAttr(), kindAttr);

    rewriter.inlineRegionBefore(op.getAllocRegion(), spmdDecl.getAllocRegion(),
                                spmdDecl.getAllocRegion().end());
    rewriter.inlineRegionBefore(op.getInitializerRegion(),
                                spmdDecl.getInitializerRegion(),
                                spmdDecl.getInitializerRegion().end());
    rewriter.inlineRegionBefore(op.getReductionRegion(),
                                spmdDecl.getReductionRegion(),
                                spmdDecl.getReductionRegion().end());
    rewriter.inlineRegionBefore(op.getAtomicReductionRegion(),
                                spmdDecl.getAtomicReductionRegion(),
                                spmdDecl.getAtomicReductionRegion().end());
    rewriter.inlineRegionBefore(op.getCleanupRegion(),
                                spmdDecl.getCleanupRegion(),
                                spmdDecl.getCleanupRegion().end());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.terminator -> spmd.terminator
//===----------------------------------------------------------------------===//

struct ConvertTerminatorOp : public OpConversionPattern<omp::TerminatorOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(omp::TerminatorOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Operation *parent = op->getParentOp();
    bool directlyInsideParallel =
        isa_and_nonnull<omp::ParallelOp, spmd::ParallelOp>(parent);
    if (directlyInsideParallel) {
      rewriter.replaceOpWithNewOp<spmd::ParallelYieldOp>(op, Value{});
    } else {
      // Covers: SingleOp, CriticalOp, SectionsOp, TaskOp, TaskgroupOp
      // — all use spmd.terminator as their implicit region terminator.
      rewriter.replaceOpWithNewOp<spmd::TerminatorOp>(op);
    }
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.yield -> spmd.yield
//===----------------------------------------------------------------------===//

struct ConvertYieldOp : public OpConversionPattern<omp::YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(omp::YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<spmd::YieldOp>(op, adaptor.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.parallel -> spmd.parallel
//
// Scoping:
//   shared    -> inferred from SSA (values captured from above, not in
//                private_vars or reduction_vars)
//   private   -> $private_vars (firstprivate folded in, see collectPrivateVars)
//   reduction -> $reduction_vars + $reduction_syms
//===----------------------------------------------------------------------===//

struct ConvertParallelOp : public OpConversionPattern<omp::ParallelOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertParallelOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.getAllocateVars().empty()) {
      op.emitWarning("[OMPToSPMD] omp.parallel allocate clause "
                     "not yet supported — skipping");
      return failure();
    }

    // Give this parallel region its own communicator scope via commDup so
    // nested barriers and collectives operate on the correct team.
    rewriter.setInsertionPoint(op);
    Value worldComm = getCommForOp(op, rewriter, modelGroup);
    auto dupOp = rewriter.create<spmd::CommDupOp>(loc, worldComm, modelGroup);
    Value comm = dupOp.getNewComm();

    SmallVector<Value> privateVars = collectPrivateVars(op);
    SmallVector<Value> reductionVars = llvm::to_vector(op.getReductionVars());
    ArrayAttr reductionSyms = op.getReductionSymsAttr();

    Value numThreads;
    if (op.getNumThreadsDimsCount() > 0)
      numThreads = op.getNumThreads(0);

    auto spmdParallel = rewriter.create<spmd::ParallelOp>(
        loc,
        /*exit_code=*/Type{}, comm, numThreads, privateVars, reductionVars,
        reductionSyms,
        /*threadLevel=*/mlir::spmd::ThreadLevelAttr{}, // absent: OMP inherits
                                                       // constraint from
                                                       // enclosing
                                                       // process-level parallel
        /*isProcessLevel=*/false, (uint32_t)modelGroup);

    // Move the omp.parallel body into spmd.parallel. The conversion framework
    // recurses into the moved blocks and converts nested ops.
    rewriter.inlineRegionBefore(op.getRegion(), spmdParallel.getRegion(),
                                spmdParallel.getRegion().end());

    // Inject getParallelComm at the entry block so orphaned directives
    // (barrier, single, critical, wsloop) can resolve their communicator
    // without requiring signature changes on the enclosing function.
    Block &entry = spmdParallel.getRegion().front();
    rewriter.setInsertionPointToStart(&entry);
    rewriter.create<spmd::GetParallelCommOp>(loc);

    // OMP parallel always has an implicit join barrier at the region end.
    // nowait is not valid on omp.parallel — the barrier is unconditional.
    // Insert it before the (still-unconverted) omp.terminator so that
    // ConvertTerminatorOp can then replace that terminator with
    // spmd.parallel_yield cleanly.
    // it seems the barrier is explicit for omp.parallel
    // Operation *terminator = spmdParallel.getRegion().front().getTerminator();
    // rewriter.setInsertionPoint(terminator);
    // rewriter.create<spmd::BarrierOp>(
    //     loc,
    //     spmd::ErrorType::get(op->getContext()),
    //     comm,
    //     /*isBlocking=*/true,
    //     /*usedModel=*/(uint32_t)modelGroup,
    //     /*isUniqueMatching=*/rewriter.getBoolAttr(true));

    // Reduction operator recovery from DeclareReductionOp is deferred to a
    // follow-up pass. The symbols are preserved in $reduction_syms.
    // if (!reductionVars.empty())
    //   op.emitRemark("[OMPToSPMD] reduction clause: spmd.allreduce emission "
    //                  "deferred — recover operators from $reduction_syms in a
    //                  " "follow-up pass");

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.barrier -> spmd.barrier
//===----------------------------------------------------------------------===//

struct ConvertBarrierOp : public OpConversionPattern<omp::BarrierOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertBarrierOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::BarrierOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.setInsertionPoint(op);
    Value comm = getCommForOp(op, rewriter, modelGroup);
    rewriter.create<spmd::BarrierOp>(
        op->getLoc(), spmd::ErrorType::get(op->getContext()), comm,
        /*isBlocking=*/true,
        /*usedModel=*/(uint32_t)modelGroup,
        /*isUniqueMatching=*/rewriter.getBoolAttr(true));
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.wsloop -> spmd.wsloop
//===----------------------------------------------------------------------===//

struct ConvertWsloopOp : public OpConversionPattern<omp::WsloopOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertWsloopOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::WsloopOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.getLinearVars().empty()) {
      op.emitWarning("[OMPToSPMD] omp.wsloop linear clause not yet supported");
      return failure();
    }

    rewriter.setInsertionPoint(op);
    Value comm = getCommForOp(op, rewriter, modelGroup);

    MLIRContext *ctx = rewriter.getContext();
    using SK = omp::ClauseScheduleKind;
    spmd::ScheduleKindAttr spmdSched;

    switch (*op.getScheduleKind()) {
    case SK::Static:
      spmdSched = spmd::ScheduleKindAttr::get(ctx, spmd::ScheduleKind::Static);
      break;
    case SK::Dynamic:
      spmdSched = spmd::ScheduleKindAttr::get(ctx, spmd::ScheduleKind::Dynamic);
      break;
    case SK::Guided:
      spmdSched = spmd::ScheduleKindAttr::get(ctx, spmd::ScheduleKind::Guided);
      break;
    case SK::Runtime:
      spmdSched = spmd::ScheduleKindAttr::get(ctx, spmd::ScheduleKind::Runtime);
      break;
    case SK::Auto:
      spmdSched = spmd::ScheduleKindAttr::get(ctx, spmd::ScheduleKind::Auto);
      break;
    case SK::Distribute:
      return rewriter.notifyMatchFailure(
          op, "distribute schedule not supported for wsloop lowering");
    default:
      llvm::reportFatalUsageError("unsupported schedule kind in SPMD lowering");
    }

    SmallVector<Value> reductionVars = llvm::to_vector(op.getReductionVars());
    ArrayAttr reductionSyms = op.getReductionSymsAttr();

    if (!reductionVars.empty())
      op.emitWarning("[OMPToSPMD] omp.wsloop reduction clause: "
                     "combine is implicit (deferred to explicit spmd.allreduce "
                     "emission — see spmd.declare_reduction description)");

    auto spmdWs = rewriter.create<spmd::WsloopOp>(
        loc, comm, spmdSched, op.getScheduleChunk(), reductionVars,
        reductionSyms, modelGroup);

    rewriter.inlineRegionBefore(op.getRegion(), spmdWs.getRegion(),
                                spmdWs.getRegion().end());

    if (!op.getNowait()) {
      rewriter.setInsertionPointAfter(spmdWs);
      rewriter.create<spmd::BarrierOp>(
          loc, spmd::ErrorType::get(op->getContext()), comm,
          /*isBlocking=*/true,
          /*usedModel=*/(uint32_t)modelGroup,
          /*isUniqueMatching=*/rewriter.getBoolAttr(true));
    }
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.loop_nest -> spmd.loop_nest
//===----------------------------------------------------------------------===//

struct ConvertLoopNestOp : public OpConversionPattern<omp::LoopNestOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(omp::LoopNestOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // spmd.loop_nest has no place for either.  Dropping collapse would make an
    // outer loop's iterations the only ones worksharing distributes, and
    // dropping tiles would change the iteration order.
    if (op.getCollapseNumLoops() > 1 || op.getTileSizes()) {
      op.emitWarning("[OMPToSPMD] omp.loop_nest collapse and tile clauses "
                     "not yet supported");
      return failure();
    }

    // loop_inclusive is part of the iteration space: without it the upper
    // bound, and the last iteration of a C loop written i <= n, drops out.
    auto spmdLoop = rewriter.create<spmd::LoopNestOp>(
        op.getLoc(), adaptor.getLoopLowerBounds(), adaptor.getLoopUpperBounds(),
        adaptor.getLoopSteps(), op.getLoopInclusiveAttr());

    rewriter.inlineRegionBefore(op.getRegion(), spmdLoop.getRegion(),
                                spmdLoop.getRegion().end());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.single -> spmd.single
//===----------------------------------------------------------------------===//

struct ConvertSingleOp : public OpConversionPattern<omp::SingleOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertSingleOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::SingleOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.getCopyprivateVars().empty()) {
      op.emitWarning("[OMPToSPMD] omp.single copyprivate clause "
                     "not yet supported — skipping");
      return failure();
    }

    rewriter.setInsertionPoint(op);
    Value comm = getCommForOp(op, rewriter, modelGroup);

    // nowait is no longer an attribute on spmd.single — the barrier is
    // emitted explicitly here when the original construct requires it.
    auto spmdSingle =
        rewriter.create<spmd::SingleOp>(loc, comm, (uint32_t)modelGroup);

    rewriter.inlineRegionBefore(op.getRegion(), spmdSingle.getRegion(),
                                spmdSingle.getRegion().end());

    // Emit the implicit join barrier unless the original had nowait.
    // In OpenMP all threads — the one that ran the body and all that
    // skipped it — meet at this single barrier point.
    if (!op.getNowait()) {
      rewriter.setInsertionPointAfter(spmdSingle);
      rewriter.create<spmd::BarrierOp>(
          loc, spmd::ErrorType::get(op->getContext()), comm,
          /*isBlocking=*/true,
          /*usedModel=*/(uint32_t)modelGroup,
          /*isUniqueMatching=*/rewriter.getBoolAttr(true));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.sections -> spmd.sections
//===----------------------------------------------------------------------===//

struct ConvertSectionsOp : public OpConversionPattern<omp::SectionsOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertSectionsOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::SectionsOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.getAllocateVars().empty()) {
      op.emitWarning(
          "[OMPToSPMD] omp.sections allocate clause not yet supported");
      return failure();
    }

    rewriter.setInsertionPoint(op);
    Value comm = getCommForOp(op, rewriter, modelGroup);

    SmallVector<Value> privateVars = llvm::to_vector(op.getPrivateVars());
    SmallVector<Value> reductionVars = llvm::to_vector(op.getReductionVars());
    ArrayAttr reductionSyms = op.getReductionSymsAttr();

    auto spmdSections =
        rewriter.create<spmd::SectionsOp>(loc, comm, privateVars, reductionVars,
                                          reductionSyms, (uint32_t)modelGroup);

    rewriter.inlineRegionBefore(op.getRegion(), spmdSections.getRegion(),
                                spmdSections.getRegion().end());

    // Emit the implicit join barrier unless the original had nowait.
    if (!op.getNowait()) {
      rewriter.setInsertionPointAfter(spmdSections);
      rewriter.create<spmd::BarrierOp>(
          loc, spmd::ErrorType::get(op->getContext()), comm,
          /*isBlocking=*/true,
          /*usedModel=*/(uint32_t)modelGroup,
          /*isUniqueMatching=*/rewriter.getBoolAttr(true));
    }

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.section -> spmd.section
//===----------------------------------------------------------------------===//

struct ConvertSectionOp : public OpConversionPattern<omp::SectionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(omp::SectionOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto spmdSection = rewriter.create<spmd::SectionOp>(op.getLoc());

    rewriter.inlineRegionBefore(op.getRegion(), spmdSection.getRegion(),
                                spmdSection.getRegion().end());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.critical -> spmd.critical
//===----------------------------------------------------------------------===//

struct ConvertCriticalOp : public OpConversionPattern<omp::CriticalOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertCriticalOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::CriticalOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringAttr nameAttr;
    if (auto symRef = op.getNameAttr())
      nameAttr = rewriter.getStringAttr(symRef.getValue());

    auto spmdCrit = rewriter.create<spmd::CriticalOp>(op.getLoc(), nameAttr,
                                                      (uint32_t)modelGroup);

    rewriter.inlineRegionBefore(op.getRegion(), spmdCrit.getRegion(),
                                spmdCrit.getRegion().end());

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.master -> if rank == 0
//===----------------------------------------------------------------------===//

struct ConvertMasterOp : public OpConversionPattern<omp::MasterOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertMasterOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::MasterOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    rewriter.setInsertionPoint(op);
    Value comm = getCommForOp(op, rewriter, modelGroup);

    auto rankOp =
        rewriter.create<spmd::GetRankInCommOp>(loc, comm, (uint32_t)modelGroup);
    Value zero =
        rewriter.create<arith::ConstantIntOp>(loc, rewriter.getI32Type(), 0);
    Value isMaster = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rankOp.getRank(), zero);

    auto ifOp = rewriter.create<scf::IfOp>(loc, isMaster,
                                           /*withElseRegion=*/false);

    // Replace the auto-created then-region with the master body region.
    // Then swap the omp.terminator for a scf.yield.
    rewriter.inlineRegionBefore(op.getRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().begin());

    // Remove the now-empty default block scf.if created.
    ifOp.getThenRegion().back().erase();

    // Replace omp.terminator with scf.yield.
    Operation *term = ifOp.getThenRegion().front().getTerminator();
    rewriter.setInsertionPoint(term);
    rewriter.create<scf::YieldOp>(loc);
    rewriter.eraseOp(term);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pattern: omp.masked [filter(x)] -> if threadNum == x  (default x = 0)
//
// omp.masked with no filter is semantically identical to omp.master:
// only thread 0 executes the body.  With filter(x) the target thread is x.
// Both are lowered to an scf.if guarded by a rank comparison — identical
// to ConvertMasterOp, just with a runtime-supplied (or constant 0) filter.
//===----------------------------------------------------------------------===//

struct ConvertMaskedOp : public OpConversionPattern<omp::MaskedOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertMaskedOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::MaskedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    rewriter.setInsertionPoint(op);
    Value comm = getCommForOp(op, rewriter, modelGroup);

    auto rankOp =
        rewriter.create<spmd::GetRankInCommOp>(loc, comm, (uint32_t)modelGroup);

    // filter is optional; absent filter defaults to 0 (master-thread
    // semantics).
    Value filterVal;
    if (Value f = op.getFilteredThreadId())
      filterVal = f;
    else
      filterVal = rewriter.create<arith::ConstantIntOp>(loc, 0, /*width=*/32u);

    Value isSelected = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::eq, rankOp.getRank(), filterVal);

    auto ifOp = rewriter.create<scf::IfOp>(loc, isSelected,
                                           /*withElseRegion=*/false);

    // Move the masked body into the scf.if then-region,
    // displacing the empty block scf.if generates by default.
    rewriter.inlineRegionBefore(op.getRegion(), ifOp.getThenRegion(),
                                ifOp.getThenRegion().begin());
    ifOp.getThenRegion().back().erase();

    // Swap the omp.terminator the inlined body still carries for scf.yield.
    // (ConvertTerminatorOp won't fire here — parent is now scf.if, not
    //  omp/spmd.parallel — so we handle it manually, exactly as master does.)
    Operation *term = ifOp.getThenRegion().front().getTerminator();
    rewriter.setInsertionPoint(term);
    rewriter.create<scf::YieldOp>(loc);
    rewriter.eraseOp(term);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTaskOp : public OpConversionPattern<omp::TaskOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertTaskOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::TaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.getAllocateVars().empty())
      op.emitWarning("[OMPToSPMD] omp.task allocate clause not yet supported");

    if (!op.getInReductionVars().empty())
      op.emitWarning("[OMPToSPMD] omp.task in_reduction clause: "
                     "reduction operators deferred to follow-up pass");

    // The detach clause surfaces as event_handle in the generated interface,
    // not getDetach().
    if (op.getEventHandle())
      op.emitWarning("[OMPToSPMD] omp.task detach clause (event_handle) "
                     "not yet supported");

    auto spmdTask = rewriter.create<spmd::TaskOp>(
        loc, adaptor.getDependVars(), adaptor.getIfExpr(),
        adaptor.getFinal(),    // omp names it `final`, not `final_expr`
        op.getUntied(),        // bool, not UnitAttr
        op.getMergeable(),     // bool, not UnitAttr
        (uint32_t)modelGroup); // uint32_t matches primitive overload

    // depend_kinds is an attribute on omp.task, not an operand in our SPMD
    // op. Carry it forward as a discardable attribute so future DAG-precision
    // analysis can read it without touching the builder signature.
    if (auto kindsAttr = op.getDependKindsAttr())
      spmdTask->setAttr("depend_kinds", kindsAttr);

    rewriter.inlineRegionBefore(op.getRegion(), spmdTask.getRegion(),
                                spmdTask.getRegion().end());

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTaskwaitOp : public OpConversionPattern<omp::TaskwaitOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertTaskwaitOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::TaskwaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNowait())
      op.emitWarning("[OMPToSPMD] omp.taskwait nowait clause: "
                     "treated as blocking full-epoch boundary (conservative)");

    // depend_vars: selective taskwait — only awaits tasks that produced
    // these outputs. Preserved for future DAG-precision; epoch analysis
    // conservatively treats as full boundary regardless.
    auto taskwait = rewriter.create<spmd::TaskwaitOp>(
        op.getLoc(), spmd::ErrorType::get(op->getContext()),
        adaptor.getDependVars(), (uint32_t)modelGroup);

    // depend_kinds is metadata, not an operand — carry it as a discardable attr
    if (auto kindsAttr = op.getDependKindsAttr())
      taskwait->setAttr("depend_kinds", kindsAttr);

    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTaskgroupOp : public OpConversionPattern<omp::TaskgroupOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertTaskgroupOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::TaskgroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.getAllocateVars().empty())
      op.emitWarning(
          "[OMPToSPMD] omp.taskgroup allocate clause not yet supported");

    if (!op.getTaskReductionVars().empty())
      op.emitWarning("[OMPToSPMD] omp.taskgroup task_reduction clause: "
                     "reduction operators deferred to follow-up pass");

    auto spmdTaskgroup = rewriter.create<spmd::TaskgroupOp>(
        loc, spmd::ErrorType::get(op->getContext()),
        adaptor.getTaskReductionVars(), op.getTaskReductionSymsAttr(),
        (uint32_t)modelGroup);

    rewriter.inlineRegionBefore(op.getRegion(), spmdTaskgroup.getRegion(),
                                spmdTaskgroup.getRegion().end());

    // The region end is the implicit epoch boundary — no explicit
    // spmd.taskwait needed; analyses inspect the spmd.taskgroup op type.
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTaskloopOp : public OpConversionPattern<omp::TaskloopOp> {
  using OpConversionPattern::OpConversionPattern;
  int modelGroup;

  ConvertTaskloopOp(MLIRContext *ctx, int modelGroup)
      : OpConversionPattern(ctx), modelGroup(modelGroup) {}

  LogicalResult
  matchAndRewrite(omp::TaskloopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    if (!op.getAllocateVars().empty())
      op.emitWarning(
          "[OMPToSPMD] omp.taskloop allocate clause not yet supported");

    if (!op.getInReductionVars().empty())
      op.emitWarning("[OMPToSPMD] omp.taskloop in_reduction clause: "
                     "reduction operators deferred to follow-up pass");

    if (!op.getReductionVars().empty())
      op.emitWarning("[OMPToSPMD] omp.taskloop reduction clause: "
                     "reduction operators deferred to follow-up pass");

    // nogroup = true: no implicit taskgroup wraps the loop; tasks remain live
    // past the loop end and must be awaited by an outer taskwait/taskgroup.
    // The epoch analysis reads this attribute — do NOT emit a taskwait here.
    auto spmdTaskloop = rewriter.create<spmd::TaskloopOp>(
        loc, adaptor.getGrainsize(), adaptor.getNumTasks(),
        op.getNogroup(), // bool, not UnitAttr
        op.getUntied(),  // bool, not UnitAttr
        (uint32_t)modelGroup);

    // Modifier attrs and mergeable are not operands in spmd.taskloop —
    // carry them as discardable attrs for future analysis precision.
    if (auto attr = op.getGrainsizeModAttr())
      spmdTaskloop->setAttr("grainsize_modifier", attr);
    if (auto attr = op.getNumTasksModAttr())
      spmdTaskloop->setAttr("num_tasks_modifier", attr);
    if (op.getMergeable())
      spmdTaskloop->setAttr("mergeable", rewriter.getUnitAttr());

    rewriter.inlineRegionBefore(op.getRegion(), spmdTaskloop.getRegion(),
                                spmdTaskloop.getRegion().end());
    rewriter.eraseOp(op);
    return success();
  }
};

struct ConvertTaskyieldOp : public OpConversionPattern<omp::TaskyieldOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult
  matchAndRewrite(omp::TaskyieldOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.eraseOp(op); // purely a scheduler hint; no analysis value
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Catch-all: warn and fail for unsupported OpenMP ops
//===----------------------------------------------------------------------===//

template <typename OmpOp>
struct WarnUnsupportedOp : public OpConversionPattern<OmpOp> {
  using OpConversionPattern<OmpOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OmpOp op, typename OmpOp::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return unsupportedOp(op, rewriter);
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct OMPToSPMD : impl::OMPToSPMDBase<OMPToSPMD> {
  using OMPToSPMDBase::OMPToSPMDBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();

    // OpenMP model group — follows MPI(0), SHMEM(1), NVSHMEM(2), NCCL(3),
    // CUDA(4).
    constexpr int ompModelGroup = 5;

    ConversionTarget target(*ctx);
    target.addLegalDialect<spmd::SPMDDialect, arith::ArithDialect,
                           memref::MemRefDialect, scf::SCFDialect,
                           LLVM::LLVMDialect>();
    target.addIllegalDialect<omp::OpenMPDialect>();

    // Supported ops are always converted.
    target.addDynamicallyLegalOp<
        omp::DeclareReductionOp, omp::TerminatorOp, omp::YieldOp,
        omp::ParallelOp, omp::BarrierOp, omp::WsloopOp, omp::LoopNestOp,
        omp::SingleOp, omp::SectionsOp, omp::SectionOp, omp::CriticalOp,
        omp::MaskedOp, omp::TaskOp, omp::TaskwaitOp, omp::TaskgroupOp,
        omp::TaskloopOp, omp::TaskyieldOp>([](Operation *) { return false; });

    RewritePatternSet patterns(ctx);

    patterns
        .add<ConvertDeclareReductionOp, ConvertTerminatorOp, ConvertYieldOp>(
            ctx);

    patterns.add<ConvertParallelOp, ConvertBarrierOp, ConvertWsloopOp,
                 ConvertLoopNestOp, ConvertSingleOp, ConvertSectionsOp,
                 ConvertSectionOp, ConvertCriticalOp, ConvertMasterOp,
                 ConvertMaskedOp, ConvertTaskOp, ConvertTaskwaitOp,
                 ConvertTaskgroupOp, ConvertTaskloopOp, ConvertTaskyieldOp>(
        ctx, ompModelGroup);

    patterns.add<
        WarnUnsupportedOp<omp::TeamsOp>, WarnUnsupportedOp<omp::DistributeOp>,
        WarnUnsupportedOp<omp::SimdOp>, WarnUnsupportedOp<omp::TargetOp>,
        WarnUnsupportedOp<omp::TargetDataOp>,
        WarnUnsupportedOp<omp::TargetEnterDataOp>,
        WarnUnsupportedOp<omp::TargetExitDataOp>,
        WarnUnsupportedOp<omp::TargetUpdateOp>,
        WarnUnsupportedOp<omp::AtomicReadOp>,
        WarnUnsupportedOp<omp::AtomicWriteOp>,
        WarnUnsupportedOp<omp::AtomicUpdateOp>,
        WarnUnsupportedOp<omp::AtomicCaptureOp>,
        WarnUnsupportedOp<omp::OrderedOp>,
        WarnUnsupportedOp<omp::OrderedRegionOp>,
        WarnUnsupportedOp<omp::FlushOp>, WarnUnsupportedOp<omp::LoopOp>>(ctx);

    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError(
          "[OMPToSPMD] conversion failed: one or more OpenMP ops "
          "could not be converted. See above warnings for details.");
      signalPassFailure();
    }
  }
};

} // namespace
} // namespace mlir::spmd