// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToOMP.cpp - Lower SPMD operations to OpenMP ---------------===//
//
// OpenMP is two things to the dialect, and this lowering undoes both.
//
// * Its runtime queries.  omp_get_thread_num and omp_get_num_threads are
//   calls, raised by the CACTS map like MPI_Comm_rank, and lowered the same
//   way: OMPLoweringTable derives them from that map and the USIS description.
//
// * Its directives.  -omp-to-spmd turns the omp dialect operations llvm-to-omp
//   raised into spmd.parallel, spmd.wsloop, spmd.single and the rest.  They
//   have no runtime call of their own -- the OpenMPIRBuilder outlines them
//   when the module is translated to LLVM IR -- so they are lowered back to
//   the omp dialect operations they came from, which
//   -convert-polygeist-to-llvm and mlir-translate already handle.
//
// Three things -omp-to-spmd adds have no OpenMP counterpart and go away:
//
// * The teams.  A thread team is an spmd.commDup of the enclosing team, and
//   the initial one an spmd.commWorld; OpenMP has no object for either.  The
//   operations that ask a team only ever ask the current one, so a team
//   lowers to nothing, and checkSPMDToOMP makes sure no operation names
//   another.
//
// * The implicit barriers.  A worksharing construct without nowait is raised
//   as the construct and an spmd.barrier right after it.  That pair lowers
//   back to the construct without nowait, and a construct no barrier follows
//   gets nowait.  Every other barrier of a thread team is an omp.barrier.
//
// * The error results, which OpenMP's directives do not have: they are 0.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDToOMP.h"
#include "LowerSPMDCommon.h"
#include "OMPLoweringTable.h"

#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::spmd {

#define GEN_PASS_DEF_LOWERSPMDTOOMP
#include "spmd/SPMDPasses.h.inc"

namespace {

/// What an operation's !spmd.error lowers to.  No directive reports anything,
/// and omp_get_thread_num cannot fail.
constexpr int32_t kSuccess = 0;

bool isTaggedForOMP(Operation *op) {
  return op && isTaggedFor(op, ProgrammingModel::OMP);
}

bool isOMPDialectOp(Operation *op) {
  return op && isa<omp::OpenMPDialect>(op->getDialect());
}

bool isThreadTeam(ParallelOp parallel) {
  return !parallel.getIsProcessLevel() && isTaggedForOMP(parallel);
}

/// Whether \p op is an spmd operation that carries no usedModel but belongs to
/// OpenMP all the same: the terminators of its regions, the loop a worksharing
/// construct wraps, one section of a sections construct, the team query the
/// raiser injects for orphaned directives, and the reduction declarations.
/// Only -omp-to-spmd produces any of them.
bool belongsToOMP(Operation *op) {
  if (isa<GetParallelCommOp, DeclareReductionOp>(op))
    return true;
  if (!isa<ParallelYieldOp, TerminatorOp, YieldOp, LoopNestOp, SectionOp>(op))
    return false;
  Operation *parent = op->getParentOp();
  if (!parent)
    return false;
  if (auto parallel = dyn_cast<ParallelOp>(parent))
    return isThreadTeam(parallel);
  return isTaggedForOMP(parent) || isOMPDialectOp(parent) ||
         isa<LoopNestOp, SectionOp, DeclareReductionOp>(parent);
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// A team is no runtime object -- nothing is left of it -- and an error code an
// int.  Everything else a
// directive carries -- the reduction variables, the loop bounds, the number of
// threads -- is already of the type OpenMP takes.
class OMPTypeConverter : public SPMDLoweringTypeConverter {
public:
  explicit OMPTypeConverter(MLIRContext *ctx) : SPMDLoweringTypeConverter(ctx) {
    auto i32 = IntegerType::get(ctx, 32);
    addConversion([i32](CommType) -> Type { return i32; });
    addConversion([i32](ErrorType) -> Type { return i32; });
  }
};

//===----------------------------------------------------------------------===//
// The teams
//===----------------------------------------------------------------------===//

/// The operation's results, in order: nothing for a team, which OpenMP has no
/// object for -- every directive and query acts on the team of the thread
/// that runs it, and checkSPMDToOMP makes sure every use of one is lowered
/// with it -- and success for an error the program reads.
SmallVector<Value> placeholderResults(ConversionPatternRewriter &rewriter,
                                      Operation *op) {
  SmallVector<Value> results;
  for (Value result : op->getResults())
    results.push_back(
        isa<CommType>(result.getType()) || result.use_empty()
            ? Value()
            : createI32Constant(rewriter, op->getLoc(), kSuccess));
  return results;
}

// spmd.commWorld, spmd.commDup and spmd.getParallelComm -> nothing, and
// success for the error of spmd.commDup.
template <typename OpT> struct LowerTeam : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, placeholderResults(rewriter, op));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Reductions
//===----------------------------------------------------------------------===//

/// The element type the reduction \p sym declares, whether it is still an
/// spmd.declare_reduction or already an omp.declare_reduction.
FailureOr<Type> getReductionType(Operation *user, Attribute sym) {
  auto ref = dyn_cast<FlatSymbolRefAttr>(sym);
  if (!ref)
    return failure();
  Operation *decl = SymbolTable::lookupNearestSymbolFrom(user, ref);
  if (auto spmdDecl = dyn_cast_or_null<DeclareReductionOp>(decl))
    return spmdDecl.getType();
  if (auto ompDecl = dyn_cast_or_null<omp::DeclareReductionOp>(decl))
    return ompDecl.getType();
  return failure();
}

/// The symbols and the by-reference flags of the reductions an operation
/// carries.  -omp-to-spmd keeps the symbols but not reduction_byref; OpenMP
/// passes a reduction by reference exactly when its declaration's type is the
/// pointer to the accumulator rather than the accumulated value, so that is
/// what the flag is rebuilt from.
LogicalResult getReductions(Operation *op, ArrayAttr syms,
                            SmallVectorImpl<Attribute> &symbols,
                            SmallVectorImpl<bool> &byref) {
  if (!syms)
    return success();
  for (Attribute sym : syms) {
    FailureOr<Type> type = getReductionType(op, sym);
    if (failed(type))
      return failure();
    symbols.push_back(sym);
    byref.push_back(isa<LLVM::LLVMPointerType>(*type));
  }
  return success();
}

// spmd.declare_reduction -> omp.declare_reduction, regions and all.  $kind is
// the dialect's own annotation of the combiner, which OpenMP reads off the
// region itself.
struct LowerDeclareReduction : public OpConversionPattern<DeclareReductionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(DeclareReductionOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto decl = omp::DeclareReductionOp::create(
        rewriter, op.getLoc(), op.getSymNameAttr(), op.getTypeAttr(),
        op.getByrefElementTypeAttr());
    rewriter.inlineRegionBefore(op.getAllocRegion(), decl.getAllocRegion(),
                                decl.getAllocRegion().end());
    rewriter.inlineRegionBefore(op.getInitializerRegion(),
                                decl.getInitializerRegion(),
                                decl.getInitializerRegion().end());
    rewriter.inlineRegionBefore(op.getReductionRegion(),
                                decl.getReductionRegion(),
                                decl.getReductionRegion().end());
    rewriter.inlineRegionBefore(op.getAtomicReductionRegion(),
                                decl.getAtomicReductionRegion(),
                                decl.getAtomicReductionRegion().end());
    rewriter.inlineRegionBefore(op.getCleanupRegion(), decl.getCleanupRegion(),
                                decl.getCleanupRegion().end());
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Terminators
//===----------------------------------------------------------------------===//

// spmd.parallel_yield of a thread team and spmd.terminator -> omp.terminator.
template <typename OpT>
struct LowerToTerminator : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 0)
      return rewriter.notifyMatchFailure(op, "an exit code of a thread team");
    rewriter.replaceOpWithNewOp<omp::TerminatorOp>(op);
    return success();
  }
};

// spmd.yield -> omp.yield, in a loop nest and in a reduction declaration.
struct LowerYield : public OpConversionPattern<YieldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(YieldOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<omp::YieldOp>(op, adaptor.getResults());
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Directives
//===----------------------------------------------------------------------===//

/// Moves the region of \p from into \p to, which has to have one region with
/// no blocks yet.
void moveRegion(ConversionPatternRewriter &rewriter, Operation *from,
                Operation *to) {
  Region &target = to->getRegion(0);
  rewriter.inlineRegionBefore(from->getRegion(0), target, target.end());
}

// A thread-level spmd.parallel -> omp.parallel.  The team it runs on is the
// one omp.parallel starts; its entry block keeps the reduction variables'
// private copies as arguments, as omp.parallel's does.
struct LowerThreadTeam : public OpConversionPattern<ParallelOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getIsProcessLevel())
      return rewriter.notifyMatchFailure(op, "a process team");

    omp::ParallelOperands clauses;
    if (Value numThreads = adaptor.getNumThreads())
      clauses.numThreadsVars.push_back(numThreads);
    llvm::append_range(clauses.reductionVars, adaptor.getReductionVars());
    if (failed(getReductions(op, op.getReductionSymsAttr(),
                             clauses.reductionSyms, clauses.reductionByref)))
      return rewriter.notifyMatchFailure(op, "unknown reduction");

    auto parallel = omp::ParallelOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, parallel);
    rewriter.eraseOp(op);
    return success();
  }
};

/// The constructs a barrier right after them belongs to, and those barriers.
struct ImplicitBarriers {
  DenseSet<Operation *> constructs;
  DenseSet<Operation *> barriers;
};

std::optional<omp::ClauseScheduleKind> toOMP(ScheduleKind kind) {
  switch (kind) {
  case ScheduleKind::Static:
    return omp::ClauseScheduleKind::Static;
  case ScheduleKind::Dynamic:
    return omp::ClauseScheduleKind::Dynamic;
  case ScheduleKind::Guided:
    return omp::ClauseScheduleKind::Guided;
  case ScheduleKind::Runtime:
    return omp::ClauseScheduleKind::Runtime;
  case ScheduleKind::Auto:
    return omp::ClauseScheduleKind::Auto;
  }
  return std::nullopt;
}

// spmd.wsloop -> omp.wsloop, nowait unless its barrier follows it.
struct LowerWsloop : public OpConversionPattern<WsloopOp> {
  LowerWsloop(const TypeConverter &converter, MLIRContext *ctx,
              const ImplicitBarriers &implicit)
      : OpConversionPattern(converter, ctx), implicit(implicit) {}

  LogicalResult
  matchAndRewrite(WsloopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    MLIRContext *ctx = rewriter.getContext();
    omp::WsloopOperands clauses;
    if (!implicit.constructs.contains(op))
      clauses.nowait = rewriter.getUnitAttr();
    if (std::optional<ScheduleKind> kind = op.getScheduleKind()) {
      std::optional<omp::ClauseScheduleKind> ompKind = toOMP(*kind);
      if (!ompKind)
        return rewriter.notifyMatchFailure(op, "unknown schedule");
      clauses.scheduleKind = omp::ClauseScheduleKindAttr::get(ctx, *ompKind);
      clauses.scheduleChunk = adaptor.getScheduleChunk();
    }
    llvm::append_range(clauses.reductionVars, adaptor.getReductionVars());
    if (failed(getReductions(op, op.getReductionSymsAttr(),
                             clauses.reductionSyms, clauses.reductionByref)))
      return rewriter.notifyMatchFailure(op, "unknown reduction");

    auto wsloop = omp::WsloopOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, wsloop);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const ImplicitBarriers &implicit;
};

// spmd.loop_nest -> omp.loop_nest.  -omp-to-spmd refuses collapse and tiles,
// so the bounds, the steps and whether the upper bound is inclusive are all
// there is.
struct LowerLoopNest : public OpConversionPattern<LoopNestOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LoopNestOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    omp::LoopNestOperands clauses;
    llvm::append_range(clauses.loopLowerBounds, adaptor.getLoopLowerBounds());
    llvm::append_range(clauses.loopUpperBounds, adaptor.getLoopUpperBounds());
    llvm::append_range(clauses.loopSteps, adaptor.getLoopSteps());
    clauses.loopInclusive = op.getLoopInclusiveAttr();
    auto loop = omp::LoopNestOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, loop);
    rewriter.eraseOp(op);
    return success();
  }
};

// spmd.single -> omp.single, nowait unless its barrier follows it.
struct LowerSingle : public OpConversionPattern<SingleOp> {
  LowerSingle(const TypeConverter &converter, MLIRContext *ctx,
              const ImplicitBarriers &implicit)
      : OpConversionPattern(converter, ctx), implicit(implicit) {}

  LogicalResult
  matchAndRewrite(SingleOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    omp::SingleOperands clauses;
    if (!implicit.constructs.contains(op))
      clauses.nowait = rewriter.getUnitAttr();
    auto single = omp::SingleOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, single);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const ImplicitBarriers &implicit;
};

// spmd.sections -> omp.sections, nowait unless its barrier follows it.
// OpenMP's sections ends in omp.terminator; the dialect's need not.
struct LowerSections : public OpConversionPattern<SectionsOp> {
  LowerSections(const TypeConverter &converter, MLIRContext *ctx,
                const ImplicitBarriers &implicit)
      : OpConversionPattern(converter, ctx), implicit(implicit) {}

  LogicalResult
  matchAndRewrite(SectionsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    omp::SectionsOperands clauses;
    if (!implicit.constructs.contains(op))
      clauses.nowait = rewriter.getUnitAttr();
    llvm::append_range(clauses.reductionVars, adaptor.getReductionVars());
    if (failed(getReductions(op, op.getReductionSymsAttr(),
                             clauses.reductionSyms, clauses.reductionByref)))
      return rewriter.notifyMatchFailure(op, "unknown reduction");

    auto sections = omp::SectionsOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, sections);
    // -omp-to-spmd keeps the terminator of a raised omp.sections, which is
    // lowered with the rest of the region; a hand-written spmd.sections has
    // none, since the dialect's needs none.
    Block &body = sections.getRegion().front();
    if (body.empty() || !isa<TerminatorOp, omp::TerminatorOp>(body.back())) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(&body);
      omp::TerminatorOp::create(rewriter, op.getLoc());
    }
    rewriter.replaceOp(op, placeholderResults(rewriter, op));
    return success();
  }

private:
  const ImplicitBarriers &implicit;
};

struct LowerSection : public OpConversionPattern<SectionOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(SectionOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto section = omp::SectionOp::create(rewriter, op.getLoc());
    moveRegion(rewriter, op, section);
    rewriter.eraseOp(op);
    return success();
  }
};

// spmd.critical -> omp.critical, and the omp.critical.declare a named one
// refers to.  -omp-to-spmd keeps the name as a string.
struct LowerCritical : public OpConversionPattern<CriticalOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CriticalOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FlatSymbolRefAttr name;
    if (StringAttr spelled = op.getNameAttr()) {
      auto module = op->getParentOfType<ModuleOp>();
      Operation *existing = SymbolTable::lookupSymbolIn(module, spelled);
      if (existing && !isa<omp::CriticalDeclareOp>(existing))
        return rewriter.notifyMatchFailure(op, "the name is another symbol");
      if (!existing) {
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPointToStart(module.getBody());
        omp::CriticalDeclareOperands clauses;
        clauses.symName = spelled;
        omp::CriticalDeclareOp::create(rewriter, op.getLoc(), clauses);
      }
      name = FlatSymbolRefAttr::get(spelled);
    }
    auto critical = omp::CriticalOp::create(rewriter, op.getLoc(), name);
    moveRegion(rewriter, op, critical);
    rewriter.replaceOp(op, placeholderResults(rewriter, op));
    return success();
  }
};

// spmd.barrier of a thread team -> omp.barrier, or nothing when it is the
// implicit barrier of the construct before it.
struct LowerBarrier : public OpConversionPattern<BarrierOp> {
  LowerBarrier(const TypeConverter &converter, MLIRContext *ctx,
               const ImplicitBarriers &implicit)
      : OpConversionPattern(converter, ctx), implicit(implicit) {}

  LogicalResult
  matchAndRewrite(BarrierOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!implicit.barriers.contains(op))
      omp::BarrierOp::create(rewriter, op.getLoc());
    rewriter.replaceOp(op, placeholderResults(rewriter, op));
    return success();
  }

private:
  const ImplicitBarriers &implicit;
};

/// The depend clause -omp-to-spmd keeps as the discardable depend_kinds.
LogicalResult getDependKinds(Operation *op, size_t numVars,
                             SmallVectorImpl<Attribute> &kinds) {
  if (numVars == 0)
    return success();
  auto attr = op->getAttrOfType<ArrayAttr>("depend_kinds");
  if (!attr || attr.size() != numVars)
    return failure();
  llvm::append_range(kinds, attr.getValue());
  return success();
}

struct LowerTask : public OpConversionPattern<TaskOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TaskOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    omp::TaskOperands clauses;
    llvm::append_range(clauses.dependVars, adaptor.getDependVars());
    if (failed(
            getDependKinds(op, clauses.dependVars.size(), clauses.dependKinds)))
      return rewriter.notifyMatchFailure(op, "depend without its kinds");
    clauses.ifExpr = adaptor.getIfExpr();
    clauses.final = adaptor.getFinalExpr();
    if (op.getUntied())
      clauses.untied = rewriter.getUnitAttr();
    if (op.getMergeable())
      clauses.mergeable = rewriter.getUnitAttr();
    auto task = omp::TaskOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, task);
    rewriter.eraseOp(op);
    return success();
  }
};

struct LowerTaskwait : public OpConversionPattern<TaskwaitOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TaskwaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> dependVars(adaptor.getDependVars());
    SmallVector<Attribute> dependKinds;
    if (failed(getDependKinds(op, dependVars.size(), dependKinds)))
      return rewriter.notifyMatchFailure(op, "depend without its kinds");
    // TaskwaitOperands' builder drops the depend clause, so the generated one
    // is used.
    omp::TaskwaitOp::create(
        rewriter, op.getLoc(),
        dependKinds.empty() ? ArrayAttr() : rewriter.getArrayAttr(dependKinds),
        dependVars, /*depend_iterated_kinds=*/ArrayAttr(),
        /*depend_iterated=*/ValueRange(), /*nowait=*/UnitAttr());
    rewriter.replaceOp(op, placeholderResults(rewriter, op));
    return success();
  }
};

struct LowerTaskgroup : public OpConversionPattern<TaskgroupOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TaskgroupOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    omp::TaskgroupOperands clauses;
    llvm::append_range(clauses.taskReductionVars,
                       adaptor.getTaskReductionVars());
    if (failed(getReductions(op, op.getTaskReductionSymsAttr(),
                             clauses.taskReductionSyms,
                             clauses.taskReductionByref)))
      return rewriter.notifyMatchFailure(op, "unknown reduction");
    auto group = omp::TaskgroupOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, group);
    rewriter.replaceOp(op, placeholderResults(rewriter, op));
    return success();
  }
};

// spmd.taskloop -> omp.taskloop.  The grainsize and num_tasks modifiers and
// mergeable are the discardable attributes -omp-to-spmd keeps them in.
struct LowerTaskloop : public OpConversionPattern<TaskloopOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TaskloopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    omp::TaskloopOperands clauses;
    clauses.grainsize = adaptor.getGrainsize();
    clauses.grainsizeMod =
        op->getAttrOfType<omp::ClauseGrainsizeTypeAttr>("grainsize_modifier");
    clauses.numTasks = adaptor.getNumTasks();
    clauses.numTasksMod =
        op->getAttrOfType<omp::ClauseNumTasksTypeAttr>("num_tasks_modifier");
    if (op.getNogroup())
      clauses.nogroup = rewriter.getUnitAttr();
    if (op.getUntied())
      clauses.untied = rewriter.getUnitAttr();
    if (op->hasAttr("mergeable"))
      clauses.mergeable = rewriter.getUnitAttr();
    auto taskloop = omp::TaskloopOp::create(rewriter, op.getLoc(), clauses);
    moveRegion(rewriter, op, taskloop);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

// spmd.getRankInComm and spmd.getSizeOfComm -> the call OMPLoweringTable picks.
// Both routines take nothing and return what the operation's result holds.
struct OMPTableCallPattern : public ConversionPattern {
  OMPTableCallPattern(const TypeConverter &converter, StringRef rootName,
                      MLIRContext *ctx)
      : ConversionPattern(converter, rootName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value>,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<std::string> reasons;
    const CallDescription *call = selectOMPCall(op, rewriter, reasons);
    if (!call)
      return rewriter.notifyMatchFailure(
          op,
          reasons.empty() ? "no OpenMP call of this shape" : reasons.front());
    if (!call->args.empty() || call->returnResult < 0 ||
        StringRef(call->returnCType).trim() != "int")
      return rewriter.notifyMatchFailure(op, "not a query returning an int");

    Type i32 = rewriter.getI32Type();
    FailureOr<LLVM::LLVMFuncOp> func = getOrDeclareRuntimeFunction(
        rewriter, op, call->callee, LLVM::LLVMFunctionType::get(i32, {}));
    if (failed(func))
      return failure();
    Location loc = op->getLoc();
    Value returned =
        LLVM::CallOp::create(rewriter, loc, *func, ValueRange()).getResult();

    SmallVector<Value> results;
    for (auto [index, type] : llvm::enumerate(op->getResultTypes())) {
      if (static_cast<int>(index) == call->returnResult)
        results.push_back(returned);
      else if (isa<ErrorType>(type))
        results.push_back(op->getResult(index).use_empty()
                              ? Value()
                              : createI32Constant(rewriter, loc, kSuccess));
      else
        return rewriter.notifyMatchFailure(op, "a result the call has not");
    }
    rewriter.replaceOp(op, results);
    return success();
  }
};

void populateOMPCallPatterns(const TypeConverter &c, RewritePatternSet &p) {
  SmallVector<StringRef> lowered;
  for (const CallDescription &call : getOMPLoweringTable())
    if (call.canonical && !llvm::is_contained(lowered, call.spmdOpName))
      lowered.push_back(call.spmdOpName);
  for (StringRef spmdOpName : lowered)
    p.add<OMPTableCallPattern>(c, spmdOpName, p.getContext());
}

//===----------------------------------------------------------------------===//
// The checks
//===----------------------------------------------------------------------===//

/// The innermost thread team around \p op in its function, which is the team
/// OpenMP runs \p op on; null outside every one of them, where the team is
/// whichever the function is called from.
ParallelOp enclosingThreadTeam(Operation *op) {
  for (Operation *parent = op->getParentOp();
       parent && !isa<FunctionOpInterface>(parent);
       parent = parent->getParentOp())
    if (auto parallel = dyn_cast<ParallelOp>(parent);
        parallel && isThreadTeam(parallel))
      return parallel;
  return {};
}

/// Whether \p comm is the team some thread-level spmd.parallel runs on.
bool isThreadTeamComm(Value comm) {
  return llvm::any_of(comm.getUsers(), [&](Operation *user) {
    auto parallel = dyn_cast<ParallelOp>(user);
    return parallel && isThreadTeam(parallel) && parallel.getComm() == comm;
  });
}

/// OpenMP only ever acts on the team of the thread that runs a directive or a
/// query, so an operation at \p position has to name that team: the team of
/// the innermost spmd.parallel around it, or -- outside every one of them in
/// its function -- anything but the team of a parallel region it is not in.
/// The raiser's spmd.getParallelComm is the current team by definition.
LogicalResult checkCurrentTeam(Operation *op, Value comm, Operation *position) {
  if (comm.getDefiningOp<GetParallelCommOp>())
    return success();
  if (ParallelOp team = enclosingThreadTeam(position)) {
    if (team.getComm() == comm)
      return success();
    InFlightDiagnostic diag =
        op->emitOpError()
        << "names a team other than the one of the parallel region it is in; "
           "OpenMP can only act on the current team";
    diag.attachNote(team.getLoc()) << "the current team starts here";
    return diag;
  }
  if (isThreadTeamComm(comm))
    return op->emitOpError()
           << "names the team of a parallel region it is not in; OpenMP can "
              "only act on the current team";
  return success();
}

/// Whether \p op is one of the operations -omp-to-spmd produces with an
/// OpenMP usedModel, all of which have an OpenMP counterpart.
bool hasOMPCounterpart(Operation *op) {
  return isa<CommWorldOp, CommDupOp, ParallelOp, BarrierOp, WsloopOp, SingleOp,
             SectionsOp, CriticalOp, TaskOp, TaskwaitOp, TaskgroupOp,
             TaskloopOp, GetRankInCommOp, GetSizeOfCommOp>(op);
}

/// Whether the reductions of \p op are declared, and its region has an entry
/// argument for each: the private copy an OpenMP construct hands its body.
/// -omp-to-spmd moves the region with its arguments, so they are there unless
/// the module was written or rewritten without them, which omp's verifier
/// would only find once the module is half lowered.
LogicalResult checkReductions(Operation *op, ArrayAttr syms) {
  SmallVector<Attribute> symbols;
  SmallVector<bool> byref;
  if (failed(getReductions(op, syms, symbols, byref)))
    return op->emitOpError() << "names a reduction that is not declared by an "
                                "spmd.declare_reduction";
  Region &region = op->getRegion(0);
  if (!symbols.empty() &&
      (region.empty() || region.front().getNumArguments() < symbols.size()))
    return op->emitOpError()
           << "has " << symbols.size()
           << " reductions, but its region has no entry argument for the "
              "private copy of each";
  return success();
}

/// The operations whose results are a team: those of OpenMP, and the raiser's
/// query for the current team.  Another model's operation cannot use one --
/// there is nothing it could hand to a call.
LogicalResult checkTeamUsers(Operation *producer) {
  bool ok = true;
  for (Value result : producer->getResults()) {
    if (!isa<CommType>(result.getType()))
      continue;
    for (Operation *user : result.getUsers()) {
      if (isTaggedForOMP(user))
        continue;
      InFlightDiagnostic diag =
          user->emitOpError()
          << "uses an OpenMP team, which is no communicator a call of another "
             "programming model can take";
      diag.attachNote(producer->getLoc()) << "team produced here";
      ok = false;
    }
  }
  return success(ok);
}

LogicalResult checkExpressible(Operation *op) {
  if (!hasOMPCounterpart(op))
    return op->emitOpError()
           << "has no OpenMP counterpart; OpenMP lowers thread teams, "
              "worksharing, tasks, barriers and the thread number and count "
              "queries";

  if (auto parallel = dyn_cast<ParallelOp>(op)) {
    if (parallel.getIsProcessLevel())
      return op->emitOpError()
             << "is a process team; OpenMP starts threads, not processes";
    if (!parallel.getPrivateVars().empty())
      return op->emitOpError()
             << "has private variables, but -omp-to-spmd kept none of the "
                "omp.private declarations that say how to copy them";
    if (parallel.getExitCode())
      return op->emitOpError() << "yields an exit code, which a thread team "
                                  "has not";
    if (failed(checkReductions(op, parallel.getReductionSymsAttr())))
      return failure();
    Value comm = parallel.getComm();
    if (auto dup = comm.getDefiningOp<CommDupOp>(); dup && isTaggedForOMP(dup))
      comm = dup.getOldComm();
    return checkCurrentTeam(op, comm, op);
  }

  if (auto barrier = dyn_cast<BarrierOp>(op)) {
    if (auto blocking = barrier->getAttrOfType<BoolAttr>("isBlocking");
        blocking && !blocking.getValue())
      return op->emitOpError()
             << "is a non-blocking barrier, which OpenMP does not have";
    return checkCurrentTeam(op, barrier.getComm(), op);
  }
  if (auto wsloop = dyn_cast<WsloopOp>(op)) {
    if (failed(checkReductions(op, wsloop.getReductionSymsAttr())))
      return failure();
    return checkCurrentTeam(op, wsloop.getComm(), op);
  }
  if (auto single = dyn_cast<SingleOp>(op))
    return checkCurrentTeam(op, single.getComm(), op);
  if (auto sections = dyn_cast<SectionsOp>(op)) {
    if (!sections.getPrivateVars().empty())
      return op->emitOpError()
             << "has private variables, but -omp-to-spmd kept none of the "
                "omp.private declarations that say how to copy them";
    if (failed(checkReductions(op, sections.getReductionSymsAttr())))
      return failure();
    return checkCurrentTeam(op, sections.getComm(), op);
  }
  if (auto group = dyn_cast<TaskgroupOp>(op))
    return checkReductions(op, group.getTaskReductionSymsAttr());

  if (auto task = dyn_cast<TaskOp>(op)) {
    SmallVector<Attribute> kinds;
    if (failed(getDependKinds(op, task.getDependVars().size(), kinds)))
      return op->emitOpError()
             << "has depend variables but no depend_kinds for them";
  }
  if (auto taskwait = dyn_cast<TaskwaitOp>(op)) {
    SmallVector<Attribute> kinds;
    if (failed(getDependKinds(op, taskwait.getDependVars().size(), kinds)))
      return op->emitOpError()
             << "has depend variables but no depend_kinds for them";
  }

  if (auto rank = dyn_cast<GetRankInCommOp>(op))
    return checkCurrentTeam(op, rank.getComm(), op);
  if (auto size = dyn_cast<GetSizeOfCommOp>(op))
    return checkCurrentTeam(op, size.getComm(), op);
  return success();
}

/// An operand no OpenMP query has a parameter for, found by reading the CACTS
/// entry backwards.
LogicalResult checkCallable(Operation *op) {
  if (!isa<GetRankInCommOp, GetSizeOfCommOp>(op))
    return success();
  OpBuilder builder(op->getContext());
  SmallVector<std::string> reasons;
  selectOMPCall(op, builder, reasons);
  for (const std::string &reason : reasons)
    op->emitOpError() << reason;
  return failure(!reasons.empty());
}

bool hasOMPQuery(ModuleOp module) {
  return module
      .walk([](Operation *op) {
        return isTaggedForOMP(op) && isa<GetRankInCommOp, GetSizeOfCommOp>(op)
                   ? WalkResult::interrupt()
                   : WalkResult::advance();
      })
      .wasInterrupted();
}

/// Which barriers -omp-to-spmd put right after a worksharing construct, for
/// the construct's implicit one: the first operation after the construct that
/// is not a constant, if it is a barrier of the same team.  Decided before the
/// conversion starts, while the constructs are still there to look at.
ImplicitBarriers findImplicitBarriers(ModuleOp module) {
  ImplicitBarriers implicit;
  module.walk([&](Operation *op) {
    if (!isTaggedForOMP(op) || !isa<WsloopOp, SingleOp, SectionsOp>(op))
      return;
    Value comm = op->getOperand(0);
    Operation *next = op->getNextNode();
    while (next && next->hasTrait<OpTrait::ConstantLike>())
      next = next->getNextNode();
    auto barrier = dyn_cast_or_null<BarrierOp>(next);
    if (!barrier || !isTaggedForOMP(barrier) || barrier.getComm() != comm ||
        implicit.barriers.contains(barrier))
      return;
    implicit.constructs.insert(op);
    implicit.barriers.insert(barrier);
  });
  return implicit;
}

} // namespace

LogicalResult checkSPMDToOMP(ModuleOp module) {
  if (failed(checkUsedModels(module, "lower-spmd-to-omp")))
    return failure();
  // Only the queries are derived from the API description; a module without
  // any needs none, and does not even load it, which would complain.
  if (hasOMPQuery(module) && getOMPLoweringTable().empty())
    return emitError(module.getLoc())
           << "lower-spmd-to-omp: no OpenMP API description; set "
              "OMP_JSON_PATH to includesProgramModels/omp/api.json";

  bool inexpressible = false;
  module.walk([&](Operation *op) {
    if (isa<GetParallelCommOp>(op) ||
        (isTaggedForOMP(op) && isa<CommWorldOp, CommDupOp>(op)))
      if (failed(checkTeamUsers(op)))
        inexpressible = true;
    if (!isTaggedForOMP(op))
      return;
    bool expressible = succeeded(checkExpressible(op));
    bool callable = succeeded(checkCallable(op));
    if (!expressible || !callable)
      inexpressible = true;
  });
  return failure(inexpressible);
}

LogicalResult lowerSPMDToOMP(ModuleOp module, StructureLowering structure) {
  MLIRContext *ctx = module.getContext();
  ctx->getOrLoadDialect<omp::OpenMPDialect>();

  if (failed(checkSPMDToOMP(module)))
    return failure();

  ImplicitBarriers implicit = findImplicitBarriers(module);

  OMPTypeConverter converter(ctx);
  RewritePatternSet patterns(ctx);
  patterns.add<LowerWsloop, LowerSingle, LowerSections, LowerBarrier>(
      converter, ctx, implicit);
  patterns.add<LowerTeam<CommWorldOp>, LowerTeam<CommDupOp>,
               LowerTeam<GetParallelCommOp>, LowerDeclareReduction,
               LowerToTerminator<ParallelYieldOp>,
               LowerToTerminator<TerminatorOp>, LowerYield, LowerThreadTeam,
               LowerLoopNest, LowerSection, LowerCritical, LowerTask,
               LowerTaskwait, LowerTaskgroup, LowerTaskloop>(converter, ctx);
  if (hasOMPQuery(module))
    populateOMPCallPatterns(converter, patterns);
  populateSPMDStructureLoweringPatterns(converter, patterns,
                                        ProgrammingModel::OMP, structure);

  if (failed(applySPMDLowering(module, ProgrammingModel::OMP, converter,
                               std::move(patterns), belongsToOMP,
                               {omp::OpenMPDialect::getDialectNamespace()})))
    return failure();

  // A lowering that ran before this one bridged its own handle to an OpenMP
  // query with a cast back to !spmd.comm -- the process team's communicator
  // the raiser folds an orphaned query's team into.  The query is a call now
  // and takes nothing, so the cast is dead, and nothing after the lowerings
  // would remove it.
  SmallVector<UnrealizedConversionCastOp> dead;
  module.walk([&](UnrealizedConversionCastOp cast) {
    if (cast->use_empty() &&
        llvm::all_of(cast->getResultTypes(),
                     [](Type type) { return isa<CommType>(type); }))
      dead.push_back(cast);
  });
  for (UnrealizedConversionCastOp cast : dead)
    cast->erase();
  return success();
}

namespace {

struct LowerSPMDtoOMPPass
    : public impl::LowerSPMDtoOMPBase<LowerSPMDtoOMPPass> {
  using LowerSPMDtoOMPBase::LowerSPMDtoOMPBase;

  void runOnOperation() override {
    if (failed(lowerSPMDToOMP(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
