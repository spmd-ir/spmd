// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDPasses.h"
#include "spmd/Utils/PassUtils.h"

#include "CACTS/cudaTransformations.h"
#include "CACTS/mpiTransformations.h"
#include "CACTS/ncclTransformations.h"
#include "CACTS/nvshmemTransformations.h"
#include "CACTS/ompTransformations.h"
#include "CACTS/shmemTransformations.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "spmd/SPMDOps.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "CACTS/common.h"
#include "CACTS/utils.h"

#include "CACTS/cudaSpmdEntries.h"
#include "CACTS/mpiSpmdEntries.h"
#include "CACTS/ncclSpmdEntries.h"
#include "CACTS/nvshmemSpmdEntries.h"
#include "CACTS/ompSpmdEntries.h"
#include "CACTS/shmemSpmdEntries.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/JSON.h"

#include "llvm/Support/Debug.h"

#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir::spmd {
// Define Base Class and Default Constructor for main Class
#define GEN_PASS_DEF_CONVERTAPICALLSTOSPMD
#include "spmd/SPMDPasses.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// RelaxGroupedCommunication
//===----------------------------------------------------------------------===//
//
// Pair every `spmd.group_end` with a *unique* `spmd.group_start` by walking
// backwards through the CFG and, transparently, through function calls. For
// every recovered pair, clear the `isBlocking` attribute on every operation
// with the `Communication` trait lying on a path between the two.
//
// Bail-out cases (pattern returns failure()):
//   * CFG merge  -> block has >1 predecessor    (group_start hides in a branch)
//   * Callee with >1 return-like terminator     (ambiguous exit)
//   * Non-call user of a function symbol        (can't resolve every caller)
//   * No GroupStart reachable at all
//
// Reaching a FunctionOpInterface entry is *not* ambiguous on its own: every
// call site of that function is enqueued as a fresh, independent search. Each
// search produces its own (start, end) pair; the union of their relaxed comm
// ops is what we rewrite. The current bracket depth is carried along so a
// group that straddles a call-graph edge still matches correctly.

// This implementation supports nested groups, however, this might be not
// allowed in the first place, todo: simplify implementation by assuming no
// nested groupStart / End
class RelaxGroupedCommunication : public OpRewritePattern<spmd::GroupEndOp> {
public:
  using OpRewritePattern<spmd::GroupEndOp>::OpRewritePattern;

private:
  enum class WalkOutcome { Found, PropagatedToCallers, Failure };
  struct WorkItem {
    Operation *origin;
    int initialDepth;
  };

public:
  LogicalResult matchAndRewrite(spmd::GroupEndOp endOp,
                                PatternRewriter &rewriter) const override {
    auto moduleOp = endOp->getParentOfType<ModuleOp>();
    if (!moduleOp)
      return rewriter.notifyMatchFailure(endOp, "no enclosing ModuleOp");

    // All comm ops must be co-located with GroupEnd in the same function.
    auto endFunc = endOp->getParentOfType<LLVM::LLVMFuncOp>();
    if (!endFunc)
      return rewriter.notifyMatchFailure(endOp, "no enclosing LLVMFuncOp");

    SymbolTableCollection symbolTable;
    SetVector<Operation *> commOpsToRelax;
    SetVector<Operation *> matchedStarts;

    SmallVector<WorkItem> worklist;
    worklist.push_back({endOp, /*initialDepth=*/0});

    DenseSet<std::pair<Operation *, int>> seen;

    while (!worklist.empty()) {
      WorkItem item = worklist.pop_back_val();
      if (!seen.insert({item.origin, item.initialDepth}).second)
        continue;

      spmd::GroupStartOp matched;
      SmallVector<Operation *> collected;

      WalkOutcome outcome =
          runTopLevelWalk(item.origin, item.initialDepth, symbolTable, moduleOp,
                          worklist, matched, collected);

      switch (outcome) {
      case WalkOutcome::Failure:
        return rewriter.notifyMatchFailure(
            endOp, "could not unambiguously resolve matching GroupStartOp");
      case WalkOutcome::Found:
        matchedStarts.insert(matched);
        break;
      case WalkOutcome::PropagatedToCallers:
        break;
      }
      for (Operation *op : collected) {
        // Only collect ops that live in the same function as GroupEnd.
        // If they are in a different function we cannot pass request Values
        // across the boundary without changing signatures — bail out.
        if (op->getParentOfType<LLVM::LLVMFuncOp>() != endFunc) {
          return rewriter.notifyMatchFailure(
              endOp, "comm op in a different function from GroupEnd, "
                     "cannot propagate request values");
        }
        commOpsToRelax.insert(op);
      }
    }

    if (matchedStarts.empty())
      return rewriter.notifyMatchFailure(endOp, "no matching GroupStartOp");

    if (commOpsToRelax.empty()) {
      for (Operation *start : matchedStarts)
        rewriter.eraseOp(start);
      rewriter.eraseOp(endOp);
      return success();
    }

    auto modelGroupOpt = getUsedModel(endOp);
    if (!modelGroupOpt)
      return rewriter.notifyMatchFailure(endOp, "no usedModel attribute");
    int modelGroup = *modelGroupOpt;

    Location loc = endOp->getLoc();
    int64_t numReqs = static_cast<int64_t>(commOpsToRelax.size());

    // ------------------------------------------------------------------ //
    // 1. Allocate request memref at the entry of the function body so it
    //    dominates all comm ops regardless of their order in the block.
    // ------------------------------------------------------------------ //
    auto reqMemrefType =
        MemRefType::get({numReqs}, rewriter.getType<spmd::ReqType>());
    Block &entryBlock = endFunc.getBody().front();
    rewriter.setInsertionPointToStart(&entryBlock);
    Value reqMemref =
        rewriter.create<memref::AllocaOp>(loc, reqMemrefType).getResult();

    // ------------------------------------------------------------------ //
    // 2. Recreate each comm op with isBlocking=false.
    //    The Optional<ReqType> result is only produced when isBlocking=false,
    //    so we must rebuild the op with the extra result type prepended.
    // ------------------------------------------------------------------ //
    SmallVector<Value> requests;
    int64_t reqIdx = 0;

    for (Operation *op : commOpsToRelax) {
      rewriter.setInsertionPoint(op);

      // Build new result type list: prepend ReqType to existing results.
      SmallVector<Type> newResultTypes;
      newResultTypes.push_back(rewriter.getType<spmd::ReqType>());
      for (Type t : op->getResultTypes())
        newResultTypes.push_back(t);

      // Flip isBlocking -> false, keep all other attributes.
      SmallVector<NamedAttribute> newAttrs;
      for (auto attr : op->getAttrs()) {
        if (attr.getName() == "isBlocking")
          newAttrs.push_back(
              rewriter.getNamedAttr("isBlocking", rewriter.getBoolAttr(false)));
        else
          newAttrs.push_back(attr);
      }

      OperationState state(op->getLoc(), op->getName());
      state.addOperands(op->getOperands());
      state.addTypes(newResultTypes);
      state.addAttributes(newAttrs);
      Operation *newOp = rewriter.create(state);

      // Remap old results (shifted by 1 due to prepended ReqType).
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        op->getResult(i).replaceAllUsesWith(newOp->getResult(i + 1));

      Value req = newOp->getResult(0);
      requests.push_back(req);

      // Store the request into the memref immediately after the new op.
      rewriter.setInsertionPointAfter(newOp);
      Value index =
          rewriter.create<arith::ConstantIndexOp>(loc, reqIdx++).getResult();
      rewriter.create<memref::StoreOp>(loc, req, reqMemref, index);

      rewriter.eraseOp(op);
    }

    // ------------------------------------------------------------------ //
    // 3. Insert spmd.waitAll immediately before GroupEnd.
    // ------------------------------------------------------------------ //
    rewriter.setInsertionPoint(endOp);

    Value count =
        rewriter
            .create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(numReqs))
            .getResult();
    // auto statusMemrefType =
    //     MemRefType::get({numReqs}, rewriter.getType<spmd::StatusType>());
    // Value statusMemref =
    //     rewriter.create<memref::AllocaOp>(loc, statusMemrefType).getResult();

    rewriter.create<spmd::WaitAllOp>(loc, rewriter.getType<spmd::ErrorType>(),
                                     count, reqMemref,
                                     Value(), //  statusMemref,
                                     modelGroup);

    // ------------------------------------------------------------------ //
    // 4. Erase GroupEnd and all matched GroupStart ops.
    // ------------------------------------------------------------------ //
    for (Operation *start : matchedStarts)
      rewriter.eraseOp(start);
    rewriter.eraseOp(endOp);

    return success();
  }

private:
  WalkOutcome runTopLevelWalk(Operation *origin, int initialDepth,
                              SymbolTableCollection &symTable,
                              ModuleOp moduleOp,
                              SmallVectorImpl<WorkItem> &worklist,
                              spmd::GroupStartOp &matched,
                              SmallVectorImpl<Operation *> &collected) const {
    SmallPtrSet<Block *, 16> visited;
    int depth = initialDepth;
    WalkOutcome r = walkInFunction(origin, depth, symTable, moduleOp, visited,
                                   matched, collected);
    if (r != WalkOutcome::PropagatedToCallers)
      return r;

    Operation *func = enclosingFunction(origin);
    if (!func)
      return WalkOutcome::Failure;
    return enqueueCallers(func, depth, moduleOp, worklist);
  }

  WalkOutcome walkInFunction(Operation *origin, int &depth,
                             SymbolTableCollection &symTable, ModuleOp moduleOp,
                             SmallPtrSetImpl<Block *> &visited,
                             spmd::GroupStartOp &matched,
                             SmallVectorImpl<Operation *> &collected) const {
    Operation *cursor = origin;
    while (cursor) {
      Block *block = cursor->getBlock();
      if (!block)
        return WalkOutcome::Failure;
      if (!visited.insert(block).second)
        return WalkOutcome::Failure;

      Block::iterator it = cursor->getIterator();
      while (it != block->begin()) {
        --it;
        Operation *op = &*it;

        if (isa<spmd::GroupEndOp>(op)) {
          ++depth;
          continue;
        }
        if (auto s = dyn_cast<spmd::GroupStartOp>(op)) {
          if (depth == 0) {
            matched = s;
            return WalkOutcome::Found;
          }
          --depth;
          continue;
        }
        if (op->hasTrait<spmd::Communication>()) {
          collected.push_back(op);
          continue;
        }
        if (auto call = dyn_cast<CallOpInterface>(op)) {
          WalkOutcome r =
              descendCall(call, depth, symTable, moduleOp, matched, collected);
          if (r == WalkOutcome::Failure)
            return WalkOutcome::Failure;
          if (r == WalkOutcome::Found)
            return WalkOutcome::Found;
          continue;
        }
      }

      auto preds = block->getPredecessors();
      auto pi = preds.begin(), pe = preds.end();
      if (pi == pe)
        return WalkOutcome::PropagatedToCallers;
      Block *only = *pi++;
      if (pi != pe)
        return WalkOutcome::Failure;
      cursor = only->getTerminator();
    }
    return WalkOutcome::Failure;
  }

  WalkOutcome descendCall(CallOpInterface call, int &depth,
                          SymbolTableCollection &symTable, ModuleOp moduleOp,
                          spmd::GroupStartOp &matched,
                          SmallVectorImpl<Operation *> &collected) const {
    CallInterfaceCallable callee = call.getCallableForCallee();
    auto sym = callee.dyn_cast<SymbolRefAttr>();
    if (!sym)
      return WalkOutcome::PropagatedToCallers;

    auto func = dyn_cast_or_null<FunctionOpInterface>(
        symTable.lookupNearestSymbolFrom(moduleOp, sym));
    if (!func || func.isExternal())
      return WalkOutcome::PropagatedToCallers;

    Operation *uniqueReturn = nullptr;
    for (Block &b : func.getFunctionBody()) {
      Operation *term = b.getTerminator();
      if (!term || !term->hasTrait<OpTrait::IsTerminator>())
        continue;
      if (term->hasTrait<OpTrait::ReturnLike>()) {
        if (uniqueReturn)
          return WalkOutcome::Failure;
        uniqueReturn = term;
      }
    }
    if (!uniqueReturn)
      return WalkOutcome::PropagatedToCallers;

    SmallPtrSet<Block *, 16> calleeVisited;
    return walkInFunction(uniqueReturn, depth, symTable, moduleOp,
                          calleeVisited, matched, collected);
  }

  Operation *enclosingFunction(Operation *op) const {
    for (Operation *p = op; p; p = p->getParentOp())
      if (isa<FunctionOpInterface>(p))
        return p;
    return nullptr;
  }

  WalkOutcome enqueueCallers(Operation *funcOp, int depth, ModuleOp moduleOp,
                             SmallVectorImpl<WorkItem> &worklist) const {
    auto uses = SymbolTable::getSymbolUses(funcOp, moduleOp);
    if (!uses)
      return WalkOutcome::Failure;

    bool any = false;
    for (SymbolTable::SymbolUse use : *uses) {
      auto call = dyn_cast<CallOpInterface>(use.getUser());
      if (!call)
        return WalkOutcome::Failure;
      worklist.push_back({call.getOperation(), depth});
      any = true;
    }
    return any ? WalkOutcome::PropagatedToCallers : WalkOutcome::Failure;
  }
};
//===----------------------------------------------------------------------===//
// Pattern: spmd.init + spmd.finalize -> spmd.parallel
//
// Not implemented as a pattnerRewrite due to the movement of the ops
//
// Wraps all ops between a matched spmd.init and spmd.finalize pair into an
// spmd.parallel region. The pattern is model-group-agnostic — it applies to
// any programming model that uses init/finalize (MPI, SHMEM, NCCL, NVSHMEM,
// CUDA). OpenMP has no init/finalize so the pattern is naturally a no-op there.
//
// Requirements:
//   - init and finalize must be in the same block
//   - init and finalize must have the same usedModel attribute
//   - neither must already be inside an spmd.parallel
//===----------------------------------------------------------------------===//

/// Carries everything wrapInitFinalizeInParallel needs for one model group.
struct ModelGroupInfo {
  int modelGroup;
  // Thread level extracted from the spmd.constThreadLevel op that defines
  // $threadRequired on the matching spmd.init.
  // nullopt when $threadRequired was not a compile-time constant.
  std::optional<mlir::spmd::ThreadLevel> threadLevel;
};

/// Collect all unique model groups reachable from a function via
/// interprocedural walk. Visits calls transitively, avoids cycles.
static void collectModelGroupsInterprocedural(
    LLVM::LLVMFuncOp funcOp, ModuleOp moduleOp,
    SmallVectorImpl<ModelGroupInfo> &orderedGroups,
    llvm::DenseSet<int> &seenGroups,
    llvm::DenseSet<LLVM::LLVMFuncOp> &visitedFuncs) {

  if (!funcOp || !visitedFuncs.insert(funcOp).second)
    return;

  funcOp.walk([&](Operation *op) {
    if (auto initOp = dyn_cast<spmd::InitOp>(op)) {
      auto mg = getUsedModel(initOp);
      if (!mg || !seenGroups.insert(*mg).second)
        return;

      // $threadRequired is always present (non-optional by construction).
      // Try to resolve it statically from the defining spmd.constThreadLevel.
      std::optional<mlir::spmd::ThreadLevel> tLevel;
      if (auto constOp = initOp.getThreadRequired()
                             .getDefiningOp<spmd::ConstThreadLevelOp>()) {
        tLevel = llvm::StringSwitch<mlir::spmd::ThreadLevel>(constOp.getLevel())
                     .Case("SINGLE", mlir::spmd::ThreadLevel::Single)
                     .Case("FUNNELED", mlir::spmd::ThreadLevel::Funneled)
                     .Case("SERIALIZED", mlir::spmd::ThreadLevel::Serialized)
                     .Case("MULTIPLE", mlir::spmd::ThreadLevel::Multiple)
                     .Default(mlir::spmd::ThreadLevel::Single);
      } else {
        LLVM_DEBUG(llvm::dbgs()
                   << "[collectModelGroups] $threadRequired for model group "
                   << *mg
                   << " is not a spmd.constThreadLevel"
                      " — threadLevel will be absent on spmd.parallel\n");
      }

      orderedGroups.push_back({*mg, tLevel});
      return;
    }

    if (auto callOp = dyn_cast<LLVM::CallOp>(op)) {
      if (!callOp.getCallee())
        return;
      auto callee = dyn_cast_or_null<LLVM::LLVMFuncOp>(
          moduleOp.lookupSymbol(*callOp.getCallee()));
      if (callee)
        collectModelGroupsInterprocedural(callee, moduleOp, orderedGroups,
                                          seenGroups, visitedFuncs);
    }
  });
}

static void wrapInitFinalizeInParallel(ModuleOp moduleOp) {
  auto mainFunc =
      dyn_cast_or_null<LLVM::LLVMFuncOp>(moduleOp.lookupSymbol("main"));
  if (!mainFunc || mainFunc.empty()) {
    llvm::errs() << "[WrapInitFinalizeInParallel] no main function found\n";
    return;
  }

  SmallVector<ModelGroupInfo> orderedGroups;
  llvm::DenseSet<int> seenGroups;
  llvm::DenseSet<LLVM::LLVMFuncOp> visitedFuncs;
  collectModelGroupsInterprocedural(mainFunc, moduleOp, orderedGroups,
                                    seenGroups, visitedFuncs);

  if (orderedGroups.empty()) {
    LLVM_DEBUG(llvm::dbgs()
               << "[WrapInitFinalizeInParallel] no spmd.init reachable "
                  "from main — skipping\n");
    return;
  }

  llvm::DenseSet<int> processedGroups;

  for (const ModelGroupInfo &info : orderedGroups) {
    if (!processedGroups.insert(info.modelGroup).second)
      continue;
    if (info.modelGroup == static_cast<int>(APIKind::OMP))
      continue;

    int modelGroup = info.modelGroup;
    Block &entryBlock = mainFunc.front();
    Location loc = mainFunc.getLoc();

    // ------------------------------------------------------------------ //
    // 1. Find ALL llvm.return ops by walking — handles multi-block CFGs
    //    (cf.switch, cf.br, cf.cond_br, …) where llvm.return can live in
    //    any basic block, not only the entry block.
    // ------------------------------------------------------------------ //
    SmallVector<LLVM::ReturnOp> returnOps;
    mainFunc.walk([&](LLVM::ReturnOp retOp) { returnOps.push_back(retOp); });

    if (returnOps.empty()) {
      llvm::errs() << "[WrapInitFinalizeInParallel] main has no llvm.return\n";
      return;
    }

    // All returns must agree on whether they carry an exit code.
    bool hasExitCode = returnOps[0]->getNumOperands() > 0;

    // ------------------------------------------------------------------ //
    // 2. Insert CommWorldOp and ParallelOp at the very top of the entry
    //    block.  Everything that follows will be moved into the region.
    // ------------------------------------------------------------------ //
    OpBuilder builder(&entryBlock, entryBlock.begin());

    Value comm =
        builder.create<spmd::CommWorldOp>(loc, modelGroup).getCommWorld();

    auto parallelOp = builder.create<spmd::ParallelOp>(
        loc,
        /*exit_code=*/hasExitCode ? builder.getI32Type() : Type{},
        /*comm=*/comm,
        /*num_threads=*/Value{},
        /*private_vars=*/ValueRange{},
        /*reduction_vars=*/ValueRange{},
        /*reduction_syms=*/ArrayAttr{},
        /*threadLevel=*/mlir::spmd::ThreadLevelAttr{},
        /*isProcessLevel=*/true,
        /*usedModel=*/(uint32_t)modelGroup);

    // ------------------------------------------------------------------ //
    // 3a. Create the parallel region's entry block and move every op that
    //     comes after the ParallelOp in main's entry block into it.
    //     This covers both:
    //       • single-block main: ops + llvm.return migrate here
    //       • multi-block main:  ops + cf.switch/cf.br/… migrate here
    // ------------------------------------------------------------------ //
    Region &parallelRegion = parallelOp.getRegion();
    Block *parallelEntry = new Block();
    parallelRegion.push_back(parallelEntry);

    {
      SmallVector<Operation *> opsToMove;
      bool pastParallelOp = false;
      for (Operation &op : entryBlock) {
        if (&op == parallelOp.getOperation()) {
          pastParallelOp = true;
          continue;
        }
        if (pastParallelOp)
          opsToMove.push_back(&op);
      }
      for (Operation *op : opsToMove)
        op->moveBefore(parallelEntry, parallelEntry->end());
    }

    // ------------------------------------------------------------------ //
    // 3b. Move every non-entry block (^bb1, ^bb2, … produced by
    //     cf.switch / cf.br / cf.cond_br) into the parallel region so
    //     that all CFG edges remain valid.
    // ------------------------------------------------------------------ //
    {
      SmallVector<Block *> otherBlocks;
      for (Block &block : mainFunc.getBody()) {
        if (&block == &entryBlock)
          continue;
        otherBlocks.push_back(&block);
      }
      for (Block *block : otherBlocks)
        block->moveBefore(&parallelRegion, parallelRegion.end());
    }

    // ------------------------------------------------------------------ //
    // 4. Walk the parallel region and replace EVERY llvm.return with a
    //    synchronising barrier followed by spmd.parallel_yield.
    //    Blocks ending with llvm.unreachable (noreturn paths, e.g. after
    //    @exit()) are left untouched — they are valid region terminators.
    // ------------------------------------------------------------------ //
    SmallVector<LLVM::ReturnOp> parallelRetOps;
    parallelRegion.walk(
        [&](LLVM::ReturnOp retOp) { parallelRetOps.push_back(retOp); });

    for (LLVM::ReturnOp retOp : parallelRetOps) {
      Value retVal =
          retOp->getNumOperands() > 0 ? retOp->getOperand(0) : Value{};
      OpBuilder retBuilder(retOp);
      Location retLoc = retOp.getLoc();

      retBuilder.create<spmd::BarrierOp>(
          retLoc, retBuilder.getType<spmd::ErrorType>(), comm,
          /*isBlocking=*/true,
          /*usedModel=*/(uint32_t)modelGroup,
          /*isUniqueMatching=*/retBuilder.getBoolAttr(true));

      retBuilder.create<spmd::ParallelYieldOp>(retLoc,
                                               hasExitCode ? retVal : Value{});
      retOp.erase();
    }

    // ------------------------------------------------------------------ //
    // 5. Re-terminate main's (now nearly empty) entry block.
    //    It contains only CommWorldOp + ParallelOp; add the final return.
    // ------------------------------------------------------------------ //
    builder.setInsertionPointToEnd(&entryBlock);
    if (hasExitCode)
      builder.create<LLVM::ReturnOp>(loc, parallelOp.getExitCode());
    else
      builder.create<LLVM::ReturnOp>(loc, ValueRange{});

    LLVM_DEBUG(llvm::dbgs()
               << "[WrapInitFinalizeInParallel] wrapped main in process-level "
               << "parallel for model group " << modelGroup
               << (hasExitCode ? " (with exit code)" : " (void)") << "\n");

    // ------------------------------------------------------------------ //
    // 6. Attach the thread-support level.
    // ------------------------------------------------------------------ //
    if (info.threadLevel) {
      parallelOp->setAttr(
          "threadLevel", mlir::spmd::ThreadLevelAttr::get(moduleOp.getContext(),
                                                          *info.threadLevel));
      LLVM_DEBUG(llvm::dbgs()
                 << "[WrapInitFinalizeInParallel] set threadLevel = "
                 << mlir::spmd::stringifyThreadLevel(*info.threadLevel)
                 << " on spmd.parallel for model group " << modelGroup << "\n");
    } else {
      LLVM_DEBUG(llvm::dbgs()
                 << "[WrapInitFinalizeInParallel] threadLevel absent on "
                    "spmd.parallel for model group "
                 << modelGroup
                 << " — analysis will conservatively assume Single\n");
    }
  }
}

void handleErrorCode(LLVM::CallOp &llvmCallOpr, Value newError) {
  OpBuilder builder(llvmCallOpr);
  Location loc = llvmCallOpr->getLoc();
  // Replace all uses of the old call error code with the new error code
  Value oldError = llvmCallOpr->getResult(0);
  Value newErrorI32 = spmd::CastSPMDErrorToI32Op::create(
      builder, loc, builder.getI32Type(), newError);
  oldError.replaceAllUsesWith(newErrorI32);
  LLVM_DEBUG(DBGS() << "Finish handleErrorCode\n");
}

std::optional<Value> handleModelSpecificOperand(
    APIKind apiKind, const llvm::json::Array &params, StringRef paramKind,
    const llvm::json::Object &paramObj, Value oldOperand, unsigned operandIdx,
    LLVM::CallOp llvmCallOp, OpBuilder &builder, Location loc, int modelGroup) {

  switch (apiKind) {
  case APIKind::MPI:
    return handleMPIOperand(params, paramKind, paramObj, oldOperand, operandIdx,
                            llvmCallOp, builder, loc, modelGroup);
  case APIKind::SHMEM:
    return handleSHMEMOperand(params, paramKind, paramObj, oldOperand,
                              operandIdx, llvmCallOp, builder, loc, modelGroup);
  case APIKind::NVSHMEM:
    return handleNVSHMEMOperand(params, paramKind, paramObj, oldOperand,
                                operandIdx, llvmCallOp, builder, loc,
                                modelGroup);
  case APIKind::NCCL:
    return handleNCCLOperand(params, paramKind, paramObj, oldOperand,
                             operandIdx, llvmCallOp, builder, loc, modelGroup);
  case APIKind::CUDA:
    return handleCUDAOperand(params, paramKind, paramObj, oldOperand,
                             operandIdx, llvmCallOp, builder, loc, modelGroup);
  case APIKind::OMP:
    return handleOMPOperand(params, paramKind, paramObj, oldOperand, operandIdx,
                            llvmCallOp, builder, loc, modelGroup);
  default: {
    LLVM_DEBUG(DBGS() << "Model-specific handler not implemented for model: "
                      << toString(apiKind) << "\n");
    return std::nullopt;
  }
  }
}

Value handleNewOperand(const llvm::json::Array &params, Value oldOperand,
                       unsigned operandIdx, LLVM::CallOp llvmCallOp,
                       APIKind apiKind, int modelGroup) {

  Location loc = oldOperand.getLoc();
  OpBuilder builder(llvmCallOp);

  assert(operandIdx < params.size());
  auto *paramObj = params[operandIdx].getAsObject();
  if (!paramObj)
    llvm::reportFatalUsageError("invalid json");

  // Parameters USIS leaves unclassified are passed through unchanged.
  auto kind = getSpmdParamKind(paramObj);
  if (!kind) {
    LLVM_DEBUG(DBGS() << "Parameter " << operandIdx
                      << " has no kind in the API description; passing the "
                         "original operand through\n");
    return oldOperand;
  }

  // =========================
  // SHARED ACROSS ALL MODELS (top-level)
  // =========================
  // if (*kind == "BUFFER") {
  //   auto rootOnly = getParamBool(paramObj, "root_only");
  //   if (!rootOnly)
  //     llvm_unreachable("missing root_only");

  //   if (*rootOnly == true) {
  //     return checkAndGetInPlace(oldOperand, operandIdx, params, llvmCallOp,
  //     modelGroup);
  //   }
  //   return oldOperand;
  // }

  // if (*kind == "STATUS") {
  //   if (isStatusIgnore(oldOperand)) {
  //     return Value(); // pass empty value for MPI_STATUS_IGNORE
  //   }

  //   if (isa<LLVM::LLVMPointerType>(oldOperand.getType())) {
  //     return loadValueFromMemref(oldOperand,
  //     builder.getType<spmd::StatusType>(), builder, loc);
  //   }
  //   else {
  //     llvm_unreachable("status expecting operand has not the type int32 or
  //     LLVMPointerType");
  //   }
  // }

  if (*kind == "BUFFER" && isa<LLVM::LLVMPointerType>(oldOperand.getType())) {
    LLVM_DEBUG(DBGS() << "Old operand is a pointer type, convert to memref\n");
    auto memrefOpt = getMemrefFromPointer(oldOperand, builder, loc);
    if (memrefOpt.has_value()) {
      oldOperand = *memrefOpt;
    } else {
      LLVM_DEBUG(DBGS() << "Failed to convert pointer operand to memref, "
                           "keeping original operand\n");
    }
  }
  // =========================
  // MODEL-SPECIFIC
  // =========================
  auto operandOpt = handleModelSpecificOperand(
      apiKind, params, *kind, *paramObj, oldOperand, operandIdx, llvmCallOp,
      builder, loc, modelGroup);
  if (operandOpt.has_value()) {
    return *operandOpt;
  }

  // =========================
  // SHARED FALLBACK
  // =========================
  LLVM_DEBUG(DBGS() << "Falling back to default handling for operand kind: "
                    << *kind << "\n");

  return oldOperand;
}

// Handles updating/replacing the results after creating the new operation.
// Replaces memory references with new values if the result is a pointer type.
static void handleNewResults(
    LLVM::CallOp &llvmCallOp,
    Operation *newOpr, // Newly created operation producing new results
    const APICallEntry &apiCallEntry,
    OpBuilder &builder, // Builder to insert replacement operations
    Location loc,       // Location for any generated operations
    int modelGroup) {
  LLVM_DEBUG(DBGS() << "Handling new results\n");
  unsigned resultCount = apiCallEntry.producesError
                             ? newOpr->getNumResults() - 1
                             : newOpr->getNumResults();
  LLVM_DEBUG(DBGS() << "resultCount to be handled: " << resultCount << "\n");
  for (unsigned i = 0; i < resultCount; ++i) { //-1 to skip error code
    LLVM_DEBUG(DBGS() << "Handle new result: " << i << "\n");
    Value newResult = newOpr->getResult(i);
    const auto &resultEntry = apiCallEntry.results[i];
    Value oldResult;
    if (resultEntry.apiCallArgIdx.has_value()) {
      oldResult = llvmCallOp.getOperand(resultEntry.apiCallArgIdx.value());
    } else if (resultEntry.apiCallResultIdx.has_value()) {
      oldResult = llvmCallOp.getResult();
    } else {
      // This result was not present in the original call, skip processing
      LLVM_DEBUG(DBGS() << "Result " << i
                        << " not present in original call, skipping\n");
      continue;
    }

    if (resultEntry.conversion) {
      LLVM_DEBUG(DBGS() << "Applying conversion for result " << i << "\n");
      newResult = resultEntry.conversion(builder, loc, newResult, modelGroup);
    }

    bool oldComesFromArg = resultEntry.apiCallArgIdx.has_value();

    auto opResult = dyn_cast<OpResult>(newResult);
    bool isAllocatedResult =
        opResult &&
        hasEffect<OpResult, MemoryEffects::Allocate>(newOpr, opResult);
    if (oldComesFromArg && isa<LLVM::LLVMPointerType>(oldResult.getType()) &&
        (isAllocatedResult ||
         (!isa<LLVM::LLVMPointerType>(newResult.getType()) &&
          !isa<MemRefType>(newResult.getType())))) {
      LLVM_DEBUG(
          DBGS()
          << "Old result is a pointer type, storing new result into memref\n");
      if (failed(storeValueIntoMemref(oldResult, newResult, builder, loc))) {
        LLVM_DEBUG(
            DBGS()
            << "Failed to store result back into pointer-backed memory\n");
        // either skip, or signal pass failure, depending on how hard this case
        // matters
      }
    } else if (oldResult.getType() == newResult.getType()) {
      LLVM_DEBUG(DBGS() << "Old result and new result have the same type, "
                           "replacing uses with dominance check\n");
      Operation *parentOpr = llvmCallOp->getParentOfType<LLVM::LLVMFuncOp>();
      DominanceInfo domInfo(parentOpr);

      oldResult.replaceUsesWithIf(newResult, [&](OpOperand &use) {
        Operation *user = use.getOwner();
        return domInfo.properlyDominates(newOpr, user);
      });
    }
  }

  LLVM_DEBUG(DBGS() << "Handle NewOperands out of old results: \n");

  builder.setInsertionPointAfter(newOpr);
  Value c0 = arith::ConstantIndexOp::create(builder, loc, 0).getResult();
  for (unsigned i = 0; i < apiCallEntry.operands.size(); ++i) {
    const OperandEntry &entry = apiCallEntry.operands[i];
    if (!entry.apiCallResIdx.has_value())
      continue;
    LLVM_DEBUG(DBGS() << "newOperand to be handled with idx: " << i << "\n");

    unsigned resultIdx = entry.apiCallResIdx.value();
    Value origResult = llvmCallOp->getResult(resultIdx);
    Value memrefVal = newOpr->getOperand(i); // already stored here
    Value loaded =
        memref::LoadOp::create(builder, loc, memrefVal, c0).getResult();
    // if (loaded.getType() != origResult.getType()) {
    //     loaded = UnrealizedConversionCastOp::create(
    //                  builder, loc, origResult.getType(), loaded)
    //                  .getResult(0);
    // }
    origResult.replaceAllUsesWith(loaded);
  }

  LLVM_DEBUG(DBGS() << "Finished handling new results and new operands\n");
}

static std::tuple<SmallVector<Value>, SmallVector<Type>,
                  SmallVector<NamedAttribute>>
reorderOperandsAndResultsAndAttributes(LLVM::CallOp &llvmCallOp,
                                       const APICallEntry &apiCallEntry,
                                       const llvm::json::Object &jsonCallEntry,
                                       const APIKind apiKind, int modelGroup) {
  SmallVector<Value> newOperands{apiCallEntry.operands.size()};
  SmallVector<Type> resultTypes;
  SmallVector<NamedAttribute> namedAttrs;
  OpBuilder builder(llvmCallOp);
  Location loc = llvmCallOp.getLoc();

  const llvm::json::Array &params =
      *jsonCallEntry.find("parameters")->second.getAsArray();

  // handle operands
  for (unsigned i = 0; i < apiCallEntry.operands.size(); ++i) {
    LLVM_DEBUG(DBGS() << "Handling operand " << i << "\n");
    const OperandEntry &operandEntry = apiCallEntry.operands[i];
    if (!operandEntry.apiCallArgIdx.has_value() &&
        !operandEntry.apiCallResIdx
             .has_value()) { // handle operands with new default values
      newOperands[i] =
          operandEntry.functionForDefaultValue(builder, loc, modelGroup);
    } else { // handle operands with values from the old llvm.call, with a
             // possible conversion
      auto argIdx = operandEntry.apiCallArgIdx;
      auto resIdx = operandEntry.apiCallResIdx;
      unsigned index;
      // Determine whether to use existing operand or result value
      Value newOperand;
      if (resIdx == std::nullopt) {
        index = argIdx.value();
        newOperand = llvmCallOp.getOperand(index);
      } else if (argIdx == std::nullopt) {
        // supporting V() — allocate a fresh memref buffer for the spmd op to
        // write into
        index = resIdx.value();
        Type elemType = llvmCallOp->getResult(index).getType();
        auto memrefType = MemRefType::get({1}, elemType);
        newOperand =
            memref::AllocaOp::create(builder, loc, memrefType).getResult();
      } else {
        llvm::reportFatalUsageError("Both argIdx and resIdx are not set for an "
                                    "operandEntry. Should be not possible.");
      }

      newOperand = handleNewOperand(params, newOperand, index, llvmCallOp,
                                    apiKind, modelGroup);
      if (operandEntry.conversion) {
        newOperand =
            operandEntry.conversion(builder, loc, newOperand, modelGroup);
      }
      newOperands[i] = newOperand;
    }
  }

  // Absent Optional operands are represented as Value() throughout the
  // conversion machinery.  The generic Op builder crashes when a null Value
  // appears in the operand list because it tries to insert into a null
  // use-list.  Filter them out here; MLIR's verifier distinguishes
  // "Optional present" from "Optional absent" purely by operand count.
  SmallVector<Value> presentOperands;
  presentOperands.reserve(newOperands.size());
  for (Value v : newOperands)
    if (v)
      presentOperands.push_back(v);

  // create resultTypes
  for (unsigned i = 0; i < apiCallEntry.results.size(); ++i) {
    LLVM_DEBUG(DBGS() << "Handling result " << i << "\n");
    const ResultEntry &resultEntry = apiCallEntry.results[i];
    if (resultEntry.apiCallArgIdx.has_value()) {
      if (auto *paramObj = params[*resultEntry.apiCallArgIdx].getAsObject()) {
        std::string kind = getSpmdParamKind(paramObj).value_or("");
        LLVM_DEBUG(DBGS() << "Result kind: " << kind << "\n");
        // A scalar out-parameter becomes an op result; an array one (non-null
        // "length") is an operand the callee writes through instead.
        bool isScalar = !getParamString(paramObj, "length").has_value();
        Type resultType =
            resultEntry.type
                ? resultEntry.type(builder)
                : llvm::StringSwitch<Type>(kind)
                      .Case("REQUEST", builder.getType<spmd::ReqType>())
                      .Case("WINDOW", builder.getType<spmd::WinType>())
                      .Case("COMMUNICATOR", builder.getType<spmd::CommType>())
                      .Case("TEAM", builder.getType<spmd::CommType>())
                      .Case("MESSAGE", builder.getType<spmd::MsgType>())
                      .Case("THREAD_LEVEL",
                            builder.getType<spmd::ThreadLevelType>())
                      .Cases({"RANK", "TEAM_SIZE", "DEVICE"},
                             builder.getI32Type())
                      // The flag of the test family. The C binding spells it
                      // int, so those entries widen it back on write-back.
                      .Case("BOOL", builder.getI1Type())
                      // The index of MPI_Waitany/MPI_Testany: one int, despite
                      // the kind's name.
                      .Case("INDEX_ARRAY",
                            isScalar ? builder.getI32Type() : Type())
                      .Case("STATUS",
                            isStatusIgnore(llvmCallOp.getOperand(
                                *resultEntry.apiCallArgIdx))
                                ? Type() // temporary only
                                : builder.getType<spmd::StatusType>())
                      .Default(llvmCallOp.getOperand(*resultEntry.apiCallArgIdx)
                                   .getType());
        if (resultType)
          resultTypes.push_back(resultType);
      }
    } else if (resultEntry.apiCallResultIdx.has_value()) {
      // Returned ranks, team sizes and device numbers are i32, like the
      // out-parameter form above.
      const llvm::json::Object *returnObj = jsonCallEntry.getObject("return");
      std::optional<llvm::StringRef> valueKind =
          returnObj ? returnObj->getString("value_kind") : std::nullopt;
      Type resultType =
          llvm::StringSwitch<Type>(valueKind.value_or(""))
              .Cases({"RANK", "TEAM_SIZE", "DEVICE"}, builder.getI32Type())
              .Default(llvmCallOp.getResult().getType());
      resultTypes.push_back(resultType);
    } else {
      // resultTypes[i] = Type(); // no value means optional result that is not
      // present
      continue;
    }
  }
  // always add error code as last result
  if (apiCallEntry.producesError)
    resultTypes.push_back(builder.getType<spmd::ErrorType>());

  // handle attributes from JSON
  // execution.blocking is required: several SPMD ops need isBlocking, so a
  // missing value is reported here rather than as a later verifier error.
  const llvm::json::Object *execObj = jsonCallEntry.getObject("execution");
  std::optional<bool> blocking =
      execObj ? execObj->getBoolean("blocking") : std::nullopt;
  if (!blocking)
    llvm::reportFatalUsageError(llvm::Twine("API description for '") +
                                apiCallEntry.spmdOpName + "' (model " +
                                toString(apiKind) +
                                ") has no execution.blocking — regenerate the "
                                "includesProgramModels JSON from USIS");
  namedAttrs.push_back(
      builder.getNamedAttr("isBlocking", builder.getBoolAttr(*blocking)));
  // add additional attributes from the apiCallEntry
  for (const auto &attrFunc : apiCallEntry.newAttributes) {
    namedAttrs.push_back(attrFunc(builder));
  }

  namedAttrs.push_back(
      builder.getNamedAttr("usedModel", builder.getI32IntegerAttr(modelGroup)));

  LLVM_DEBUG(DBGS() << "Finished reordering operands and results\n");

  return std::make_tuple(std::move(presentOperands), std::move(resultTypes),
                         std::move(namedAttrs));
}

void handlePostProcessing(
    const SmallVector<std::function<void(Operation *, Operation *)>>
        &postProcessing,
    Operation *newOpr, Operation *oldOpr) {
  for (auto f : postProcessing) {
    f(newOpr, oldOpr);
  }
  LLVM_DEBUG(DBGS() << "Finished post-processing\n");
}

struct ApiCallMetadata {
  const APICallEntry *apiCallEntry;
  const llvm::json::Object *jsonEntry;
  APIKind apiKind;
  int modelGroup;
};

std::optional<ApiCallMetadata>
resolveApiCallMetadata(LLVM::CallOp callOp, const ApiRegistryEntry &apiEntry) {
  // Callee must exist at this point
  assert(callOp.getCallee() &&
         "CallOp must have a callee before resolving metadata");
  llvm::StringRef callee = *callOp.getCallee();
  llvm::StringRef stripped = stripNumericSuffix(callee);

  // 1. Lookup SPMD mapping
  auto it = apiEntry.apiCallToSpmd->find(stripNumericSuffix(callee));
  if (it == apiEntry.apiCallToSpmd->end()) {
    LLVM_DEBUG(DBGS() << "No SPMD entry for: " << callee
                      << " (stripped: " << stripped << ")\n");
    return std::nullopt;
  }
  const APICallEntry &apiCallEntry = it->second;

  // 2. Lookup JSON
  std::string calleeLower = toLowerStr(stripped);
  auto jsonIt = apiEntry.entryIndex->find(calleeLower);
  if (jsonIt == apiEntry.entryIndex->end()) {
    LLVM_DEBUG(
        DBGS() << "\n---------------------------------------------------- No "
                  "JSON entry for: "
               << calleeLower
               << " ----------------------------------------------------\n");
    return std::nullopt;
  }

  return ApiCallMetadata{&apiCallEntry, jsonIt->second, apiEntry.apiKind,
                         apiEntry.modelGroup};
}

bool convertApiCall(LLVM::CallOp &callOp, const ApiRegistryEntry &apiEntry) {
  auto metaOpt = resolveApiCallMetadata(callOp, apiEntry);
  if (!metaOpt)
    return false;

  const auto &[apiCallEntry, jsonEntry, apiName, modelGroup] = *metaOpt;

  auto [operands, results, attrs] = reorderOperandsAndResultsAndAttributes(
      callOp, *apiCallEntry, *jsonEntry, apiEntry.apiKind, apiEntry.modelGroup);

  OpBuilder builder(callOp);
  Location loc = callOp.getLoc();
  auto nameAttr =
      StringAttr::get(builder.getContext(), apiCallEntry->spmdOpName);

  Operation *newOp = builder.create(loc, nameAttr, operands, results, attrs);

  handleNewResults(callOp, newOp, *apiCallEntry, builder, loc, modelGroup);
  if (apiName == APIKind::MPI && apiCallEntry->producesError)
    handleErrorCode(callOp, newOp->getResult(newOp->getNumResults() - 1));
  handlePostProcessing(apiCallEntry->postProcessing, newOp,
                       callOp.getOperation());

  callOp.erase();

  LLVM_DEBUG(DBGS() << "Finished converting API call\n");
  return true;
}

void handleCallOp(LLVM::CallOp &llvmCallOp, const ApiRegistry &registry) {
  auto calleeOpt = llvmCallOp.getCallee();
  if (!calleeOpt)
    return;

  llvm::StringRef callee = *calleeOpt;
  LLVM_DEBUG(DBGS() << "\nCurrently handling callOp: " << callee << "\n");

  const ApiRegistryEntry *entry = registry.matchByCallee(callee);
  if (!entry) {
    LLVM_DEBUG(
        DBGS() << "Callee does not match any known programming model prefix\n");
    return;
  }

  convertApiCall(llvmCallOp, *entry);
}

static void deleteApiCallees(ModuleOp moduleOp,
                             const llvm::StringMap<APICallEntry> &table) {
  SmallVector<Operation *> symTableOps;
  symTableOps.push_back(moduleOp);
  moduleOp->walk(
      [&](gpu::GPUModuleOp gpuMod) { symTableOps.push_back(gpuMod); });

  for (Operation *symTableOp : symTableOps) {
    SmallVector<Operation *> toErase;
    for (auto &op : symTableOp->getRegion(0).front()) {
      auto symOp = dyn_cast<SymbolOpInterface>(op);
      if (!symOp)
        continue;
      StringRef strippedName = stripNumericSuffix(symOp.getName());
      if (table.count(strippedName)) {
        toErase.push_back(&op);
      }
    }
    for (auto *op : toErase)
      op->erase();
  }
}

struct ConvertApiCallsToSPMD
    : impl::ConvertApiCallsToSPMDBase<ConvertApiCallsToSPMD> {
  using ConvertApiCallsToSPMDBase::ConvertApiCallsToSPMDBase;

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    ApiRegistry registry;
    bool ok = true;

    ok &= registry.registerApi(APIKind::MPI, "MPI_", mpiToSpmdOperands, 0);
    ok &=
        registry.registerApi(APIKind::SHMEM, "shmem_", shmemToSpmdOperands, 1);
    ok &= registry.registerApi(APIKind::NVSHMEM, "nvshmem",
                               nvshmemToSpmdOperands, 2);
    ok &= registry.registerApi(APIKind::NCCL, "nccl", ncclToSpmdOperands, 3);
    ok &= registry.registerApi(APIKind::CUDA, "cuda", cudaToSpmdOperands, 4);
    ok &= registry.registerApi(APIKind::OMP, "omp_", ompToSpmdOperands, 5);

    if (!ok) {
      llvm::errs() << "[SPMD] ERROR: One or more API registrations failed. "
                   << "Aborting ConvertApiCallsToSPMD pass. "
                   << "Ensure all required *_JSON_PATH env-vars are set.\n";
      signalPassFailure();
      return;
    }

    // Opt-in coverage check; see ApiRegistry::checkCoverage.
    if (const char *check = std::getenv("SPMD_CHECK_API_COVERAGE"))
      if (llvm::StringRef(check) == "1") {
        // Listed explicitly, so a new miss is never swallowed by a pattern.
        static constexpr llvm::StringRef kKnownUnresolved[] = {
            // Not declared by NVSHMEM 3.7.1 (kept as placeholders).
            "nvshmem_realloc",
            "nvshmem_byte_atomic_compare_swap_nbi",
            "nvshmem_byte_atomic_fetch_add_nbi",
            "nvshmem_byte_atomic_fetch_and_nbi",
            "nvshmem_byte_atomic_fetch_inc_nbi",
            "nvshmem_byte_atomic_fetch_nbi",
            "nvshmem_byte_atomic_fetch_or_nbi",
            "nvshmem_byte_atomic_fetch_xor_nbi",
            "nvshmem_int_atomic_compare_swap_nbi",
            "nvshmem_int_atomic_fetch_add_nbi",
            "nvshmem_int_atomic_fetch_and_nbi",
            "nvshmem_int_atomic_fetch_inc_nbi",
            "nvshmem_int_atomic_fetch_nbi",
            "nvshmem_int_atomic_fetch_or_nbi",
            "nvshmem_int_atomic_fetch_xor_nbi",
            "nvshmem_uint_atomic_compare_swap_nbi",
            "nvshmem_uint_atomic_fetch_add_nbi",
            "nvshmem_uint_atomic_fetch_and_nbi",
            "nvshmem_uint_atomic_fetch_inc_nbi",
            "nvshmem_uint_atomic_fetch_nbi",
            "nvshmem_uint_atomic_fetch_or_nbi",
            "nvshmem_uint_atomic_fetch_xor_nbi",
            // Over-generated by standardAmoTypes: bitwise atomics exist only
            // for unsigned types, and NVSHMEM has no byte type (TODO in
            // shmemSpmdEntries.cpp).
            "nvshmem_byte_atomic_compare_swap",
            "nvshmem_byte_atomic_fetch",
            "nvshmem_byte_atomic_fetch_add",
            "nvshmem_byte_atomic_fetch_and",
            "nvshmem_byte_atomic_fetch_inc",
            "nvshmem_byte_atomic_fetch_or",
            "nvshmem_byte_atomic_fetch_xor",
            "nvshmem_int_atomic_fetch_and",
            "nvshmem_int_atomic_fetch_or",
            "nvshmem_int_atomic_fetch_xor",
            "shmem_int_atomic_fetch_and",
            "shmem_int_atomic_fetch_and_nbi",
            "shmem_int_atomic_fetch_or",
            "shmem_int_atomic_fetch_or_nbi",
            "shmem_int_atomic_fetch_xor",
            "shmem_int_atomic_fetch_xor_nbi",
            "shmem_long_atomic_fetch_and",
            "shmem_long_atomic_fetch_and_nbi",
            "shmem_long_atomic_fetch_or",
            "shmem_long_atomic_fetch_or_nbi",
            "shmem_long_atomic_fetch_xor",
            "shmem_long_atomic_fetch_xor_nbi",
            "shmem_longlong_atomic_fetch_and",
            "shmem_longlong_atomic_fetch_and_nbi",
            "shmem_longlong_atomic_fetch_or",
            "shmem_longlong_atomic_fetch_or_nbi",
            "shmem_longlong_atomic_fetch_xor",
            "shmem_longlong_atomic_fetch_xor_nbi",
            "shmem_ptrdiff_atomic_fetch_and",
            "shmem_ptrdiff_atomic_fetch_and_nbi",
            "shmem_ptrdiff_atomic_fetch_or",
            "shmem_ptrdiff_atomic_fetch_or_nbi",
            "shmem_ptrdiff_atomic_fetch_xor",
            "shmem_ptrdiff_atomic_fetch_xor_nbi",
            "shmem_size_atomic_fetch_and",
            "shmem_size_atomic_fetch_and_nbi",
            "shmem_size_atomic_fetch_or",
            "shmem_size_atomic_fetch_or_nbi",
            "shmem_size_atomic_fetch_xor",
            "shmem_size_atomic_fetch_xor_nbi",
        };
        if (!registry.checkCoverage(kKnownUnresolved, llvm::errs())) {
          llvm::errs() << "[SPMD] ERROR: unresolved API-map names above. Each "
                          "one is a call the analyses will silently skip.\n";
          signalPassFailure();
          return;
        }
      }

    LLVM_DEBUG(DBGS() << "API Call Conversions" << "\n");
    moduleOp->walk(
        [&](LLVM::CallOp llvmCallOp) { handleCallOp(llvmCallOp, registry); });

    LLVM_DEBUG(DBGS() << "delete API calls" << "\n");
    deleteApiCallees(moduleOp, mpiToSpmdOperands);
    deleteApiCallees(moduleOp, shmemToSpmdOperands);
    deleteApiCallees(moduleOp, nvshmemToSpmdOperands);
    deleteApiCallees(moduleOp, ncclToSpmdOperands);
    deleteApiCallees(moduleOp, cudaToSpmdOperands);
    deleteApiCallees(moduleOp, ompToSpmdOperands);

    RewritePatternSet patterns(&getContext());
    GreedyRewriteConfig config;

    // TODO: above conversion functions could be rewritten as patterns, useful
    // for complex cases
    patterns.insert<SwitchOpUnrealizedCasted,
                    // ReplaceSPMDOpOnArithSelectWithSCFIf,
                    RelaxGroupedCommunication>(&getContext());

    for (const auto &entry : registry.getEntries()) {
      switch (entry.apiKind) {
      case APIKind::MPI:
        populateMPIPatterns(patterns, entry);
        break;
      case APIKind::SHMEM:
        // populateSHMEMPatterns(patterns, entry);
        break;
      case APIKind::NVSHMEM:
        // populateNVSHMEMPatterns(patterns, entry);
        break;
      case APIKind::NCCL:
        // populateNCCLPatterns(patterns, entry);
        break;
      case APIKind::CUDA:
        // populateCUDAPatterns(patterns, entry);
        break;
      case APIKind::OMP:
        populateOMPPatterns(patterns, entry);
        break;
      }
    }
    (void)applyPatternsGreedily(moduleOp, std::move(patterns), config);

    wrapInitFinalizeInParallel(moduleOp);
    // Create a pass manager to run the canonicalizer pass
    // PassManager pm(moduleOp.getContext());
    // pm.addPass(createCanonicalizerPass());
    // Run the pass manager on the current function
    //   if (failed(pm.run(moduleOp))) {
    //     signalPassFailure();
    //   }
  }
};

} // namespace
} // namespace mlir::spmd
