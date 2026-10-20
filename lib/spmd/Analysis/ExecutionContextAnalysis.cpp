// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/ExecutionContextAnalysis.h"

#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <iterator>
#include <optional>

#define DEBUG_TYPE "execution-context-analysis"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

using namespace mlir;
using namespace mlir::dataflow;
using namespace mlir::spmd;

namespace {

constexpr int32_t kUniformExecutorSentinel = -1;

/// Walk up the parent chain and return the innermost enclosing
/// spmd::ParallelOp, or nullptr if `op` is outside all parallel regions.
/// Used to select the right level key when querying UniformityLattice.
static Operation *innermostEnclosingParallelOp(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<spmd::ParallelOp>(parent))
      return parent;
  return nullptr;
}

//===----------------------------------------------------------------------===//
// Small sorted-vector set operations
//===----------------------------------------------------------------------===//

static void sortAndUnique(SmallVectorImpl<int32_t> &values) {
  llvm::sort(values);
  values.erase(std::unique(values.begin(), values.end()), values.end());
}

static SmallVector<int32_t> getSortedUniqueCopy(ArrayRef<int32_t> values) {
  SmallVector<int32_t> copy(values);
  sortAndUnique(copy);
  return copy;
}

static SmallVector<int32_t> setUnionVec(ArrayRef<int32_t> lhs,
                                        ArrayRef<int32_t> rhs) {
  SmallVector<int32_t> result(lhs);
  result.append(rhs.begin(), rhs.end());
  sortAndUnique(result);
  return result;
}

static SmallVector<int32_t> setIntersectionVec(ArrayRef<int32_t> lhs,
                                               ArrayRef<int32_t> rhs) {
  SmallVector<int32_t> sortedLhs = getSortedUniqueCopy(lhs);
  SmallVector<int32_t> sortedRhs = getSortedUniqueCopy(rhs);
  SmallVector<int32_t> result;
  std::set_intersection(sortedLhs.begin(), sortedLhs.end(), sortedRhs.begin(),
                        sortedRhs.end(), std::back_inserter(result));
  return result;
}

static SmallVector<int32_t> setDifferenceVec(ArrayRef<int32_t> lhs,
                                             ArrayRef<int32_t> rhs) {
  SmallVector<int32_t> sortedLhs = getSortedUniqueCopy(lhs);
  SmallVector<int32_t> sortedRhs = getSortedUniqueCopy(rhs);
  SmallVector<int32_t> result;
  std::set_difference(sortedLhs.begin(), sortedLhs.end(), sortedRhs.begin(),
                      sortedRhs.end(), std::back_inserter(result));
  return result;
}

//===----------------------------------------------------------------------===//
// Misc helpers
//===----------------------------------------------------------------------===//

static bool hasElseRegion(scf::IfOp ifOp) {
  return ifOp.elseBlock() != nullptr;
}

static bool isRankValue(Value value) {
  return isa_and_nonnull<spmd::GetRankInCommOp>(value.getDefiningOp());
}

static std::optional<int32_t> getConstantInt(Value value) {
  auto constantOp = value.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return std::nullopt;
  auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValue());
  if (!intAttr)
    return std::nullopt;
  return static_cast<int32_t>(intAttr.getInt());
}

//===----------------------------------------------------------------------===//
// LevelInfo helpers
//===----------------------------------------------------------------------===//

static void setKnownLevelInfo(LevelInfo &info, ExecKind kind,
                              ArrayRef<int32_t> execs) {
  info.execKind = kind;
  info.executors.assign(execs.begin(), execs.end());
}

static LevelInfo canonicalizeLevelInfo(LevelInfo info) {
  sortAndUnique(info.executors);
  if (!info.hasKnownExecutors())
    info.executors.clear();
  if (info.execKind == ExecKind::AllBut && info.executors.empty())
    info.execKind = ExecKind::All;
  return info;
}

//===----------------------------------------------------------------------===//
// Uniform execution helpers
//===----------------------------------------------------------------------===//

/// A call to a function that does not return - exit, abort and the like, which
/// clang marks noreturn on the call and on the declaration.
static bool isNoReturnCall(CallOpInterface call) {
  auto llvmCall = dyn_cast<LLVM::CallOp>(call.getOperation());
  if (!llvmCall)
    return false;
  if (llvmCall.getNoreturn())
    return true;
  auto callee = dyn_cast_or_null<LLVM::LLVMFuncOp>(call.resolveCallable());
  return callee && callee.getNoreturn();
}

/// All ranks execute every level of `ctx`, and nothing on the way lets how
/// often, or whether at all, diverge between them.
static bool isUniformContext(const ExecutionContext &ctx) {
  return ctx.uniformCount && ctx.noEarlyExit &&
         llvm::all_of(ctx.levels, [](const LevelInfo &level) {
           return level.execKind == ExecKind::All;
         });
}

/// The blocks reachable from `from` along one or more CFG edges.
static llvm::SmallPtrSet<Block *, 8> getReachableBlocks(Block *from) {
  llvm::SmallPtrSet<Block *, 8> reached;
  SmallVector<Block *> worklist(from->getSuccessors());
  while (!worklist.empty()) {
    Block *block = worklist.pop_back_val();
    if (reached.insert(block).second)
      llvm::append_range(worklist, block->getSuccessors());
  }
  return reached;
}

/// The condition a cf.cond_br or cf.switch decides on.
static Value getBranchCondition(Operation *branch) {
  if (auto condBr = dyn_cast<cf::CondBranchOp>(branch))
    return condBr.getCondition();
  return cast<cf::SwitchOp>(branch).getFlag();
}

/// Control reaches the ops of `block` once each time it reaches the entry of
/// its region: no cf.cond_br or cf.switch whose condition `isUniform` rejects
/// decides whether `block` runs, or whether it runs again. The first is
/// CFGControlInfo's varyingControlBranches, which `info` holds for the region.
/// The second it does not see - a loop body post-dominates its own back edge -
/// so a branch is also checked when `block` lies on a cycle through it.
/// CFGControlInfo only knows those two branches; in a region where any other
/// terminator chooses between successors, only the entry block counts.
static bool
isBlockReachedUniformly(Block *block,
                        const UniformityValueAnalysis::CFGControlInfo &info,
                        llvm::function_ref<bool(Value)> isUniform) {
  for (Block &other : *block->getParent()) {
    Operation *terminator = other.empty() ? nullptr : &other.back();
    if (terminator && terminator->getNumSuccessors() > 1 &&
        !isa<cf::CondBranchOp, cf::SwitchOp>(terminator))
      return block->isEntryBlock();
  }
  auto it = info.varyingControlBranches.find(block);
  if (it != info.varyingControlBranches.end() &&
      !llvm::all_of(it->second, [&](Operation *branch) {
        return isUniform(getBranchCondition(branch));
      }))
    return false;

  llvm::SmallPtrSet<Block *, 8> fromBlock = getReachableBlocks(block);
  if (!fromBlock.contains(block))
    return true; // not on a cycle
  for (Block &other : *block->getParent()) {
    Operation *terminator = other.empty() ? nullptr : &other.back();
    if (!isa_and_nonnull<cf::CondBranchOp, cf::SwitchOp>(terminator))
      continue;
    // `other` decides whether control comes round to `block` again.
    if ((&other == block || fromBlock.contains(&other)) &&
        getReachableBlocks(&other).contains(block) &&
        !isUniform(getBranchCondition(terminator)))
      return false;
  }
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// ExecutionContextLattice::join
//===----------------------------------------------------------------------===//

ChangeResult ExecutionContextLattice::join(const AbstractDenseLattice &other) {
  const auto &rhs = static_cast<const ExecutionContextLattice &>(other);
  if (!rhs.isInitialized())
    return ChangeResult::NoChange;
  if (!isInitialized())
    return setValue(rhs.getValue());
  return setValue(
      ExecutionContextAnalysis::joinContext(getValue(), rhs.getValue()));
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — entry state
//===----------------------------------------------------------------------===//

void ExecutionContextAnalysis::setToEntryState(
    ExecutionContextLattice *lattice) {
  if (auto *pp = dyn_cast<ProgramPoint *>(lattice->getAnchor())) {
    if (Block *block = pp->getBlock()) {
      if (auto funcOp = dyn_cast<FunctionOpInterface>(block->getParentOp())) {
        ExecutionContext entry = ExecutionContext::getEntry();
        // Every rank enters main, once. A function whose callers are not all
        // known - a public one - may be entered from anywhere, any number of
        // times.
        entry.uniformCount = funcOp.getName() == "main";
        propagateIfChanged(lattice, lattice->setValue(entry));
        return;
      }
    }
  }
  // All other program points remain uninitialized; propagation fills them.
}
//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — uniformity queries
//===----------------------------------------------------------------------===//

LogicalResult ExecutionContextAnalysis::getUniformity(Operation *dependentOp,
                                                      Value value,
                                                      Operation *level,
                                                      Uniformity &out) {
  auto *lattice = getOrCreateFor<UniformityLattice>(
      getProgramPointBefore(dependentOp), value);
  if (!lattice || !lattice->isInitialized())
    return failure();
  out = lattice->getUniformityAtLevel(level);
  return success();
}

bool ExecutionContextAnalysis::isConditionNonUniform(
    RegionBranchOpInterface branchOp, const ExecutionContext &ctx) {
  Operation *op = branchOp.getOperation();
  SmallVector<Value> condValues = getConditionOperands(op);
  // Outside all parallels: the condition is trivially uniform.
  if (ctx.levels.empty())
    return false;

  for (Value value : condValues) {
    for (const LevelInfo &l : ctx.levels) {
      Uniformity uniformity;
      // If any lattice is not ready, defer (return false = no refinement yet).
      if (failed(getUniformity(op, value, l.parallelOp, uniformity)))
        return false;
      if (uniformity.isNonUniform())
        return true;
    }
  }
  return false;
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — executor extraction
//===----------------------------------------------------------------------===//

ExecutorResult ExecutionContextAnalysis::getExecutor(Value value,
                                                     Operation *level) {
  if (!value)
    return std::optional<int32_t>{}; // non-uniform

  if (std::optional<int32_t> constant = getConstantInt(value))
    return std::optional<int32_t>(*constant); // concrete constant

  Operation *anchorOp = value.getDefiningOp();
  if (!anchorOp)
    anchorOp = value.getParentBlock()->getParentOp();

  Uniformity uniformity;
  if (failed(getUniformity(anchorOp, value, level, uniformity)))
    return failure(); // not ready yet — defer

  if (!uniformity.isUniform())
    return std::optional<int32_t>{}; // non-uniform
  return std::optional<int32_t>(kUniformExecutorSentinel);
}

void ExecutionContextAnalysis::getCmpOpsNonReversed(
    SmallVectorImpl<Operation *> &cmpOps, Operation *condOp) const {
  if (!condOp)
    llvm::reportFatalUsageError("all predicates must have a defining op");

  if (auto orOp = dyn_cast<arith::OrIOp>(condOp)) {
    getCmpOpsNonReversed(cmpOps, orOp.getRhs().getDefiningOp());
    getCmpOpsNonReversed(cmpOps, orOp.getLhs().getDefiningOp());
    return;
  }

  if (auto cmpIOp = dyn_cast<arith::CmpIOp>(condOp)) {
    if (cmpIOp.getPredicate() != arith::CmpIPredicate::eq)
      return;
    cmpOps.push_back(cmpIOp.getOperation());
    return;
  }

  llvm::reportFatalUsageError("unimplemented case in getCmpOpsNonReversed");
}

void ExecutionContextAnalysis::getCmpOps(SmallVectorImpl<Operation *> &cmpOps,
                                         Operation *condOp,
                                         bool &reversedCase) const {
  if (!condOp)
    return;

  if (auto orOp = dyn_cast<arith::OrIOp>(condOp)) {
    getCmpOpsNonReversed(cmpOps, orOp.getRhs().getDefiningOp());
    getCmpOpsNonReversed(cmpOps, orOp.getLhs().getDefiningOp());
    return;
  }

  if (auto cmpIOp = dyn_cast<arith::CmpIOp>(condOp)) {
    switch (cmpIOp.getPredicate()) {
    case arith::CmpIPredicate::eq:
      cmpOps.push_back(cmpIOp.getOperation());
      return;
    case arith::CmpIPredicate::ne:
      cmpOps.push_back(cmpIOp.getOperation());
      reversedCase = true;
      return;
    case arith::CmpIPredicate::ult:
      cmpOps.push_back(cmpIOp.getOperation());
      return;
    default:
      return;
    }
  }
}

Operation *ExecutionContextAnalysis::getRankLevel(Value rankVal,
                                                  const ExecutionContext &ctx) {
  // Caller (pushIfFound / ult branch) has already verified isRankValue().
  auto rankDefOp = cast<spmd::GetRankInCommOp>(rankVal.getDefiningOp());
  assert(!ctx.empty() &&
         "GetRankInCommOp found outside all parallel regions — malformed IR");

  // Match by model group: the rank op's communicator corresponds to exactly
  // one level in the active context.  The context from `before` is correct
  // here because if `before` changes this visit is retriggered anyway.
  int rankModel = static_cast<int>(rankDefOp.getUsedModel());
  for (const LevelInfo &l : ctx.levels)
    if (l.modelGroup == rankModel)
      return l.parallelOp;

  llvm::reportFatalUsageError(
      "GetRankInCommOp model group not found in execution "
      "context — rank op has no enclosing matching ParallelOp");
}

FailureOr<SmallVector<int32_t>>
ExecutionContextAnalysis::getExecutorsForIf(scf::IfOp ifOp, bool &reversedCase,
                                            const ExecutionContext &ctx,
                                            Operation *&targetLevel) {
  SmallVector<int32_t> executors;
  SmallVector<Operation *> cmpOps;
  getCmpOps(cmpOps, ifOp.getCondition().getDefiningOp(), reversedCase);

  for (Operation *cmpOp : cmpOps) {
    auto cmpIOp = cast<arith::CmpIOp>(cmpOp);
    Value lhs = cmpIOp.getOperand(0);
    Value rhs = cmpIOp.getOperand(1);

    if (cmpIOp.getPredicate() == arith::CmpIPredicate::ult) {
      if (!isRankValue(lhs))
        continue;
      Operation *rankLevel = getRankLevel(lhs, ctx); // asserts on mismatch
      targetLevel = rankLevel;                       // refine this level

      if (std::optional<int32_t> upperLimit = getConstantInt(rhs)) {
        for (int32_t rank = 0; rank < *upperLimit; ++rank)
          executors.push_back(rank);
      } else {
        Uniformity uniformity;
        if (failed(getUniformity(ifOp, rhs, rankLevel, uniformity)))
          return failure(); // not ready — propagate up
        if (uniformity.isUniform())
          executors.push_back(kUniformExecutorSentinel);
        // else: non-uniform upper bound, skip
      }
      continue;
    }

    auto pushIfFound = [&](Value rankVal, Value otherVal)
        -> FailureOr<bool> { // true = pushed, false = skip, failure = not ready
      if (!isRankValue(rankVal))
        return false;
      Operation *rankLevel = getRankLevel(rankVal, ctx);
      targetLevel = rankLevel; // refine this level
      ExecutorResult res = getExecutor(otherVal, rankLevel);
      if (failed(res))
        return failure();   // not ready
      if (auto exec = *res) // success(v): has a single executor
        executors.push_back(*exec);
      // success(nullopt): non-uniform, skip
      return true;
    };

    auto lhsResult = pushIfFound(lhs, rhs);
    if (failed(lhsResult))
      return failure();
    if (*lhsResult)
      continue; // lhs was the rank value, handled

    auto rhsResult = pushIfFound(rhs, lhs);
    if (failed(rhsResult))
      return failure();
  }

  if (executors.size() > 1 &&
      llvm::is_contained(executors, kUniformExecutorSentinel))
    executors.clear();

  sortAndUnique(executors);
  return executors;
}

Operation *ExecutionContextAnalysis::innermostNonUniformLevel(
    scf::IfOp ifOp, const ExecutionContext &ctx) {
  SmallVector<Value> conds = getConditionOperands(ifOp.getOperation());
  for (const LevelInfo &l : llvm::reverse(ctx.levels)) {
    for (Value v : conds) {
      Uniformity u;
      if (failed(getUniformity(ifOp, v, l.parallelOp, u)))
        continue;
      if (u.isNonUniform())
        return l.parallelOp;
    }
  }
  return nullptr;
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — if-branch info
//===----------------------------------------------------------------------===//
ExecutionContextAnalysis::IfBranchInfo
ExecutionContextAnalysis::getIfBranchInfo(
    scf::IfOp ifOp, const ExecutionContextLattice &before) {
  IfBranchInfo result;
  if (!before.isInitialized())
    return result;
  const ExecutionContext &ctx = before.getValue();

  if (!isConditionNonUniform(ifOp, ctx))
    return result;

  bool reversedCase = false;
  Operation *targetLevel = nullptr;
  auto executorsOrFailure =
      getExecutorsForIf(ifOp, reversedCase, ctx, targetLevel);

  // Uniformity info not yet available — leave info.valid = false so the
  // caller propagates the unrefined context instead of emitting Dynamic.
  if (failed(executorsOrFailure))
    return result;
  SmallVector<int32_t> &executors = *executorsOrFailure;
  result.valid = true;

  // No rank guard matched (data-dependent condition): refine the innermost
  // level at which the condition is actually non-uniform, not blindly top().
  if (!targetLevel)
    targetLevel = innermostNonUniformLevel(ifOp, ctx);
  result.targetLevel = targetLevel;

  auto makeLevelRefinement = [](ExecKind kind,
                                ArrayRef<int32_t> execs) -> LevelInfo {
    LevelInfo l;
    l.execKind = kind;
    l.executors.assign(execs.begin(), execs.end());
    return l;
  };

  if (executors.empty()) {
    result.thenRefinement = makeLevelRefinement(ExecKind::Dynamic, {});
    result.elseRefinement = makeLevelRefinement(ExecKind::Dynamic, {});
    return result;
  }

  if (executors.size() == 1 && executors.front() == kUniformExecutorSentinel) {
    if (reversedCase) {
      result.thenRefinement = makeLevelRefinement(ExecKind::AllButOne, {});
      result.elseRefinement = makeLevelRefinement(ExecKind::One, {});
    } else {
      result.thenRefinement = makeLevelRefinement(ExecKind::One, {});
      result.elseRefinement = makeLevelRefinement(ExecKind::AllButOne, {});
    }
    return result;
  }

  if (reversedCase) {
    result.thenRefinement = makeLevelRefinement(ExecKind::AllBut, executors);
    result.elseRefinement = makeLevelRefinement(ExecKind::Static, executors);
  } else {
    result.thenRefinement = makeLevelRefinement(ExecKind::Static, executors);
    result.elseRefinement = makeLevelRefinement(ExecKind::AllBut, executors);
  }

  result.thenRefinement = canonicalizeLevelInfo(result.thenRefinement);
  result.elseRefinement = canonicalizeLevelInfo(result.elseRefinement);
  return result;
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — join / refine
//===----------------------------------------------------------------------===//
// normally if we fix setToEntryState to now have an akward initialization of
// certain lattices,
// then we wouldnever need so precise joining I guess
LevelInfo ExecutionContextAnalysis::joinLevelInfo(const LevelInfo &lhs,
                                                  const LevelInfo &rhs) {
  if (lhs == rhs)
    return canonicalizeLevelInfo(lhs);

  LevelInfo result;
  result.isProcessLevel = lhs.isProcessLevel;
  // Both sides describe the same parallel level, so their ParallelOp* agree —
  // except when one of them is the synthetic top level ExecutionContext::
  // getEntry() puts on a callable's entry block (modelGroup -1, no
  // ParallelOp). The dense framework seeds that before the callsites are
  // known and never retracts it, so it is joined with the real caller context
  // on every interprocedural edge. Take the identity from whichever side
  // actually has one: with parallelOp left null, every uniformity query
  // against the enclosing spmd.parallel answers "uniform" and rank guards
  // inside called functions stop refining.
  assert((lhs.parallelOp == rhs.parallelOp || !lhs.parallelOp ||
          !rhs.parallelOp) &&
         "joining LevelInfos with mismatched ParallelOp — IR is malformed");
  result.parallelOp = lhs.parallelOp ? lhs.parallelOp : rhs.parallelOp;
  result.modelGroup = lhs.modelGroup >= 0 ? lhs.modelGroup : rhs.modelGroup;
  result.execKind = ExecKind::Dynamic; // default: no information

  // ── Tier 1: Dynamic is the top element — absorbs everything ──────────
  if (lhs.execKind == ExecKind::Dynamic || rhs.execKind == ExecKind::Dynamic)
    return canonicalizeLevelInfo(result); // Dynamic

  // ── Tier 2: All is just below top ────────────────────────────────────
  if (lhs.execKind == ExecKind::All || rhs.execKind == ExecKind::All) {
    result.execKind = ExecKind::All;
    return canonicalizeLevelInfo(result);
  }

  // ── Tier 3: One and AllButOne — single-thread kinds ──────────────────
  // One ∪ One           = One  (already handled by lhs==rhs above)
  // One ∪ Static{x}     = One  (both paths: exactly one thread, identity lost)
  // One ∪ Static{x,y,…} = Dynamic (one thread vs multiple — can't characterize)
  // One ∪ AllBut{S}     = Dynamic (one vs all-but-S — can't bound the union)
  // One ∪ AllButOne     = Dynamic
  if (lhs.execKind == ExecKind::One || rhs.execKind == ExecKind::One) {
    const LevelInfo &other = (lhs.execKind == ExecKind::One) ? rhs : lhs;
    if (other.execKind == ExecKind::Static && other.executors.size() == 1)
      result.execKind = ExecKind::One; // both paths: exactly one thread
    // else: Dynamic (default)
    return canonicalizeLevelInfo(result);
  }

  // AllButOne ∪ anything (not handled above) → Dynamic: we don't know
  // which single thread is excluded, so we can't combine with other kinds.
  if (lhs.execKind == ExecKind::AllButOne ||
      rhs.execKind == ExecKind::AllButOne)
    return canonicalizeLevelInfo(result); // Dynamic

  // ── Tier 4: Static and AllBut — precise set kinds ────────────────────
  // Static{A} ∪ Static{B}  = Static{A∪B}
  if (lhs.execKind == ExecKind::Static && rhs.execKind == ExecKind::Static) {
    setKnownLevelInfo(result, ExecKind::Static,
                      setUnionVec(lhs.executors, rhs.executors));
    return canonicalizeLevelInfo(result);
  }

  // AllBut{A} ∪ AllBut{B}  = AllBut{A∩B}  (smaller excluded set → more threads)
  if (lhs.execKind == ExecKind::AllBut && rhs.execKind == ExecKind::AllBut) {
    setKnownLevelInfo(result, ExecKind::AllBut,
                      setIntersectionVec(lhs.executors, rhs.executors));
    return canonicalizeLevelInfo(result);
  }

  // Static{A} ∪ AllBut{B}  = AllBut{B \ A}
  // (A is now covered; only B-threads-not-in-A remain excluded)
  auto joinStaticAllBut = [&](ArrayRef<int32_t> staticExecs,
                              ArrayRef<int32_t> allButExecs) {
    auto excluded = setDifferenceVec(allButExecs, staticExecs);
    if (excluded.empty())
      result.execKind = ExecKind::All;
    else
      setKnownLevelInfo(result, ExecKind::AllBut, excluded);
  };

  if (lhs.execKind == ExecKind::Static && rhs.execKind == ExecKind::AllBut)
    joinStaticAllBut(lhs.executors, rhs.executors);
  else if (lhs.execKind == ExecKind::AllBut && rhs.execKind == ExecKind::Static)
    joinStaticAllBut(rhs.executors, lhs.executors);

  return canonicalizeLevelInfo(result);
}

// mismatching level size should not happen in well-formed SPMD IR
ExecutionContext
ExecutionContextAnalysis::joinContext(const ExecutionContext &lhs,
                                      const ExecutionContext &rhs) {
  if (!lhs.initialized)
    return rhs;
  if (!rhs.initialized)
    return lhs;

  assert(lhs.levels.size() == rhs.levels.size() &&
         "joining ExecutionContexts at different depths — IR has a "
         "control-flow merge across mismatched parallelism nesting");

  ExecutionContext result;
  result.initialized = true;
  result.uniformCount = lhs.uniformCount && rhs.uniformCount;
  result.noEarlyExit = lhs.noEarlyExit && rhs.noEarlyExit;
  result.levels.reserve(lhs.levels.size());
  for (size_t i = 0, e = lhs.levels.size(); i < e; ++i)
    result.levels.push_back(joinLevelInfo(lhs.levels[i], rhs.levels[i]));
  return result;
}

LevelInfo
ExecutionContextAnalysis::refineLevelInfo(const LevelInfo &base,
                                          const LevelInfo &constraint) {
  // All acts as identity for refinement.
  if (constraint.execKind == ExecKind::All)
    return canonicalizeLevelInfo(base);
  if (base.execKind == ExecKind::All) {
    // Take the ExecKind/executors from constraint, but keep the level's
    // identity (modelGroup, isProcessLevel) from base — otherwise
    // innermostThreadLevel() breaks because proc=true overwrites proc=false.
    LevelInfo result = constraint;
    result.modelGroup = base.modelGroup;
    result.isProcessLevel = base.isProcessLevel;
    result.parallelOp = base.parallelOp;
    return canonicalizeLevelInfo(result);
  }

  LevelInfo result = base;
  result.execKind = ExecKind::Dynamic;

  switch (base.execKind) {
  case ExecKind::Static:
    switch (constraint.execKind) {
    case ExecKind::Static:
      setKnownLevelInfo(
          result, ExecKind::Static,
          setIntersectionVec(base.executors, constraint.executors));
      break;
    case ExecKind::AllBut:
      setKnownLevelInfo(result, ExecKind::Static,
                        setDifferenceVec(base.executors, constraint.executors));
      break;
    default:
      break;
    }
    break;

  case ExecKind::AllBut:
    switch (constraint.execKind) {
    case ExecKind::Static:
      setKnownLevelInfo(result, ExecKind::Static,
                        setDifferenceVec(constraint.executors, base.executors));
      break;
    case ExecKind::AllBut:
      setKnownLevelInfo(result, ExecKind::AllBut,
                        setUnionVec(base.executors, constraint.executors));
      break;
    default:
      break;
    }
    break;

  case ExecKind::All:
    llvm_unreachable("handled above");
  default:
    break;
  }

  return canonicalizeLevelInfo(result);
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — propagation helper
//===----------------------------------------------------------------------===//

void ExecutionContextAnalysis::propagateContext(
    const ExecutionContext &ctx, ExecutionContextLattice *after) {
  ExecutionContextLattice tmp(after->getAnchor());
  (void)tmp.setValue(ctx);
  propagateIfChanged(after, after->join(tmp));
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — transfer functions
//===----------------------------------------------------------------------===//

LogicalResult
ExecutionContextAnalysis::visitOperation(Operation *op,
                                         const ExecutionContextLattice &before,
                                         ExecutionContextLattice *after) {
  if (!before.isInitialized())
    return success();

  ExecutionContext ctx = before.getValue();

  if (ctx.empty()) {
    propagateAfter(op, ctx, after);
    return success();
  }

  // Helper: seed the entry program-point of every region of `op`.
  // Needed for spmd ops that do not implement RegionBranchOpInterface.
  auto propagateIntoBody = [&](Operation *regionOp,
                               const ExecutionContext &bodyCtx) {
    for (Region &region : regionOp->getRegions()) {
      if (region.empty())
        continue;
      Block &body = region.front();
      ProgramPoint *blockPP = getProgramPointBefore(&body);
      auto *blockLat = getOrCreate<ExecutionContextLattice>(blockPP);
      // The body runs on the ranks that reach the op, as often as they do.
      ExecutionContext entryCtx = bodyCtx;
      entryCtx.uniformCount =
          entryCtx.uniformCount &&
          isReachedUniformly(regionOp->getBlock(), ctx,
                             getProgramPointAfter(regionOp));
      propagateContext(entryCtx, blockLat);

      //  if (!body.empty()){
      //   Operation &firstOp = body.front();
      //   ProgramPoint *opPP = getProgramPointBefore(&firstOp);
      //  auto *opLat = getOrCreate<ExecutionContextLattice>(opPP);
      //  propagateContext(bodyCtx, opLat);
      //  }
    }
  };

  // spmd.parallel — push a new LevelInfo for the body.
  if (auto parallelOp = dyn_cast<spmd::ParallelOp>(op)) {
    ExecutionContext bodyCtx = ctx;
    LevelInfo newLevel;
    newLevel.modelGroup = (int)parallelOp.getUsedModel();
    newLevel.isProcessLevel = parallelOp.getIsProcessLevel();
    newLevel.execKind = ExecKind::All;
    newLevel.parallelOp = parallelOp.getOperation();
    bodyCtx.levels.push_back(newLevel);
    // A thread team runs the body once per thread, and nothing here shows
    // that the team has the same size on every rank.
    if (!newLevel.isProcessLevel)
      bodyCtx.uniformCount = false;
    propagateIntoBody(op, bodyCtx);
    propagateAfter(op, ctx, after);
    return success();
  }

  // spmd.single — body executed by exactly one thread (ExecKind::One).
  // All threads continue past the op after the implicit barrier.
  if (isa<spmd::SingleOp>(op)) {
    ExecutionContext bodyCtx = ctx;
    if (!bodyCtx.empty()) {
      bodyCtx.top().execKind = ExecKind::One;
      bodyCtx.top().executors.clear();
    }
    propagateIntoBody(op, bodyCtx);
    propagateAfter(op, ctx, after);
    return success();
  }

  // spmd.critical — serialised: one thread at a time (ExecKind::One).
  if (isa<spmd::CriticalOp>(op)) {
    ExecutionContext bodyCtx = ctx;
    if (!bodyCtx.empty()) {
      bodyCtx.top().execKind = ExecKind::One;
      bodyCtx.top().executors.clear();
    }
    propagateIntoBody(op, bodyCtx);
    propagateAfter(op, ctx, after);
    return success();
  }

  // spmd.sections — the outer thread pool is still fully active.
  // The sections region itself (where spmd.section ops live) sees the
  // unchanged context; each spmd.section op will narrow it to One.
  if (isa<spmd::SectionsOp>(op)) {
    propagateIntoBody(op, ctx);     // unchanged ctx into sections region
    propagateAfter(op, ctx, after); // unchanged ctx after the sections op
    return success();
  }

  // spmd.section — exactly one (arbitrary) thread executes this body.
  // ExecKind::One is intentionally NOT master-only:
  //   FUNNELED: isMasterOnly() returns false → violation fires correctly.
  //   SERIALIZED: isThreadLevelConcurrent() returns false → correct within
  //     one section body. Cross-section concurrency is caught by
  //     CollectiveMismatchAnalysis::handleSections (see below).
  if (isa<spmd::SectionOp>(op)) {
    ExecutionContext bodyCtx = ctx;
    if (!bodyCtx.empty()) {
      bodyCtx.top().execKind = ExecKind::One;
      bodyCtx.top().executors.clear();
    }
    propagateIntoBody(op, bodyCtx);
    propagateAfter(op, ctx, after); // restored after the section op
    return success();
  }

  // spmd.wsloop / spmd.taskloop — the iteration space is partitioned across the
  // thread team, so the body is NOT executed uniformly by all threads. Model
  // the innermost (thread) level as Dynamic — an unknown distributed subset —
  // rather than All. (single/critical use One; worksharing is a distributed
  // subset, not a single thread.) The construct itself is always inside a
  // thread-level parallel, so top() is that thread level. The outer context is
  // restored after the construct.
  if (isa<spmd::WsloopOp, spmd::TaskloopOp>(op)) {
    ExecutionContext bodyCtx = ctx;
    if (!bodyCtx.empty()) {
      bodyCtx.top().execKind = ExecKind::Dynamic;
      bodyCtx.top().executors.clear();
    }
    propagateIntoBody(op, bodyCtx);
    propagateAfter(op, ctx, after);
    return success();
  }

  // spmd.loop_nest — replicated within whatever wraps it; propagate unchanged
  // (an enclosing wsloop/taskloop has already set the partition on top()).
  if (isa<spmd::LoopNestOp>(op)) {
    propagateIntoBody(op, ctx);
    propagateAfter(op, ctx, after);
    return success();
  }

  if (isa<spmd::TaskOp>(op)) {
    ExecutionContext bodyCtx = ctx;
    if (!bodyCtx.empty()) {
      bodyCtx.top().execKind = ExecKind::One; // one thread, arbitrary
      bodyCtx.top().executors.clear();
    }
    propagateIntoBody(op, bodyCtx);
    propagateAfter(op, ctx, after); // task creation point: outer ctx unchanged
    return success();
  }

  // spmd.taskgroup — all threads participate in the scope.
  // The implicit epoch boundary at the region end is structural;
  // the ECA models only the execution context, not scheduling order.
  if (isa<spmd::TaskgroupOp>(op)) {
    propagateIntoBody(op, ctx); // outer ctx: all threads active in scope
    propagateAfter(op, ctx, after);
    return success();
  }

  // Region-holding ops the dense framework cannot see into: neither
  // RegionBranchOpInterface (whose region edges are driven by
  // visitRegionBranchControlFlowTransfer) nor a callable, so nothing ever
  // seeds their body's entry program point. `gpu.launch` is the case that
  // matters here: a CUDA kernel launch wraps the `llvm.call` to the device
  // stub, and leaving the body unseeded left every point inside the kernel
  // uninitialized. PendingAccessAnalysis then failed to obtain an execution
  // context for any op in the kernel and silently skipped its whole transfer,
  // so no race inside a device kernel was ever reported. The body executes in
  // the same set of ranks as the launch site, so pass the context through
  // unchanged.
  if (op->getNumRegions() > 0 && !isa<RegionBranchOpInterface>(op) &&
      !isa<FunctionOpInterface>(op) && !isa<ModuleOp>(op))
    propagateIntoBody(op, ctx);

  // All other ops (including RegionBranchOpInterface or DeclareReductionOp):
  // propagate unchanged. Per-branch executor refinement for scf.if is handled
  // in visitRegionBranchControlFlowTransfer.
  propagateAfter(op, ctx, after);
  return success();
}

void ExecutionContextAnalysis::visitRegionBranchControlFlowTransfer(
    RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
    std::optional<unsigned> regionTo, const ExecutionContextLattice &before,
    ExecutionContextLattice *after) {
  if (!before.isInitialized())
    return;

  ExecutionContext ctx = before.getValue();

  // ── scf.if: refine the matching-level for each branch ────────────────
  if (auto ifOp = dyn_cast<scf::IfOp>(branch.getOperation())) {
    IfBranchInfo info = getIfBranchInfo(ifOp, before);
    if (info.valid && !ctx.empty()) {
      auto applyRefinement = [&](const LevelInfo &refinement) {
        if (info.targetLevel) {
          for (LevelInfo &l : ctx.levels) {
            if (l.parallelOp == info.targetLevel) {
              l = refineLevelInfo(l, refinement);
              return;
            }
          }
        }
        // Fallback (no resolvable target): refine the innermost level.
        ctx.top() = refineLevelInfo(ctx.top(), refinement);
      };

      if (!regionFrom && regionTo == 0u)
        applyRefinement(info.thenRefinement);
      else if (!regionFrom && regionTo == 1u && hasElseRegion(ifOp))
        applyRefinement(info.elseRefinement);
    }
  }

  // ── Every other RegionBranchOpInterface (scf.for, scf.while,
  // scf.index_switch
  // ...) ────────────────────────────────────────────────────────────────────
  // The dense framework routes ALL region-branch ops through this hook instead
  // of visitOperation, and the default base implementation joins `before` into
  // `after`. Returning without propagating anything here left every program
  // point inside and after such an op uninitialized, which silently truncated
  // the analysis: a single scf.for anywhere before an scf.if made the guard's
  // ExecutionContext unavailable, so no divergence — and therefore no
  // collective mismatch — could be detected past it. These ops do not change
  // the set of executing ranks/threads, so the context passes through
  // unchanged - apart from how often they run, which follows.
  auto *point = cast<ProgramPoint *>(after->getAnchor());
  if (!regionTo) {
    // Leaving the op: every rank that entered it leaves it once, however
    // often it ran a region inside. Only an early exit is kept.
    const ExecutionContextLattice *atEntry =
        getOrCreateFor<ExecutionContextLattice>(
            point, getProgramPointBefore(branch.getOperation()));
    if (!atEntry->isInitialized())
      return;
    ctx.uniformCount = atEntry->getValue().uniformCount;
    // The region may be left from a block that only some ranks reach.
    if (regionFrom && !branch->getRegion(*regionFrom).hasOneBlock())
      ctx.noEarlyExit = false;
  } else {
    // Entering a region, from the op or from another region: it runs as
    // often on every rank only if what decides that is uniform.
    bool reached =
        regionFrom || isReachedUniformly(branch->getBlock(), ctx, point);
    ctx.uniformCount =
        ctx.uniformCount && reached && entersUniformly(branch, ctx, point);
  }
  propagateContext(ctx, after);
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — calls
//===----------------------------------------------------------------------===//

void ExecutionContextAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, CallControlFlowAction action,
    const ExecutionContextLattice &before, ExecutionContextLattice *after) {
  // Which ranks execute the callee, and the code after the call, stays as the
  // framework computes it; only how often is adjusted. The exit edge restores
  // the count from before the call, which a join with the framework's result
  // could no longer raise, so that edge is propagated here alone.
  if (action != CallControlFlowAction::ExitCallee)
    Base::visitCallControlFlowTransfer(call, action, before, after);
  if (!before.isInitialized())
    return;

  ExecutionContext ctx = before.getValue();
  auto *point = cast<ProgramPoint *>(after->getAnchor());
  switch (action) {
  case CallControlFlowAction::EnterCallee:
    // The callee runs whenever the call does.
    if (isReachedUniformly(call->getBlock(), ctx, point))
      return;
    ctx.uniformCount = false;
    break;
  case CallControlFlowAction::ExitCallee: {
    // Back at the call, the callee ran once per call however it branched -
    // unless not every rank reached a return.
    const ExecutionContextLattice *atCall =
        getOrCreateFor<ExecutionContextLattice>(
            point, getProgramPointBefore(call.getOperation()));
    if (!atCall->isInitialized())
      return;
    ctx.uniformCount = atCall->getValue().uniformCount;
    if (auto callable =
            dyn_cast_or_null<CallableOpInterface>(call.resolveCallable()))
      if (Region *body = callable.getCallableRegion())
        for (Block &block : *body)
          if (!block.empty() && block.back().hasTrait<OpTrait::ReturnLike>() &&
              !isReachedUniformly(&block, ctx, point))
            ctx.noEarlyExit = false;
    break;
  }
  case CallControlFlowAction::ExternalCallee:
    // A function that does not return ends the ranks that call it; if not all
    // of them do, the others go on without them.
    if (!isNoReturnCall(call) ||
        (isUniformContext(ctx) &&
         isReachedUniformly(call->getBlock(), ctx, point)))
      return;
    ctx.noEarlyExit = false;
    break;
  }
  propagateContext(ctx, after);
}

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis — uniform execution
//===----------------------------------------------------------------------===//

void ExecutionContextAnalysis::propagateAfter(Operation *op,
                                              ExecutionContext ctx,
                                              ExecutionContextLattice *after) {
  // The framework does not flow the exits of a region op that is neither a
  // RegionBranchOpInterface nor a callable into the point after it; the
  // context after spmd.parallel, spmd.task, gpu.launch is the one before them.
  // Which ranks run on is right that way, but an early exit inside the body
  // must survive it: a rank that called exit() in an spmd.parallel does not
  // reach the code after it.
  if (op->getNumRegions() > 0 && !isa<RegionBranchOpInterface>(op) &&
      !isa<FunctionOpInterface>(op) && !isa<ModuleOp>(op)) {
    ProgramPoint *point = getProgramPointAfter(op);
    for (Region &region : op->getRegions()) {
      for (Block &block : region) {
        if (block.empty())
          continue;
        Operation *terminator = &block.back();
        // Exits of the region: terminators with nowhere to branch to. An
        // llvm.unreachable is not one, it only ends a path that already left.
        if (terminator->getNumSuccessors() != 0 ||
            !terminator->hasTrait<OpTrait::IsTerminator>() ||
            isa<LLVM::UnreachableOp>(terminator))
          continue;
        const auto *atExit = getOrCreateFor<ExecutionContextLattice>(
            point, getProgramPointAfter(terminator));
        if (!atExit->isInitialized())
          continue;
        if (!atExit->getValue().noEarlyExit ||
            !isReachedUniformly(&block, atExit->getValue(), point))
          ctx.noEarlyExit = false;
      }
    }
  }
  propagateContext(ctx, after);
}

bool ExecutionContextAnalysis::isUniformAtEveryLevel(
    Value value, const ExecutionContext &ctx, ProgramPoint *dependent) {
  const auto *lattice = getOrCreateFor<UniformityLattice>(dependent, value);
  if (!lattice->isInitialized())
    return true;
  return llvm::all_of(ctx.levels, [&](const LevelInfo &level) {
    return lattice->getUniformityAtLevel(level.parallelOp).isUniform();
  });
}

bool ExecutionContextAnalysis::isReachedUniformly(Block *block,
                                                  const ExecutionContext &ctx,
                                                  ProgramPoint *dependent) {
  Region *region = block->getParent();
  if (region->hasOneBlock())
    return true;
  std::unique_ptr<UniformityValueAnalysis::CFGControlInfo> &info =
      cfgControlInfo[region];
  if (!info)
    info = UniformityValueAnalysis::computeCFGControlInfo(*region);
  return isBlockReachedUniformly(block, *info, [&](Value condition) {
    return isUniformAtEveryLevel(condition, ctx, dependent);
  });
}

bool ExecutionContextAnalysis::entersUniformly(RegionBranchOpInterface branch,
                                               const ExecutionContext &ctx,
                                               ProgramPoint *dependent) {
  Operation *op = branch.getOperation();
  SmallVector<Value> decisive;
  if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    decisive = {forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep()};
  } else if (auto parallelOp = dyn_cast<scf::ParallelOp>(op)) {
    llvm::append_range(decisive, parallelOp.getLowerBound());
    llvm::append_range(decisive, parallelOp.getUpperBound());
    llvm::append_range(decisive, parallelOp.getStep());
  } else if (isa<affine::AffineForOp>(op)) {
    llvm::append_range(decisive, op->getOperands());
  } else if (isa<scf::IfOp, scf::WhileOp, scf::IndexSwitchOp,
                 affine::AffineIfOp>(op)) {
    decisive = getConditionOperands(op);
  } else if (!isa<scf::ExecuteRegionOp>(op)) {
    // A construct not known here may run its regions any number of times.
    return false;
  }
  return llvm::all_of(decisive, [&](Value value) {
    return isUniformAtEveryLevel(value, ctx, dependent);
  });
}

//===----------------------------------------------------------------------===//
// Queries
//===----------------------------------------------------------------------===//

bool mlir::spmd::isUniformValue(Value value, Operation *op,
                                DataFlowSolver &solver) {
  const auto *ctxLattice = solver.lookupState<ExecutionContextLattice>(
      solver.getProgramPointBefore(op));
  const auto *lattice = solver.lookupState<UniformityLattice>(value);
  if (!ctxLattice || !ctxLattice->isInitialized() || !lattice ||
      !lattice->isInitialized())
    return false;
  return llvm::all_of(
      ctxLattice->getValue().levels, [&](const LevelInfo &level) {
        return lattice->getUniformityAtLevel(level.parallelOp).isUniform();
      });
}

bool mlir::spmd::isUniformlyExecuted(Operation *op, DataFlowSolver &solver) {
  const auto *lattice = solver.lookupState<ExecutionContextLattice>(
      solver.getProgramPointBefore(op));
  if (!lattice || !lattice->isInitialized() ||
      !isUniformContext(lattice->getValue()))
    return false;
  // Divergence between the blocks of one region is not carried by the lattice
  // (it ends where the paths meet again), so the op's own block is checked
  // here; enclosing ops had theirs checked when their regions were entered.
  Block *block = op->getBlock();
  if (!block || block->getParent()->hasOneBlock())
    return true;
  auto info =
      UniformityValueAnalysis::computeCFGControlInfo(*block->getParent());
  return isBlockReachedUniformly(block, *info, [&](Value condition) {
    return isUniformValue(condition, op, solver);
  });
}