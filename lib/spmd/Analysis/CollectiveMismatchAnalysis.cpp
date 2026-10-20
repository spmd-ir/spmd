// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/CollectiveMismatchAnalysis.h"
#include "spmd/Analysis/HandleFacts.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#include <optional>

#define DEBUG_TYPE "collective-mismatch-analysis"

using namespace mlir;
using namespace mlir::spmd;

namespace {

static bool getIsUniqueMatching(Operation *opr) {
  if (auto attr = opr->getAttrOfType<BoolAttr>("isUniqueMatching"))
    return attr.getValue();
  return false;
}

static int getModelGroup(Operation *op) {
  if (auto a = op->getAttrOfType<IntegerAttr>("usedModel"))
    return static_cast<int>(a.getInt());
  return -1;
}

//===----------------------------------------------------------------------===//
// Interface helpers
//===----------------------------------------------------------------------===//

static bool isLoopLike(RegionBranchOpInterface rbOp) {
  if (isa<LoopLikeOpInterface>(rbOp.getOperation()))
    return true;
  for (Region &region : rbOp->getRegions()) {
    SmallVector<RegionSuccessor> successors;
    rbOp.getSuccessorRegions(region, successors);
    for (const RegionSuccessor &succ : successors)
      if (!succ.isParent())
        return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// Attribute helpers
//===----------------------------------------------------------------------===//

static bool getIsBlocking(Operation *opr) {
  if (auto attr = opr->getAttrOfType<BoolAttr>("isBlocking"))
    return attr.getValue();
  return true;
}

//===----------------------------------------------------------------------===//
// Divergence helpers (kind-aware)
//===----------------------------------------------------------------------===//

// All process-level models partition the same physical ranks; all thread-level
// models share the same thread team. A collective of a given kind is therefore
// desynchronised by control-flow divergence at *any* active level of that kind,
// and never by divergence of the opposite kind. `wantProcessLevel` selects the
// kind. Returns true iff the branch condition is non-uniform at some active
// level of that kind. (CollectiveMismatchAnalysis runs against a frozen solver,
// so lookupState is correct here.)
//
// Takes a plain Operation* rather than a RegionBranchOpInterface because the
// divergence that splits the ranks does not have to come from a region-carrying
// op: `getConditionOperands` also answers for an `arith.select`, which is what
// an if-converted fork looks like (see `checkCommValueFork`). For an op that
// carries no condition at all it returns {} and this reports false, which is
// what the structural forks (sections/single/critical) already relied on.
static bool isConditionNonUniformAtKind(DataFlowSolver &solver, Operation *op,
                                        bool wantProcessLevel) {
  if (!op)
    return false;
  auto *ctxLattice = solver.lookupState<ExecutionContextLattice>(
      solver.getProgramPointBefore(op));
  if (!ctxLattice || !ctxLattice->isInitialized())
    return false;
  const ExecutionContext &ctx = ctxLattice->getValue();
  if (ctx.levels.empty())
    return false;

  for (Value value : getConditionOperands(op)) {
    auto *lattice = solver.lookupState<UniformityLattice>(value);
    assert(
        lattice && lattice->isInitialized() &&
        "UniformityLattice not defined or uninitialized for condition value");
    for (const LevelInfo &l : ctx.levels) {
      if (l.isProcessLevel != wantProcessLevel)
        continue;
      if (lattice->getUniformityAtLevel(l.parallelOp).isNonUniform())
        return true;
    }
  }
  return false;
}

// Resolve a collective's binding level — the innermost active level whose
// modelGroup matches the op's usedModel — and report whether it is process- or
// thread-level. nullopt when no matching level is active (treated
// conservatively by the gate, i.e. checked under both kinds, and never dropped
// at a parallel boundary).
static std::optional<bool> collectiveKindIsProcess(DataFlowSolver &solver,
                                                   Operation *coll) {
  int m = getModelGroup(coll);
  auto *ctxLattice = solver.lookupState<ExecutionContextLattice>(
      solver.getProgramPointBefore(coll));
  if (!ctxLattice || !ctxLattice->isInitialized())
    return std::nullopt;
  const ExecutionContext &ctx = ctxLattice->getValue();
  for (const LevelInfo &l : llvm::reverse(ctx.levels))
    if (l.modelGroup == m)
      return l.isProcessLevel;
  return std::nullopt;
}

//===----------------------------------------------------------------------===//
// Loop trip-count helper
//===----------------------------------------------------------------------===//

static bool calculateNumIterations(scf::ForOp forOp, int64_t *out) {
  auto lb = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
  auto ub = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
  auto step = forOp.getStep().getDefiningOp<arith::ConstantOp>();
  if (!lb || !ub || !step)
    return false;
  int64_t lbV = cast<IntegerAttr>(lb.getValue()).getInt();
  int64_t ubV = cast<IntegerAttr>(ub.getValue()).getInt();
  int64_t stepV = cast<IntegerAttr>(step.getValue()).getInt();
  if (stepV == 0)
    return false;
  *out = (ubV - lbV + stepV - 1) / stepV;
  return true;
}

//===----------------------------------------------------------------------===//
// Block CFG utility
//===----------------------------------------------------------------------===//

static void
buildRPO(Block *entry, SmallVectorImpl<Block *> &rpoOut,
         llvm::DenseSet<std::pair<Block *, Block *>> &backEdgesOut) {
  SmallVector<std::pair<Block *, Block::succ_iterator>> stack;
  llvm::DenseSet<Block *> onStack, visited;

  auto push = [&](Block *b) {
    if (visited.insert(b).second) {
      onStack.insert(b);
      stack.push_back({b, b->succ_begin()});
    }
  };
  push(entry);

  while (!stack.empty()) {
    auto &[block, succIt] = stack.back();
    if (succIt == block->succ_end()) {
      rpoOut.push_back(block);
      onStack.erase(block);
      stack.pop_back();
      continue;
    }
    Block *succ = *succIt++;
    if (onStack.count(succ))
      backEdgesOut.insert({block, succ});
    else
      push(succ);
  }
  std::reverse(rpoOut.begin(), rpoOut.end());
}

//===----------------------------------------------------------------------===//
// Equivalence class helpers
//===----------------------------------------------------------------------===//

static SmallVector<llvm::DenseSet<HandleAtom>>
buildCommEquivalenceClasses(DataFlowSolver &solver, Operation *moduleOpr) {
  SmallVector<llvm::DenseSet<HandleAtom>> classes;
  llvm::DenseSet<HandleAtom> seenAtoms;

  // Rule 1: spmd.commWorld atoms grouped PER MODEL GROUP. commWorld values of
  // different models are distinct communicators (e.g. MPI world vs OMP/SHMEM
  // world), so they must not share an equivalence class — otherwise collectives
  // of different models would be positionally compared against one another.
  llvm::DenseMap<int, llvm::DenseSet<HandleAtom>> worldByModel;
  moduleOpr->walk([&](Operation *op) {
    if (op->getName().getStringRef() != "spmd.commWorld")
      return;
    int mg = getModelGroup(op);
    for (Value result : op->getResults()) {
      if (!isa<spmd::CommType>(result.getType()))
        continue;
      auto *lat = solver.lookupState<HandleFactsLattice>(result);
      assert(lat && !(lat->getValue().unknown) &&
             "CommOriginsAnalysis should have created the lattice states");
      for (HandleAtom atom : lat->getValue().atoms) {
        worldByModel[mg].insert(atom);
        seenAtoms.insert(atom);
      }
    }
  });
  // Deterministic order across model groups.
  SmallVector<int> mgs;
  for (auto &kv : worldByModel)
    mgs.push_back(kv.first);
  llvm::sort(mgs);
  for (int mg : mgs)
    if (!worldByModel[mg].empty())
      classes.push_back(std::move(worldByModel[mg]));

  // Fallback: every remaining atom is its own class.
  moduleOpr->walk([&](Operation *op) {
    for (Value result : op->getResults()) {
      if (!isa<spmd::CommType>(result.getType()))
        continue;
      auto *lat = solver.lookupState<HandleFactsLattice>(result);
      assert(lat && !(lat->getValue().unknown) &&
             "CommOriginsAnalysis should have created the lattice states");
      for (HandleAtom atom : lat->getValue().atoms)
        if (seenAtoms.insert(atom).second)
          classes.push_back({atom});
    }
  });
  return classes;
}

} // namespace

//===----------------------------------------------------------------------===//
// CollectiveMismatchAnalysis — comm resolution
//===----------------------------------------------------------------------===//

HandleFactsValue
CollectiveMismatchAnalysis::resolveCommFacts(Value val,
                                             Operation *contextOp) const {
  return resolveHandleFacts(solver, val, contextOp, HandleKind::Communicator);
}

bool CollectiveMismatchAnalysis::commValueInClass(
    Value commVal, Operation *contextOp,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  HandleFactsValue facts = resolveCommFacts(commVal, contextOp);
  if (facts.unknown)
    return true;
  for (HandleAtom atom : facts.atoms)
    if (commClass.count(atom))
      return true;
  return false;
}

bool CollectiveMismatchAnalysis::isSearchedComm(
    Operation *opr, const llvm::DenseSet<HandleAtom> &commClass) const {
  for (Value operand : opr->getOperands()) {
    if (!isa<spmd::CommType>(operand.getType()))
      continue;
    if (commValueInClass(operand, opr, commClass))
      return true;
  }
  return false;
}

//===----------------------------------------------------------------------===//
// CollectiveMismatchAnalysis — value-level forks
//===----------------------------------------------------------------------===//

namespace {
/// A fork that exists only as a value: some op picks between two communicators
/// and the collective consuming the pick carries no branch of its own.
struct CommValueFork {
  /// The op the condition is read off (`isConditionNonUniformAtKind`) and the
  /// one reported as the cause.
  Operation *forkOp;
  Value armA;
  Value armB;
};
} // namespace

/// Follow the handle-forwarding ops `resolveCommFacts` treats as transparent
/// and describe the value fork that ultimately produces `val`, if any.
static std::optional<CommValueFork> findCommValueFork(Value val) {
  // Bounded: the chain is a few casts at most, and the bound also guards
  // against a cycle through unusual IR.
  for (unsigned steps = 0; steps < 16; ++steps) {
    Operation *defOp = val.getDefiningOp();
    if (!defOp)
      return std::nullopt;

    if (auto sel = dyn_cast<arith::SelectOp>(defOp))
      return CommValueFork{sel, sel.getTrueValue(), sel.getFalseValue()};

    // `cond ? commA : commB` only collapses to an `arith.select` when both
    // arms are speculatable. Load one of them from memory — which is what an
    // out-parameter communicator like `MPI_Comm_split`'s looks like — and the
    // canonicalizer has to leave the `scf.if` standing, yielding the
    // communicator instead. Same fork, one indirection further out; the
    // regions themselves hold no collective, so `handleFork` compares two
    // empty sequences and finds nothing.
    if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
      if (ifOp.getThenRegion().empty() || ifOp.getElseRegion().empty())
        return std::nullopt;
      auto thenYield =
          dyn_cast_or_null<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
      auto elseYield =
          dyn_cast_or_null<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
      if (!thenYield || !elseYield)
        return std::nullopt;
      unsigned resIdx = cast<OpResult>(val).getResultNumber();
      if (resIdx >= thenYield.getNumOperands() ||
          resIdx >= elseYield.getNumOperands())
        return std::nullopt;
      return CommValueFork{ifOp, thenYield.getOperand(resIdx),
                           elseYield.getOperand(resIdx)};
    }

    StringRef name = defOp->getName().getStringRef();
    bool isTransparent = isa<UnrealizedConversionCastOp>(defOp) ||
                         name == "enzymexla.pointer2memref" ||
                         name == "enzymexla.memref2pointer";
    if (!isTransparent || defOp->getNumOperands() != 1)
      return std::nullopt;
    val = defOp->getOperand(0);
  }
  return std::nullopt;
}

void CollectiveMismatchAnalysis::checkCommValueFork(
    Operation *opr, const TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  for (Value operand : opr->getOperands()) {
    if (!isa<spmd::CommType>(operand.getType()))
      continue;
    std::optional<CommValueFork> fork = findCommValueFork(operand);
    if (!fork)
      continue;

    // One synthesized branch per arm. The record is dropped when that arm's
    // communicator is outside the class under test, which is what turns the
    // value fork into the [collective] vs [] comparison. An arm that does not
    // resolve counts as a member (`commValueInClass`), so both arms keep their
    // record and the comparison stays silent — the same false-negative bias
    // the checker already takes for an unresolvable communicator.
    auto seqForArm = [&](Value armComm) -> FlatSequence {
      FlatSequence seq;
      if (!commValueInClass(armComm, opr, commClass))
        return seq;
      seq.emplace_back(
          opr,
          CollectiveOpKey(opr->getName(), getIsBlocking(opr),
                          getIsUniqueMatching(opr), opr, getModelGroup(opr)),
          ctx.slotCounter, ctx.cumulativeIterations, getModelGroup(opr),
          collectiveKindIsProcess(solver, opr));
      return seq;
    };

    SmallVector<FlatSequence> armSeqs;
    armSeqs.push_back(seqForArm(fork->armA));
    armSeqs.push_back(seqForArm(fork->armB));
    if (armSeqs[0].empty() && armSeqs[1].empty())
      continue;

    checkBranchesByKind(armSeqs, fork->forkOp, ctx,
                        /*structuralThreadFork=*/false);
  }
}

//===----------------------------------------------------------------------===//
// CollectiveMismatchAnalysis — mismatch detection
//===----------------------------------------------------------------------===//

void CollectiveMismatchAnalysis::checkAndReportMismatch(
    ArrayRef<FlatSequence> branchSeqs, Operation *forkOp) const {
  size_t maxLen = 0;
  for (auto &seq : branchSeqs)
    maxLen = std::max(maxLen, seq.size());

  for (size_t pos = 0; pos < maxLen; pos++) {
    SmallVector<Operation *> conflicting;
    std::optional<CollectiveOpKey> refKey;
    std::optional<uint64_t> refCumIter;
    bool mismatch = false;

    for (auto &seq : branchSeqs) {
      if (pos >= seq.size()) {
        mismatch = true;
        continue;
      }
      const CollectiveRecord &rec = seq[pos];
      conflicting.push_back(rec.op);
      if (!refKey.has_value()) {
        refKey = rec.key;
        refCumIter = rec.cumulativeIterations;
      } else {
        if (!(*refKey == rec.key)) {
          mismatch = true;
        }
        if (*refCumIter != rec.cumulativeIterations) {
          mismatch = true;
        }
      }
    }

    if (mismatch && !conflicting.empty())
      errorInfos.emplace_back(conflicting, SmallVector<Operation *>{forkOp});
  }
}

// Per-kind, per-model gating wrapper around checkAndReportMismatch.
//
// Collectives are partitioned by modelGroup (so an MPI barrier only compares
// against an MPI barrier, etc.). Each partition is checked only if the fork
// actually splits the participants of that collective's kind:
//   * condition fork (scf.if/switch): non-uniform at a same-kind level, OR we
//     are already inside a same-kind divergent context;
//   * structural thread fork (single/critical/sections/wsloop/taskloop):
//     thread collectives are always split; process collectives only if
//     processes were already split.
void CollectiveMismatchAnalysis::checkBranchesByKind(
    ArrayRef<FlatSequence> branchSeqs, Operation *forkOp,
    const TraversalContext &ctx, bool structuralThreadFork) const {

  llvm::DenseMap<int, std::optional<bool>> kindByModel;
  SmallVector<int> models;
  for (const FlatSequence &seq : branchSeqs)
    for (const CollectiveRecord &rec : seq) {
      auto it = kindByModel.find(rec.modelGroup);
      if (it == kindByModel.end()) {
        kindByModel[rec.modelGroup] = rec.isProcessLevel;
        models.push_back(rec.modelGroup);
      } else if (!it->second.has_value()) {
        it->second = rec.isProcessLevel; // adopt a known kind if one appears
      }
    }

  auto procDiv = [&] {
    return isConditionNonUniformAtKind(solver, forkOp, true) ||
           ctx.inProcessDivergent;
  };
  auto threadDiv = [&] {
    return isConditionNonUniformAtKind(solver, forkOp, false) ||
           ctx.inThreadDivergent;
  };

  auto kindDivergent = [&](std::optional<bool> proc) -> bool {
    if (structuralThreadFork) {
      if (!proc.has_value())
        return true;                        // unknown kind → conservative
      return *proc ? ctx.inProcessDivergent // process: only if already split
                   : true;                  // thread: structurally split
    }
    if (!proc.has_value())
      return procDiv() || threadDiv();
    return *proc ? procDiv() : threadDiv();
  };

  for (int m : models) {
    if (crossModelOnly)
      break; // per-model comparisons already ran per class
    if (!kindDivergent(kindByModel[m]))
      continue;
    SmallVector<FlatSequence> mSeqs;
    mSeqs.reserve(branchSeqs.size());
    for (const FlatSequence &seq : branchSeqs) {
      FlatSequence f;
      for (const CollectiveRecord &rec : seq)
        if (rec.modelGroup == m)
          f.push_back(rec);
      mSeqs.push_back(std::move(f));
    }
    checkAndReportMismatch(mSeqs, forkOp);
  }

  // Cross-model ordering.
  //
  // Partitioning by model is right for *matching* (an MPI barrier must not be
  // paired with an NCCL one), but on its own it cannot see a mismatch that
  // exists only in the interleaving. A hybrid program where the branches run
  //     rank 0:  nccl reduce ; MPI bcast
  //     others:  MPI bcast   ; nccl reduce
  // is per-model consistent — every model's own sequence is [x] against [x] —
  // yet it deadlocks, because both models' collectives are executed by the SAME
  // processes and rank 0 blocks in the reduce while everyone else blocks in the
  // bcast. Ordering therefore has to be checked across models too.
  //
  // Grouping is by participant kind rather than by model: process-level
  // collectives share the physical ranks and thread-level ones share the thread
  // team, so ordering constrains each group internally but never across the two
  // (a thread barrier and a process barrier are not mutually ordered). Records
  // whose kind is unresolved are checked under both, conservatively. Only worth
  // running when more than one model is present — with a single model this is
  // exactly the per-model check above.
  //
  // Nonblocking collectives are deliberately NOT filtered out here, even though
  // merely *initiating* one cannot deadlock. Dropping them would make
  //     if (c) { MPI_Ibarrier(&r); nccl_allreduce(); MPI_Wait(&r); }
  //     else   { nccl_allreduce(); MPI_Ibarrier(&r); MPI_Wait(&r); }
  // pass — correctly, it is deadlock-free — but it would equally silence
  //     if (c) { MPI_Ibcast(&r); MPI_Wait(&r); nccl_allreduce(); }
  //     else   { nccl_allreduce(); MPI_Ibcast(&r); MPI_Wait(&r); }
  // which deadlocks: the first branch blocks in MPI_Wait until the second
  // initiates the bcast, while the second blocks in the allreduce waiting for
  // the first. Keeping the initiation in the sequence over-reports the first
  // shape (a false positive) rather than missing the second (a false negative),
  // which is the trade this checker wants. Modelling it exactly means anchoring
  // a nonblocking collective's ordering constraint at its completion
  // (wait/test) instead of its initiation — a request-tracking feature, not a
  // filter here.
  if (crossModelOnly && models.size() > 1) {
    for (bool procKind : {true, false}) {
      if (!kindDivergent(std::optional<bool>(procKind)))
        continue;
      SmallVector<FlatSequence> kindSeqs;
      kindSeqs.reserve(branchSeqs.size());
      bool anyRecord = false;
      for (const FlatSequence &seq : branchSeqs) {
        FlatSequence f;
        for (const CollectiveRecord &rec : seq)
          if (rec.isProcessLevel.value_or(procKind) == procKind) {
            f.push_back(rec);
            anyRecord = true;
          }
        kindSeqs.push_back(std::move(f));
      }
      if (anyRecord)
        checkAndReportMismatch(kindSeqs, forkOp);
    }
  }
}

FlatSequence
CollectiveMismatchAnalysis::mergeFlatSequences(ArrayRef<FlatSequence> seqs) {
  // Return a union of all branch sequences. Records are interleaved
  // position-by-position so that the parent sees the full set of possible
  // collectives at each execution position.
  FlatSequence merged;
  size_t maxLen = 0;
  for (auto &seq : seqs)
    maxLen = std::max(maxLen, seq.size());
  for (size_t i = 0; i < maxLen; i++)
    for (auto &seq : seqs)
      if (i < seq.size())
        merged.push_back(seq[i]);
  return merged;
}

//===----------------------------------------------------------------------===//
// CollectiveMismatchAnalysis — traversal
//===----------------------------------------------------------------------===//

FlatSequence CollectiveMismatchAnalysis::traverseRegion(
    Region &region, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  if (region.empty())
    return {};
  if (region.hasOneBlock())
    return traverseSingleBlock(region.front(), ctx, commClass);
  return traverseCFRegion(region, ctx, commClass);
}

FlatSequence CollectiveMismatchAnalysis::traverseSingleBlock(
    Block &block, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  FlatSequence result;

  for (Operation &oprRef : block) {
    Operation *opr = &oprRef;

    // ── Collective ────────────────────────────────────────────────────
    if (opr->hasTrait<spmd::Collective>() && isSearchedComm(opr, commClass)) {
      // An if-converted fork has no region to descend into, so it is compared
      // here rather than in handleFork.
      checkCommValueFork(opr, ctx, commClass);

      bool uniqueMatch = getIsUniqueMatching(opr);
      result.emplace_back(opr,
                          CollectiveOpKey(opr->getName(), getIsBlocking(opr),
                                          uniqueMatch, opr, getModelGroup(opr)),
                          ctx.slotCounter++, ctx.cumulativeIterations,
                          getModelGroup(opr),
                          collectiveKindIsProcess(solver, opr));
      // continue; commented out for ops like parallelOp that are collective and
      // have a region
    }

    // ── Call ─────────────────────────────────────────────────────────
    if (auto call = dyn_cast<CallOpInterface>(opr)) {
      FlatSequence callSeq = handleCall(call, ctx, commClass);
      for (auto &r : callSeq)
        result.push_back(r);
      continue;
    }

    // ── Worksharing loop / taskloop: iteration-space thread partition ──
    // Checked BEFORE the RegionBranchOpInterface dispatch so it wins even if
    // wsloop/taskloop happen to be loop-like.
    if (isa<spmd::WsloopOp, spmd::TaskloopOp>(opr)) {
      FlatSequence wsSeq = handleWorksharingLoop(opr, ctx, commClass);
      for (auto &r : wsSeq)
        result.push_back(r);
      continue;
    }

    // ── Structured control flow ───────────────────────────────────────
    if (auto rbOp = dyn_cast<RegionBranchOpInterface>(opr)) {
      FlatSequence rbSeq = isLoopLike(rbOp) ? handleLoop(rbOp, ctx, commClass)
                                            : handleFork(rbOp, ctx, commClass);
      for (auto &r : rbSeq)
        result.push_back(r);
      continue;
    }

    // ── spmd.critical: serialises ranks → implicit fork ───────────────
    if (isa<spmd::CriticalOp>(opr)) {
      FlatSequence critSeq = handleCritical(opr, ctx, commClass);
      for (auto &r : critSeq)
        result.push_back(r);
      continue;
    }

    // ── spmd.single: one rank runs the body, others skip ─────────────
    if (isa<spmd::SingleOp>(opr)) {
      FlatSequence singleSeq = handleSingle(opr, ctx, commClass);
      for (auto &r : singleSeq)
        result.push_back(r);
      continue;
    }

    // spmd.sections: concurrent fork across N section bodies.
    if (isa<spmd::SectionsOp>(opr)) {
      FlatSequence sectSeq = handleSections(opr, ctx, commClass);
      for (auto &r : sectSeq)
        result.push_back(r);
      continue;
    }

    // ── spmd.parallel: new parallelism scope ──────────────────────────
    // Recurse with a kind-aware divergence reset / boundary (handleParallel).
    if (isa<spmd::ParallelOp>(opr)) {
      FlatSequence parSeq = handleParallel(opr, ctx, commClass);
      for (auto &r : parSeq)
        result.push_back(r);
      continue;
    }

    // ── Other ops with regions (NOT RegionBranchOpInterface) ────────────
    // e.g. spmd.loop_nest, spmd.declare_reduction. Transparent wrappers:
    // recurse into every region without forking.
    if (opr->getNumRegions() > 0) {
      for (Region &region : opr->getRegions()) {
        FlatSequence regionSeq = traverseRegion(region, ctx, commClass);
        for (auto &r : regionSeq)
          result.push_back(r);
      }
      continue;
    }
  }

  return result;
}

FlatSequence CollectiveMismatchAnalysis::handleParallel(
    Operation *parallelOp, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  TraversalContext childCtx = ctx;

  bool isThreadParallel = false;
  if (auto par = dyn_cast<spmd::ParallelOp>(parallelOp))
    isThreadParallel = !par.getIsProcessLevel();

  // A thread-level parallel spawns a fresh, full thread team, so any thread-
  // level divergence accumulated outside does NOT carry into the new team.
  // A process-level parallel is only a region marker over the same physical
  // ranks, so process-level divergence MUST persist.
  if (isThreadParallel)
    childCtx.inThreadDivergent = false;

  FlatSequence body;
  for (Region &region : parallelOp->getRegions()) {
    FlatSequence rSeq = traverseRegion(region, childCtx, commClass);
    for (auto &rec : rSeq)
      body.push_back(rec);
  }
  ctx.slotCounter = childCtx.slotCounter;

  // Boundary: a thread-level parallel is a fresh team. Thread-kind collectives
  // inside it are self-contained and are checked within this recursion (by
  // inner forks / worksharing loops). They must NOT propagate to an enclosing
  // thread fork (e.g. an outer wsloop or single), which would otherwise compare
  // them against an empty branch and report a spurious mismatch. Process-kind
  // (and kind-unknown) collectives still propagate — same physical ranks.
  if (!isThreadParallel)
    return body;

  FlatSequence forwarded;
  forwarded.reserve(body.size());
  for (CollectiveRecord &rec : body)
    if (rec.isProcessLevel.value_or(true)) // keep process-kind and unknown
      forwarded.push_back(rec);
  return forwarded;
}

FlatSequence CollectiveMismatchAnalysis::handleWorksharingLoop(
    Operation *wsOp, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  // spmd.wsloop / spmd.taskloop: the iteration space is partitioned across the
  // thread team, so the body is NOT executed uniformly by all threads — a
  // thread may run zero iterations. This is structurally a thread fork, exactly
  // like single/critical/sections: model it as [body], [empty] so any thread-
  // level collective inside (e.g. an OMP barrier, including one reached through
  // a call) is reported as reached by some threads but not others.
  //
  // No SCF-loop semantics are applied: spmd.loop_nest is not an scf.for and is
  // never trip-count folded, so the body is traversed once as the partitioned
  // unit. taskloop is routed here too; structuralThreadFork already checks
  // thread-kind unconditionally, which is the maximally-conservative model its
  // dynamic scheduling requires.
  TraversalContext bodyCtx = ctx;
  bodyCtx.inThreadDivergent = true; // gate any genuine forks nested in the body

  FlatSequence bodySeq;
  for (Region &region : wsOp->getRegions()) {
    FlatSequence rSeq = traverseRegion(region, bodyCtx, commClass);
    for (auto &rec : rSeq)
      bodySeq.push_back(rec);
  }

  // Branch 0: a thread that runs ≥1 iteration. Branch 1: a thread that runs 0.
  SmallVector<FlatSequence> branches = {std::move(bodySeq), {}};

  checkBranchesByKind(branches, wsOp, ctx, /*structuralThreadFork=*/true);

  ctx.slotCounter = bodyCtx.slotCounter;
  return mergeFlatSequences(branches);
}

FlatSequence CollectiveMismatchAnalysis::handleFork(
    RegionBranchOpInterface forkOp, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  SmallVector<RegionSuccessor> entrySuccessors;
  forkOp.getSuccessorRegions(RegionBranchPoint::parent(), entrySuccessors);

  size_t baseSlot = ctx.slotCounter;
  SmallVector<FlatSequence> branchSeqs;
  SmallVector<size_t> branchEndSlots;

  // Per-kind divergence introduced by THIS fork's condition (independent of
  // branch contents — so we can mark the descent context before traversing).
  bool forkProcDiv =
      isConditionNonUniformAtKind(solver, forkOp.getOperation(), true);
  bool forkThreadDiv =
      isConditionNonUniformAtKind(solver, forkOp.getOperation(), false);

  for (const RegionSuccessor &succ : entrySuccessors) {
    if (succ.isParent()) {
      // Bypass branch (e.g. missing else) — zero collectives.
      branchSeqs.push_back({});
      branchEndSlots.push_back(baseSlot);
      continue;
    }

    TraversalContext branchCtx = ctx;
    branchCtx.slotCounter = baseSlot;
    branchCtx.inProcessDivergent = ctx.inProcessDivergent || forkProcDiv;
    branchCtx.inThreadDivergent = ctx.inThreadDivergent || forkThreadDiv;

    FlatSequence branchSeq =
        traverseRegion(*succ.getSuccessor(), branchCtx, commClass);

    branchEndSlots.push_back(branchCtx.slotCounter);
    branchSeqs.push_back(std::move(branchSeq));
  }

  // Per-kind, per-model mismatch comparison.
  checkBranchesByKind(branchSeqs, forkOp.getOperation(), ctx,
                      /*structuralThreadFork=*/false);

  // Advance the slot counter past the longest branch.
  ctx.slotCounter =
      *std::max_element(branchEndSlots.begin(), branchEndSlots.end());

  // Give the parent the union of all branch sequences.
  return mergeFlatSequences(branchSeqs);
}

FlatSequence CollectiveMismatchAnalysis::handleSections(
    Operation *sectionsOp, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {

  size_t baseSlot = ctx.slotCounter;
  SmallVector<FlatSequence> branches;

  // Traverse each spmd.section child as an independent branch.
  // Sections execute concurrently — no ordering guarantee between them —
  // so this is structurally identical to a thread-level multivalued fork.
  for (Region &sectionsRegion : sectionsOp->getRegions()) {
    for (Operation &child : sectionsRegion.front()) {
      if (!isa<spmd::SectionOp>(&child))
        continue;

      TraversalContext sectionCtx = ctx;
      sectionCtx.slotCounter = baseSlot;
      sectionCtx.inThreadDivergent = true; // sections split the thread team

      FlatSequence sectionSeq;
      for (Region &sectionBody : child.getRegions()) {
        FlatSequence r = traverseRegion(sectionBody, sectionCtx, commClass);
        for (auto &rec : r)
          sectionSeq.push_back(rec);
      }
      branches.push_back(std::move(sectionSeq));
    }
  }

  // Conservative: if there are more threads than sections some threads execute
  // nothing — model that as an empty branch so mismatch is flagged when
  // only some sections contain collectives.
  branches.push_back({});

  // Structural thread fork: thread-level collectives are always compared;
  // process-level collectives only if processes were already split.
  checkBranchesByKind(branches, sectionsOp, ctx,
                      /*structuralThreadFork=*/true);

  size_t maxSlot = ctx.slotCounter;
  for (auto &branch : branches)
    maxSlot = std::max(maxSlot, baseSlot + branch.size());
  ctx.slotCounter = maxSlot;

  return mergeFlatSequences(branches);
}

FlatSequence CollectiveMismatchAnalysis::handleLoop(
    RegionBranchOpInterface loopOp, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {

  TraversalContext bodyCtx = ctx;

  // Fold constant trip counts into cumulativeIterations.
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp.getOperation())) {
    int64_t n = 0;
    if (calculateNumIterations(forOp, &n))
      bodyCtx.cumulativeIterations =
          ctx.cumulativeIterations * static_cast<uint64_t>(n);
  }
  // Non-constant loops: cumulativeIterations stays at ctx value (conservative).

  FlatSequence bodySeq;
  for (Region &r : loopOp->getRegions()) {
    FlatSequence rSeq = traverseRegion(r, bodyCtx, commClass);
    for (auto &rec : rSeq)
      bodySeq.push_back(rec);
  }

  ctx.slotCounter = bodyCtx.slotCounter;
  return bodySeq;
}

FlatSequence CollectiveMismatchAnalysis::handleCall(
    CallOpInterface call, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {

  auto sym = call.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym)
    return {};

  auto func = dyn_cast_or_null<FunctionOpInterface>(
      SymbolTable::lookupNearestSymbolFrom(call.getOperation(), sym));
  if (!func || func.isExternal() || func.getFunctionBody().empty())
    return {};

  // Recursion guard: bail if this function is already on the call stack.
  for (Operation *callOp : ctx.callStack) {
    auto prevCall = dyn_cast<CallOpInterface>(callOp);
    if (!prevCall)
      continue;
    auto prevSym = prevCall.getCallableForCallee().dyn_cast<SymbolRefAttr>();
    if (!prevSym)
      continue;
    auto prevFunc = dyn_cast_or_null<FunctionOpInterface>(
        SymbolTable::lookupNearestSymbolFrom(callOp, prevSym));
    if (prevFunc == func)
      return {};
  }

  TraversalContext calleeCtx = ctx;
  calleeCtx.callStack.push_back(call.getOperation());

  FlatSequence calleeSeq =
      traverseRegion(func.getFunctionBody(), calleeCtx, commClass);

  ctx.slotCounter = calleeCtx.slotCounter;
  return calleeSeq;
}

FlatSequence CollectiveMismatchAnalysis::traverseCFRegion(
    Region &region, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {
  if (region.empty())
    return {};

  SmallVector<Block *> rpo;
  llvm::DenseSet<std::pair<Block *, Block *>> backEdges;
  buildRPO(&region.front(), rpo, backEdges);

  struct BlockInfo {
    size_t slotCounter = 0;
    FlatSequence sequence;
    bool initialized = false;
  };

  llvm::DenseMap<Block *, BlockInfo> blockInfo;
  blockInfo[&region.front()].slotCounter = ctx.slotCounter;
  blockInfo[&region.front()].initialized = true;

  FlatSequence exitSeq;
  size_t exitSlot = ctx.slotCounter;
  bool exitSeen = false;

  for (Block *block : rpo) {
    auto &info = blockInfo[block];
    if (!info.initialized)
      continue;

    TraversalContext blockCtx = ctx;
    blockCtx.slotCounter = info.slotCounter;
    FlatSequence blockSeq = info.sequence;

    // Process ops in this block.
    for (Operation &oprRef : *block) {
      Operation *opr = &oprRef;

      if (opr->hasTrait<spmd::Collective>() && isSearchedComm(opr, commClass)) {
        checkCommValueFork(opr, blockCtx, commClass);

        bool uniqueMatch = getIsUniqueMatching(opr);
        blockSeq.emplace_back(
            opr,
            CollectiveOpKey(opr->getName(), getIsBlocking(opr), uniqueMatch,
                            opr, getModelGroup(opr)),
            blockCtx.slotCounter++, blockCtx.cumulativeIterations,
            getModelGroup(opr), collectiveKindIsProcess(solver, opr));
        // continue; commented out for ops like parallelOp that have a region
        // and are collective
      }

      if (auto call = dyn_cast<CallOpInterface>(opr)) {
        FlatSequence callSeq = handleCall(call, blockCtx, commClass);
        for (auto &r : callSeq)
          blockSeq.push_back(r);
        continue;
      }

      // Worksharing loop / taskloop (before the RBI dispatch).
      if (isa<spmd::WsloopOp, spmd::TaskloopOp>(opr)) {
        FlatSequence wsSeq = handleWorksharingLoop(opr, blockCtx, commClass);
        for (auto &r : wsSeq)
          blockSeq.push_back(r);
        continue;
      }

      if (auto rbOp = dyn_cast<RegionBranchOpInterface>(opr)) {
        FlatSequence rbSeq = isLoopLike(rbOp)
                                 ? handleLoop(rbOp, blockCtx, commClass)
                                 : handleFork(rbOp, blockCtx, commClass);
        for (auto &r : rbSeq)
          blockSeq.push_back(r);
        continue;
      }

      if (isa<spmd::CriticalOp>(opr)) {
        FlatSequence critSeq = handleCritical(opr, blockCtx, commClass);
        for (auto &r : critSeq)
          blockSeq.push_back(r);
        continue;
      }

      if (isa<spmd::SingleOp>(opr)) {
        FlatSequence singleSeq = handleSingle(opr, blockCtx, commClass);
        for (auto &r : singleSeq)
          blockSeq.push_back(r);
        continue;
      }

      // spmd.sections: concurrent fork across N section bodies.
      if (isa<spmd::SectionsOp>(opr)) {
        FlatSequence sectSeq = handleSections(opr, blockCtx, commClass);
        for (auto &r : sectSeq)
          blockSeq.push_back(r);
        continue;
      }

      // spmd.parallel: new parallelism scope (kind-aware boundary/reset).
      if (isa<spmd::ParallelOp>(opr)) {
        FlatSequence parSeq = handleParallel(opr, blockCtx, commClass);
        for (auto &r : parSeq)
          blockSeq.push_back(r);
        continue;
      }

      // ── Other ops with regions that are NOT RegionBranchOpInterface ───
      if (opr->getNumRegions() > 0) {
        for (Region &region : opr->getRegions()) {
          FlatSequence regionSeq = traverseRegion(region, blockCtx, commClass);
          for (auto &r : regionSeq)
            blockSeq.push_back(r);
        }
        continue;
      }
    }

    // Propagate through the block terminator.
    Operation *terminator = block->getTerminator();
    if (!terminator)
      continue;

    if (terminator->getNumSuccessors() == 0) {
      // Exit block: merge into exit state.
      if (!exitSeen) {
        exitSeq = blockSeq;
        exitSlot = blockCtx.slotCounter;
        exitSeen = true;
      } else {
        exitSlot = std::max(exitSlot, blockCtx.slotCounter);
        for (auto &r : blockSeq)
          exitSeq.push_back(r);
      }
      continue;
    }

    if (!isa<BranchOpInterface>(terminator))
      continue;

    // For multivalued cf branches, fork the state to each non-back-edge
    // successor and let RPO join at merge blocks. Full sequence comparison for
    // cf forks would require dominator-tree analysis to find join points —
    // left as a conservative approximation.
    for (Block *succ : terminator->getSuccessors()) {
      if (backEdges.count({block, succ}))
        continue;
      auto &succInfo = blockInfo[succ];
      if (!succInfo.initialized) {
        succInfo.slotCounter = blockCtx.slotCounter;
        succInfo.sequence = blockSeq;
        succInfo.initialized = true;
      } else {
        // Join point: conservative merge.
        succInfo.slotCounter =
            std::max(succInfo.slotCounter, blockCtx.slotCounter);
        for (auto &r : blockSeq)
          succInfo.sequence.push_back(r);
      }
    }
  }

  ctx.slotCounter = exitSlot;
  return exitSeq;
}

FlatSequence CollectiveMismatchAnalysis::handleSingle(
    Operation *singleOp, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {

  // Branch 0: the one thread that executes the single body. Mark the thread
  // team as divergent so nested forks inside the body are treated divergently
  // at the thread level (process divergence is inherited from ctx, not added).
  TraversalContext singleCtx = ctx;
  singleCtx.inThreadDivergent = true;

  FlatSequence singleSeq;
  for (Region &region : singleOp->getRegions()) {
    FlatSequence r = traverseRegion(region, singleCtx, commClass);
    for (auto &rec : r)
      singleSeq.push_back(rec);
  }

  // Branch 1: all other threads — they skip the body entirely.
  SmallVector<FlatSequence> branches = {std::move(singleSeq), {}};

  // Structural thread fork: the fork is inherent to the construct.
  checkBranchesByKind(branches, singleOp, ctx, /*structuralThreadFork=*/true);

  ctx.slotCounter = singleCtx.slotCounter;
  return mergeFlatSequences(branches);
}

FlatSequence CollectiveMismatchAnalysis::handleCritical(
    Operation *critOp, TraversalContext &ctx,
    const llvm::DenseSet<HandleAtom> &commClass) const {

  // Branch 0: the one thread currently holding the lock and executing the body.
  TraversalContext critCtx = ctx;
  critCtx.inThreadDivergent = true;

  FlatSequence critSeq;
  for (Region &region : critOp->getRegions()) {
    FlatSequence r = traverseRegion(region, critCtx, commClass);
    for (auto &rec : r)
      critSeq.push_back(rec);
  }

  // Branch 1: all other threads blocked at the mutex — no collectives visible.
  SmallVector<FlatSequence> branches = {std::move(critSeq), {}};

  // Structural thread fork.
  checkBranchesByKind(branches, critOp, ctx, /*structuralThreadFork=*/true);

  ctx.slotCounter = critCtx.slotCounter;
  return mergeFlatSequences(branches);
}

//===----------------------------------------------------------------------===//
// CollectiveMismatchAnalysis — entry point
//===----------------------------------------------------------------------===//

LogicalResult CollectiveMismatchAnalysis::run(Operation *top) {
  auto moduleOp = dyn_cast<ModuleOp>(top);
  if (!moduleOp)
    return success();
  Operation *mainOpr = moduleOp.lookupSymbol("main");
  if (!mainOpr)
    return success();

  auto equivClasses = buildCommEquivalenceClasses(solver, moduleOp);

  for (const llvm::DenseSet<HandleAtom> &commClass : equivClasses) {
    TraversalContext ctx;
    // Traverse the main function's region. All mismatch reporting happens
    // on-the-fly inside handleFork / handle* as branches are compared.
    traverseRegion(mainOpr->getRegion(0), ctx, commClass);
  }

  // Extra traversal for hybrid programs: the per-class traversals above each
  // see only one communicator's collectives, so a mismatch that exists purely
  // in how two models' collectives interleave is invisible to all of them.
  // Traverse once more over the union of every class — every collective is
  // recorded — and let checkBranchesByKind run only its cross-model comparison.
  llvm::DenseSet<int> modelGroups;
  moduleOp->walk([&](Operation *op) {
    if (op->getName().getStringRef() == "spmd.commWorld")
      modelGroups.insert(getModelGroup(op));
  });
  if (modelGroups.size() > 1) {
    llvm::DenseSet<HandleAtom> allAtoms;
    for (const llvm::DenseSet<HandleAtom> &commClass : equivClasses)
      for (HandleAtom atom : commClass)
        allAtoms.insert(atom);

    crossModelOnly = true;
    TraversalContext ctx;
    traverseRegion(mainOpr->getRegion(0), ctx, allAtoms);
    crossModelOnly = false;
  }

  // The traversals overlap, so the same conflict can be reported more than
  // once. Keep the first occurrence of each.
  SmallVector<CollectiveMismatchErrorInfo> deduped;
  for (const CollectiveMismatchErrorInfo &err : errorInfos) {
    bool seen = false;
    for (const CollectiveMismatchErrorInfo &kept : deduped) {
      if (kept.conflictOprs == err.conflictOprs &&
          kept.causingOprs == err.causingOprs) {
        seen = true;
        break;
      }
    }
    if (!seen)
      deduped.push_back(err);
  }
  errorInfos = std::move(deduped);

  return success();
}