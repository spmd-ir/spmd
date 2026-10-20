// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/UniformityAnalysis.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/MemoryOffsets.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "uniformity-analysis"

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// Anonymous-namespace helpers
//===----------------------------------------------------------------------===//

namespace {

std::optional<SmallVector<Value>>
collectLoopBoundsAndStep(LoopLikeOpInterface loop) {
  SmallVector<Value> boundsAndStep;
  Operation *loopOp = loop;

  if (auto affineLoop = dyn_cast<affine::AffineForOp>(loopOp)) {
    boundsAndStep.append(affineLoop.getLowerBoundOperands().begin(),
                         affineLoop.getLowerBoundOperands().end());
    boundsAndStep.append(affineLoop.getUpperBoundOperands().begin(),
                         affineLoop.getUpperBoundOperands().end());
    return boundsAndStep;
  }

  auto lb = loop.getSingleLowerBound();
  auto ub = loop.getSingleUpperBound();
  auto step = loop.getSingleStep();
  if (!lb || !ub || !step)
    return std::nullopt;

  auto addIfValue = [&](OpFoldResult ofr) {
    if (auto v = llvm::dyn_cast_if_present<Value>(ofr))
      boundsAndStep.push_back(v);
  };
  addIfValue(*lb);
  addIfValue(*ub);
  addIfValue(*step);

  return boundsAndStep;
}

//===----------------------------------------------------------------------===//
// cf helpers
//===----------------------------------------------------------------------===//

Value getBranchControllingCondition(Operation *branchOp) {
  if (auto cb = dyn_cast<cf::CondBranchOp>(branchOp))
    return cb.getCondition();
  if (auto sw = dyn_cast<cf::SwitchOp>(branchOp))
    return sw.getFlag();
  return Value();
}

bool isConditionalBranch(Operation *op) {
  return isa_and_nonnull<cf::CondBranchOp, cf::SwitchOp>(op);
}

void computeSuccessorReachables(
    Operation *branch, SmallVectorImpl<llvm::DenseSet<Block *>> &reachables) {
  reachables.assign(branch->getNumSuccessors(), {});
  for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
    SmallVector<Block *> worklist;
    worklist.push_back(branch->getSuccessor(i));
    while (!worklist.empty()) {
      Block *b = worklist.pop_back_val();
      if (!reachables[i].insert(b).second)
        continue;
      for (Block *s : b->getSuccessors())
        worklist.push_back(s);
    }
  }
}
} // anonymous namespace

UniformityValueAnalysis::~UniformityValueAnalysis() = default;

/*static*/
std::unique_ptr<UniformityValueAnalysis::CFGControlInfo>
UniformityValueAnalysis::computeCFGControlInfo(Region &region) {
  auto info = std::make_unique<CFGControlInfo>();
  if (region.getBlocks().size() <= 1)
    return info;

  PostDominanceInfo PDI;

  for (Block &A : region) {
    Operation *T = A.getTerminator();
    if (!isConditionalBranch(T))
      continue;

    SmallVector<llvm::DenseSet<Block *>, 4> reachables;
    computeSuccessorReachables(T, reachables);

    llvm::DenseMap<Block *, unsigned> counts;
    for (auto &set : reachables)
      for (Block *B : set)
        ++counts[B];

    for (auto &kv : counts) {
      Block *B = kv.first;
      unsigned count = kv.second;

      if (!PDI.postDominates(B, &A))
        info->varyingControlBranches[B].push_back(T);

      if (count >= 2)
        info->syncDependentBranches[B].push_back(T);
    }
  }

  return info;
}

UniformityValueAnalysis::CFGControlInfo &
UniformityValueAnalysis::getCFGControlInfo(Region *region) {
  auto it = cfgCache.find(region);
  if (it != cfgCache.end())
    return *it->second;
  auto info = computeCFGControlInfo(*region);
  auto inserted = cfgCache.try_emplace(region, std::move(info));
  return *inserted.first->second;
}

//===----------------------------------------------------------------------===//
// Uniformity / printing
//===----------------------------------------------------------------------===//

raw_ostream &operator<<(raw_ostream &os, const Uniformity &uniformity) {
  if (uniformity.isUninitialized())
    return os << "<UNINITIALIZED>";
  switch (uniformity.getKind()) {
  case Uniformity::Kind::Unknown:
    return os << "unknown";
  case Uniformity::Kind::Uniform:
    return os << "uniform";
  case Uniformity::Kind::NonUniform:
    return os << "non-uniform";
  }
  llvm_unreachable("Unhandled Uniformity::Kind");
}

void Uniformity::print(raw_ostream &os) const { os << *this; }

//===----------------------------------------------------------------------===//
// UniformityLattice
//===----------------------------------------------------------------------===//

ChangeResult UniformityLattice::joinAtLevel(Operation *parallelOp,
                                            const Uniformity &u) {
  assert(initialized && "joinAtLevel called on uninitialized lattice");
  // Uniform is the structural default (missing entry). No need to store it.
  if (u.isUniform() || u.isUninitialized())
    return ChangeResult::NoChange;

  auto [it, inserted] = perLevel.try_emplace(parallelOp, u);
  if (inserted)
    return ChangeResult::Change;

  Uniformity joined = Uniformity::join(it->second, u);
  if (joined == it->second)
    return ChangeResult::NoChange;
  it->second = joined;
  return ChangeResult::Change;
}

ChangeResult UniformityLattice::markInitialized() {
  if (initialized)
    return ChangeResult::NoChange;
  initialized = true;
  return ChangeResult::Change;
}

ChangeResult UniformityLattice::join(const AbstractSparseLattice &other_) {
  const auto &other = static_cast<const UniformityLattice &>(other_);
  if (!other.initialized)
    return ChangeResult::NoChange;

  ChangeResult result = markInitialized();
  for (auto &[op, u] : other.perLevel)
    result |= joinAtLevel(op, u);
  return result;
}

void UniformityLattice::onUpdate(DataFlowSolver *solver) const {
  dataflow::AbstractSparseLattice::onUpdate(solver);

  // For values defined by a RegionBranchOpInterface op (e.g. scf.if results),
  // join with the branch op's operand uniformities at every parallel level so
  // that varying-control taints results hierarchically.
  Value value = cast<Value>(anchor);
  if (Operation *defOp = value.getDefiningOp()) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(defOp)) {
      auto *self = solver->getOrCreateState<UniformityLattice>(value);
      for (Value operand : branch->getOperands()) {
        auto *operandLattice = solver->lookupState<UniformityLattice>(operand);
        if (!operandLattice || !operandLattice->isInitialized())
          continue;
        solver->propagateIfChanged(self, self->join(*operandLattice));
      }
    }
  }

  // Keep the constant-value lattice ticking so the framework visits nested
  // regions (preserved from the previous design — needed for SCF result
  // forwarding to fire in some configurations).
  auto *constantValue =
      solver->getOrCreateState<dataflow::Lattice<dataflow::ConstantValue>>(
          value);
  solver->propagateIfChanged(
      constantValue,
      constantValue->join(dataflow::ConstantValue::getUnknownConstant()));
}

void UniformityLattice::print(raw_ostream &os) const {
  if (!initialized) {
    os << "<uninitialized>";
    return;
  }
  if (perLevel.empty()) {
    os << "uniform(all-levels)";
    return;
  }
  os << "Uniformity{";
  bool first = true;
  for (auto &[op, u] : perLevel) {
    if (!first)
      os << ", ";
    os << op << "->" << u;
    first = false;
  }
  os << "}";
}

//===----------------------------------------------------------------------===//
// UniformityMemoryState — dense, per program point, N-level
//===----------------------------------------------------------------------===//

ChangeResult UniformityMemoryState::join(const AbstractDenseLattice &other_) {
  const auto &other = static_cast<const UniformityMemoryState &>(other_);
  ChangeResult result = ChangeResult::NoChange;

  if (other.allTop && !allTop) {
    allTop = true;
    perLevel.clear();
    return ChangeResult::Change;
  }
  if (allTop)
    return result;

  for (auto &[level, otherLS] : other.perLevel) {
    LevelState &myLS = perLevel[level];

    // Join NonUniform slots: they subsume Unknown, so evict from unknown.
    for (const Slot &slot : otherLS.nonUniformSlots) {
      myLS.unknownSlots.erase(slot);
      if (myLS.nonUniformSlots.insert(slot).second)
        result = ChangeResult::Change;
      myLS.unknownClasses.erase(slot.first);
      myLS.nonUniformClasses.insert(slot.first);
    }
    // Join Unknown slots: only if not already at NonUniform.
    for (const Slot &slot : otherLS.unknownSlots) {
      if (myLS.nonUniformSlots.contains(slot))
        continue; // NonUniform subsumes Unknown.
      if (myLS.unknownSlots.insert(slot).second)
        result = ChangeResult::Change;
      if (!myLS.nonUniformClasses.contains(slot.first))
        myLS.unknownClasses.insert(slot.first);
    }
  }
  return result;
}

ChangeResult UniformityMemoryState::mark(DistinctAttr cls, Uniformity u,
                                         Operation *level, int64_t offset) {
  if (allTop || !cls || u.isUniform() || u.isUninitialized())
    return ChangeResult::NoChange;

  LevelState &ls = perLevel[level];
  Slot slot{cls, offset};

  if (u.isNonUniform()) {
    // NonUniform subsumes Unknown: evict from unknown if present.
    bool removedUnknown = ls.unknownSlots.erase(slot) > 0;
    bool insertedNU = ls.nonUniformSlots.insert(slot).second;
    ls.nonUniformClasses.insert(cls);
    if (!ls.unknownSlots.empty()) {
      // The class may still hold Unknown slots at other offsets; only drop it
      // from the Unknown summary once none are left.
      bool anyUnknownLeft = false;
      for (const Slot &s : ls.unknownSlots)
        if (s.first == cls) {
          anyUnknownLeft = true;
          break;
        }
      if (!anyUnknownLeft)
        ls.unknownClasses.erase(cls);
    } else {
      ls.unknownClasses.erase(cls);
    }
    return (removedUnknown || insertedNU) ? ChangeResult::Change
                                          : ChangeResult::NoChange;
  }

  // Unknown: only store if this slot is not already at NonUniform.
  if (ls.nonUniformSlots.contains(slot))
    return ChangeResult::NoChange;
  bool inserted = ls.unknownSlots.insert(slot).second;
  if (!ls.nonUniformClasses.contains(cls))
    ls.unknownClasses.insert(cls);
  return inserted ? ChangeResult::Change : ChangeResult::NoChange;
}

ChangeResult UniformityMemoryState::mark(const enzyme::AliasClassSet &classes,
                                         Uniformity u, Operation *level,
                                         int64_t offset) {
  if (allTop)
    return ChangeResult::NoChange;
  if (classes.isUndefined())
    return ChangeResult::NoChange;

  // Unknown alias set: we cannot bound which memory is written at any level,
  // so conservatively mark all levels and all classes as NonUniform (allTop).
  if (classes.isUnknown())
    return markAllTop();
  if (u.isUniform() || u.isUninitialized())
    return ChangeResult::NoChange;

  ChangeResult result = ChangeResult::NoChange;
  for (DistinctAttr cls : classes.getElements())
    result |= mark(cls, u, level, offset);
  return result;
}

ChangeResult UniformityMemoryState::markAllTop() {
  if (allTop)
    return ChangeResult::NoChange;
  allTop = true;
  perLevel.clear(); // Subsumed by allTop.
  return ChangeResult::Change;
}

Uniformity UniformityMemoryState::getUniformity(DistinctAttr cls,
                                                Operation *level,
                                                int64_t offset) const {
  if (allTop)
    return Uniformity::getNonUniform();
  if (!cls)
    return Uniformity::getUnknown();

  auto it = perLevel.find(level);
  if (it == perLevel.end())
    return Uniformity::getUniform(); // Missing level key ⟹ Uniform.

  const LevelState &ls = it->second;

  // A read whose offset is not statically known may observe any write to the
  // object, so it consults the per-class summaries.
  if (offset == kWholeObject) {
    if (ls.nonUniformClasses.contains(cls))
      return Uniformity::getNonUniform();
    if (ls.unknownClasses.contains(cls))
      return Uniformity::getUnknown();
    return Uniformity::getUniform();
  }

  // A read at a known offset sees writes to that same offset, plus any write
  // that could not be pinned to one (recorded under kWholeObject).
  Slot exact{cls, offset};
  Slot whole{cls, kWholeObject};
  if (ls.nonUniformSlots.contains(exact) || ls.nonUniformSlots.contains(whole))
    return Uniformity::getNonUniform();
  if (ls.unknownSlots.contains(exact) || ls.unknownSlots.contains(whole))
    return Uniformity::getUnknown();
  return Uniformity::getUniform();
}

Uniformity
UniformityMemoryState::getUniformity(const enzyme::AliasClassSet &classes,
                                     Operation *level, int64_t offset) const {
  if (allTop)
    return Uniformity::getNonUniform();
  if (classes.isUndefined())
    return Uniformity::getUniform();
  if (classes.isUnknown())
    return Uniformity::getNonUniform();

  Uniformity result = Uniformity::getUniform();
  for (DistinctAttr cls : classes.getElements()) {
    result = Uniformity::join(result, getUniformity(cls, level, offset));
    if (result.isNonUniform())
      break; // Can't get worse.
  }
  return result;
}

void UniformityMemoryState::print(raw_ostream &os) const {
  os << "UniformityMemoryState{";
  if (allTop) {
    os << "allTop";
  } else if (perLevel.empty()) {
    os << "uniform(all-levels)";
  } else {
    bool firstLevel = true;
    for (auto &[level, ls] : perLevel) {
      if (!firstLevel)
        os << "; ";
      auto printSlots = [&](const llvm::DenseSet<Slot> &slots) {
        bool first = true;
        for (const Slot &slot : slots) {
          if (!first)
            os << ", ";
          os << slot.first << "@";
          if (slot.second == kWholeObject)
            os << "*";
          else
            os << slot.second;
          first = false;
        }
      };
      os << "level=" << level << ":{nonUniform=[";
      printSlots(ls.nonUniformSlots);
      os << "], unknown=[";
      printSlots(ls.unknownSlots);
      os << "]}";
      firstLevel = false;
    }
  }
  os << "}";
}

//===----------------------------------------------------------------------===//
// UniformityValueAnalysis — sparse forward, N-level
//===----------------------------------------------------------------------===//

void UniformityValueAnalysis::setToEntryState(UniformityLattice *lattice) {
  // GPU kernel arguments: all explicitly uniform (empty map = Uniform
  // everywhere is the optimistic starting point; seeds will refine).
  if (Region *region = lattice->getAnchor().getParentRegion()) {
    if (auto gpuFuncOp = dyn_cast<gpu::GPUFuncOp>(region->getParentOp())) {
      if (gpuFuncOp.isKernel()) {
        for (Value arg : gpuFuncOp.front().getArguments()) {
          auto *argLattice = getLatticeElement(arg);
          propagateIfChanged(argLattice, argLattice->markInitialized());
        }
        return;
      }
    }
  }
  // Default: mark initialized with an empty map (Uniform at all levels).
  propagateIfChanged(lattice, lattice->markInitialized());
}

void UniformityValueAnalysis::applySyncDependenceTaint(Operation *firstOp) {
  Block *parent = firstOp->getBlock();
  if (!parent || parent->getNumArguments() == 0)
    return;
  Region *region = parent->getParent();
  if (!region || region->getBlocks().size() <= 1)
    return;

  auto &cfgInfo = getCFGControlInfo(region);
  auto it = cfgInfo.syncDependentBranches.find(parent);
  if (it == cfgInfo.syncDependentBranches.end())
    return;

  const auto *ctxState = getOrCreateFor<ParallelContextState>(
      getProgramPointAfter(firstOp), getProgramPointBefore(firstOp));
  if (!ctxState || !ctxState->isInitialized())
    return;

  // For each level and each sync-dependent branch, taint block args when
  // the controlling condition is non-uniform at that level.
  for (Operation *level : ctxState->getStack()) {
    for (Operation *brancher : it->second) {
      Value cond = getBranchControllingCondition(brancher);
      if (!cond)
        continue;

      const auto *condLat = getOrCreateFor<UniformityLattice>(
          getProgramPointAfter(firstOp), cond);
      if (!condLat || !condLat->isInitialized())
        continue;

      const Uniformity u = condLat->getUniformityAtLevel(level);
      if (u.isUniform())
        continue;

      Uniformity taint = u.isNonUniform() ? Uniformity::getNonUniform()
                                          : Uniformity::getUnknown();
      for (BlockArgument arg : parent->getArguments()) {
        auto *argLat = getLatticeElement(arg);
        propagateIfChanged(argLat, argLat->markInitialized());
        propagateIfChanged(argLat, argLat->joinAtLevel(level, taint));
      }
    }
  }
}

void UniformityValueAnalysis::applyVaryingControlTaintAtLevel(
    Operation *op, ArrayRef<UniformityLattice *> results, Operation *level) {
  Block *parent = op->getBlock();
  if (!parent)
    return;
  Region *region = parent->getParent();
  if (!region || region->getBlocks().size() <= 1)
    return;

  auto &cfgInfo = getCFGControlInfo(region);
  auto it = cfgInfo.varyingControlBranches.find(parent);
  if (it == cfgInfo.varyingControlBranches.end())
    return;

  Uniformity worst = Uniformity::getUniform();
  for (Operation *brancher : it->second) {
    Value cond = getBranchControllingCondition(brancher);
    if (!cond)
      continue;
    const auto *condLat =
        getOrCreateFor<UniformityLattice>(getProgramPointAfter(op), cond);
    if (!condLat || !condLat->isInitialized())
      continue;
    worst = Uniformity::join(worst, condLat->getUniformityAtLevel(level));
    if (worst.isNonUniform())
      break;
  }

  if (worst.isUniform())
    return;
  propagateAllAtLevel(results, level, worst);
}

LogicalResult UniformityValueAnalysis::visitOperation(
    Operation *op, ArrayRef<const UniformityLattice *> operands,
    ArrayRef<UniformityLattice *> results) {
  LLVM_DEBUG(llvm::dbgs() << "UA: Visiting operation: " << *op << "\n");

  // cf sync-dependence taint at the first op of each block (per-level).
  if (Block *parent = op->getBlock())
    if (op == &parent->front())
      applySyncDependenceTaint(op);

  // Obtain the active parallel levels from ParallelContextAnalysis. Defer
  // if the context is not yet available.
  const auto *ctxState = getOrCreateFor<ParallelContextState>(
      getProgramPointAfter(op), getProgramPointBefore(op));
  if (!ctxState || !ctxState->isInitialized()) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Parallel context not yet ready\n");
    return success();
  }
  ArrayRef<Operation *> stack = ctxState->getStack();

  // 1. Trait-based seed: NonUniform at every active parallel level.
  if (op->hasTrait<spmd::ResultsNonUniform>()) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Non-uniform results trait\n");
    for (UniformityLattice *result : results) {
      propagateIfChanged(result, result->markInitialized());
      for (Operation *level : stack)
        propagateIfChanged(
            result, result->joinAtLevel(level, Uniformity::getNonUniform()));
    }
    return success();
  }

  // Defer until all operand lattices are initialized.
  if (llvm::any_of(operands, [](const UniformityLattice *l) {
        return !l->isInitialized();
      })) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Operand(s) not yet initialized\n");
    return success();
  }

  // Mark all results as initialized (empty map = Uniform at all levels).
  for (UniformityLattice *result : results)
    propagateIfChanged(result, result->markInitialized());

  // 2. For each active parallel level, compute and propagate independently.
  //    Missing entry in an operand's perLevel map ⟹ Uniform at that level
  //    (value was defined before the parallel region was entered).
  for (Operation *level : stack) {
    // Join operand uniformities at this level.
    Uniformity joined = Uniformity::getUniform();
    for (const UniformityLattice *l : operands)
      joined = Uniformity::join(joined, l->getUniformityAtLevel(level));

    // 3. Memory-effect-free → uniformity follows operands only.
    if (isMemoryEffectFree(op)) {
      propagateAllAtLevel(results, level, joined);
      applyVaryingControlTaintAtLevel(op, results, level);
      continue;
    }

    // 4. Operands already NonUniform at this level — no need to inspect memory.
    if (joined.isNonUniform()) {
      propagateAllAtLevel(results, level, Uniformity::getNonUniform());
      applyVaryingControlTaintAtLevel(op, results, level);
      continue;
    }

    // 5. Memory effects: classify reads against the per-level dense memory
    //    state, passing `level` so that loads from shared memory inside a
    //    thread-level parallel are correctly classified as thread-uniform.
    analyzeMemoryEffectsAtLevel(op, operands, results, level);
    applyVaryingControlTaintAtLevel(op, results, level);
  }

  return success();
}

void UniformityValueAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ValueRange nonSuccessorInputs, ArrayRef<UniformityLattice *> argLattices) {
  LLVM_DEBUG(llvm::dbgs() << "UA: Visiting non-ctrl-flow args of: " << *op
                          << "\n");

  // Any operand still uninitialized? Defer.
  if (llvm::any_of(op->getOperands(), [&](Value v) {
        auto *lat = solver.getOrCreateState<UniformityLattice>(v);
        return !lat->isInitialized();
      }))
    return;

  const auto *ctxState = getOrCreateFor<ParallelContextState>(
      getProgramPointAfter(op), getProgramPointBefore(op));
  if (!ctxState || !ctxState->isInitialized())
    return;

  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    if (auto iv = loop.getSingleInductionVar()) {
      auto bs = collectLoopBoundsAndStep(loop);
      if (bs) {
        auto *ivLat = getLatticeElement(*iv);
        propagateIfChanged(ivLat, ivLat->markInitialized());
        for (Operation *level : ctxState->getStack()) {
          Uniformity ivU = Uniformity::getUniform();
          for (Value v : *bs) {
            auto *lat = solver.getOrCreateState<UniformityLattice>(v);
            if (!lat->isInitialized()) {
              ivU = Uniformity::getUnknown();
              break;
            }
            ivU = Uniformity::join(ivU, lat->getUniformityAtLevel(level));
          }
          propagateIfChanged(ivLat, ivLat->joinAtLevel(level, ivU));
        }
      }
    }
  }

  Base::visitNonControlFlowArguments(op, successor, nonSuccessorInputs,
                                     argLattices);
}

void UniformityValueAnalysis::analyzeMemoryEffectsAtLevel(
    Operation *op, ArrayRef<const UniformityLattice *> operands,
    ArrayRef<UniformityLattice *> results, Operation *level) {
  auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectOp) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Unknown memory effects\n");
    propagateAllAtLevel(results, level, Uniformity::getUnknown());
    return;
  }

  if (hasSingleEffect<MemoryEffects::Allocate>(memEffectOp)) {
    LLVM_DEBUG(llvm::dbgs().indent(2) << "Allocate effect → uniform\n");
    // Allocations produce a fresh pointer; uniform at every level.
    return;
  }

  SmallVector<MemoryEffects::EffectInstance> effects;
  memEffectOp.getEffects(effects);

  // Classify read effects against the per-level memory state at `level`.
  // A load from shared memory inside a thread-level parallel correctly returns
  // Uniform at the thread level when the stored data was thread-uniform, even
  // if the same alias class is process-nonuniform (stored as NonUniform at the
  // outer process level).
  Uniformity readU = Uniformity::getUniform();
  bool hasUnclassifiedRead = false;
  for (auto &effect : effects) {
    if (!isa<MemoryEffects::Read>(effect.getEffect()))
      continue;
    Value addr = effect.getValue();
    if (!addr) {
      // Symmetrically to the write side: a read of a non-addressable resource
      // reads no program data, so it says nothing about the result's
      // uniformity. (Ops whose result really is rank-dependent — spmd.
      // getRankInComm and friends — carry the ResultsNonUniform trait and are
      // seeded from there, not from this effect.)
      if (effect.getResource() && !effect.getResource()->isAddressable())
        continue;
      hasUnclassifiedRead = true;
      readU = Uniformity::join(readU, Uniformity::getUnknown());
      continue;
    }
    auto cls = classifyReadEffect(op, addr, level);
    if (!cls)
      return; // alias info not ready — defer entire op
    readU = Uniformity::join(readU, *cls);
    if (readU.isNonUniform())
      break;
  }

  if (hasUnclassifiedRead && readU.isUniform())
    readU = Uniformity::getUnknown();

  // Combine read uniformity with operand uniformity at this level.
  Uniformity opU = Uniformity::getUniform();
  for (const UniformityLattice *l : operands)
    opU = Uniformity::join(opU, l->getUniformityAtLevel(level));

  propagateAllAtLevel(results, level, Uniformity::join(readU, opU));
}

const DataLayout &UniformityValueAnalysis::getDataLayout(Operation *op) {
  if (!cachedDataLayout)
    cachedDataLayout = std::make_unique<DataLayout>(DataLayout::closest(op));
  return *cachedDataLayout;
}

const DataLayout &UniformityMemoryAnalysis::getDataLayout(Operation *op) {
  if (!cachedDataLayout)
    cachedDataLayout = std::make_unique<DataLayout>(DataLayout::closest(op));
  return *cachedDataLayout;
}

std::optional<Uniformity>
UniformityValueAnalysis::classifyReadEffect(Operation *op, Value addr,
                                            Operation *level) {
  // Query alias classes from the frozen alias solver.
  const auto *aliasState = getAliasState(addr);
  if (!aliasState)
    return std::nullopt;
  if (aliasState->isUndefined())
    return std::nullopt;

  // Query the per-level dense memory state before this op at `level`.
  const auto *memState = getOrCreateFor<UniformityMemoryState>(
      getProgramPointAfter(op), getProgramPointBefore(op));
  if (!memState)
    return std::nullopt;

  // Restrict the query to the byte this read actually observes when that is
  // statically known, so a read of one struct field does not pick up a
  // non-uniform write to a sibling field of the same allocation.
  int64_t offset = accessByteOffset(op, addr, getDataLayout(op))
                       .value_or(UniformityMemoryState::kWholeObject);

  return memState->getUniformity(aliasState->getAliasClassesObject(), level,
                                 offset);
}

void UniformityValueAnalysis::propagateAllAtLevel(
    ArrayRef<UniformityLattice *> results, Operation *level,
    const Uniformity &uniformity) {
  for (UniformityLattice *result : results)
    propagateIfChanged(result, result->joinAtLevel(level, uniformity));
}

void UniformityValueAnalysis::propagateAllAtLevel(
    ValueRange values, Operation *level, const Uniformity &uniformity) {
  for (Value value : values) {
    auto *lattice = getLatticeElement(value);
    propagateIfChanged(lattice, lattice->joinAtLevel(level, uniformity));
  }
}

//===----------------------------------------------------------------------===//
// UniformityMemoryAnalysis — dense forward, N-level
//===----------------------------------------------------------------------===//

void UniformityMemoryAnalysis::setToEntryState(UniformityMemoryState *lattice) {
  // Default: empty perLevel map → all alias classes Uniform at every level.
  propagateIfChanged(lattice, ChangeResult::NoChange);
}

Uniformity UniformityMemoryAnalysis::classifyStoredValue(Operation *op,
                                                         Value addr,
                                                         Operation *level) {
  // The stored value's uniformity at `level` is the join of every non-address
  // operand's uniformity at `level`. Conservative for ops with metadata
  // operands; replace per-dialect with a StoreLikeOpInterface if needed.
  Uniformity u = Uniformity::getUniform();
  for (Value operand : op->getOperands()) {
    if (operand == addr)
      continue;
    const auto *lat =
        getOrCreateFor<UniformityLattice>(getProgramPointAfter(op), operand);
    if (!lat || !lat->isInitialized()) {
      // Defer-ish: Unknown is sound, dependency is registered so we re-fire.
      u = Uniformity::join(u, Uniformity::getUnknown());
      continue;
    }
    u = Uniformity::join(u, lat->getUniformityAtLevel(level));
    if (u.isNonUniform())
      break;
  }
  return u;
}

LogicalResult
UniformityMemoryAnalysis::visitOperation(Operation *op,
                                         const UniformityMemoryState &before,
                                         UniformityMemoryState *after) {

  ChangeResult result = after->join(before);

  // Region ops: the framework recurses into nested regions.
  if (op->getNumRegions() > 0) {
    propagateIfChanged(after, result);
    return success();
  }

  auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!memEffectOp) {
    // Non-call ops without effect interface: be conservative.
    if (!isa<CallOpInterface>(op))
      result |= after->markAllTop();
    propagateIfChanged(after, result);
    return success();
  }

  // Collect all active parallel levels. We classify stored value uniformity
  // at every level independently so that the per-level memory state reflects
  // exactly which levels see non-uniform data in each alias class.
  const auto *ctxState = getOrCreateFor<ParallelContextState>(
      getProgramPointAfter(op), getProgramPointBefore(op));
  ArrayRef<Operation *> stack = {};
  if (ctxState && ctxState->isInitialized())
    stack = ctxState->getStack();

  SmallVector<MemoryEffects::EffectInstance> effects;
  memEffectOp.getEffects(effects);

  for (auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;

    Value addr = effect.getValue();
    if (!addr) {
      // A valueless effect on a NON-addressable resource does not touch
      // program memory — it names an abstract namespace used only to order
      // operations against each other (MLIR forbids attaching a Value to such
      // a resource at all, see EffectInstance::checkResourceAllowsValue).
      // `spmd.init`/`spmd.finalize` declare exactly such a write on
      // RuntimeResource, so treating it as an opaque store used to set allTop
      // near the top of every SPMD program and pin every subsequent load to
      // NonUniform — silently disabling the whole memory side of this analysis.
      // Only a valueless effect on an addressable resource is a genuinely
      // unknown store.
      if (effect.getResource() && !effect.getResource()->isAddressable())
        continue;
      result |= after->markAllTop();
      continue;
    }

    const auto *aliasState =
        aliasSolver.lookupState<enzyme::AliasClassLattice>(addr);
    if (!aliasState || aliasState->isUndefined()) {
      // Unknown alias set: conservatively pessimize everything.
      result |= after->markAllTop();
      continue;
    }

    // Byte written, when statically known; otherwise the whole object.
    int64_t offset = accessByteOffset(op, addr, getDataLayout(op))
                         .value_or(UniformityMemoryState::kWholeObject);

    if (stack.empty()) {
      // Outside all parallel regions: record under nullptr level.
      // This keeps the API uniform and lets queries at the nullptr level work.
      Uniformity storedU = classifyStoredValue(op, addr, /*level=*/nullptr);
      if (!storedU.isUniform())
        result |= after->mark(aliasState->getAliasClassesObject(), storedU,
                              /*level=*/nullptr, offset);
    } else {
      // For each active parallel level, classify the stored value's uniformity
      // at that level and record it independently. This is the key change from
      // the flat design: a store that writes process-nonuniform but
      // thread-uniform data will set NonUniform at the process level and leave
      // the thread level as Uniform (missing entry), so a subsequent load at
      // the thread level correctly returns Uniform.
      for (Operation *level : stack) {
        Uniformity storedU = classifyStoredValue(op, addr, level);
        if (!storedU.isUniform())
          result |= after->mark(aliasState->getAliasClassesObject(), storedU,
                                level, offset);
      }
    }
  }

  propagateIfChanged(after, result);
  return success();
}

void UniformityMemoryAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const UniformityMemoryState &before, UniformityMemoryState *after) {
  ChangeResult result = after->join(before);
  switch (action) {
  case dataflow::CallControlFlowAction::EnterCallee:
  case dataflow::CallControlFlowAction::ExitCallee:
    break;
  case dataflow::CallControlFlowAction::ExternalCallee: {
    // An external callee is opaque, so by default assume it stores anywhere.
    // The exception is a callee that receives no pointer-like operand at all
    // (`malloc(8)`, `shmem_my_pe()`, ...): it was handed no way to name any of
    // the caller's memory, so it cannot store into it. Without this, the malloc
    // that allocates a struct pessimizes every later read of that struct's
    // fields to NonUniform.
    bool mayNameCallerMemory = false;
    for (Value arg : call.getArgOperands()) {
      Type t = arg.getType();
      if (isa<LLVM::LLVMPointerType, MemRefType>(t) || isa<BaseMemRefType>(t)) {
        mayNameCallerMemory = true;
        break;
      }
    }
    if (mayNameCallerMemory)
      result |= after->markAllTop();
    break;
  }
  }
  propagateIfChanged(after, result);
}

//===----------------------------------------------------------------------===//
// Public queries
//===----------------------------------------------------------------------===//

Uniformity getUniformityAtLevel(Value value, Operation *parallelOp,
                                DataFlowSolver &solver) {
  const auto *lattice = solver.lookupState<UniformityLattice>(value);
  if (!lattice || !lattice->isInitialized())
    return Uniformity::getUninitialized();
  return lattice->getUniformityAtLevel(parallelOp);
}

namespace {

static Operation *innermostEnclosingParallelOp(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp())
    if (isa<spmd::ParallelOp>(parent))
      return parent;
  return nullptr;
}

static Uniformity lookupUniformityAtLevel(DataFlowSolver &solver, Value value,
                                          Operation *level) {
  const auto *lattice = solver.lookupState<UniformityLattice>(value);
  if (!lattice || !lattice->isInitialized())
    return Uniformity::getUniform(); // optimistic default for isDivergent
  return lattice->getUniformityAtLevel(level);
}

} // anonymous namespace

bool isDivergent(Operation *op, DataFlowSolver &solver) {
  assert(op && op->getBlock() && "Expecting a valid operation in a block");

  Operation *level = innermostEnclosingParallelOp(op);

  // scf path: walk enclosing region branches (scf.if / scf.for / scf.while).
  bool scfDivergent =
      llvm::any_of(getEnclosingRegionBranches(op), [&](Operation *branchOp) {
        return llvm::any_of(getConditionOperands(branchOp), [&](Value v) {
          const Uniformity u = lookupUniformityAtLevel(solver, v, level);
          assert(!u.isUninitialized() &&
                 "lattice element should be initialized after analysis");
          return !u.isUniform();
        });
      });
  if (scfDivergent)
    return true;

  // cf path: ad-hoc compute side-table for this op's region.
  Region *region = op->getParentRegion();
  if (!region || region->getBlocks().size() <= 1)
    return false;

  auto cfgInfo = UniformityValueAnalysis::computeCFGControlInfo(*region);
  auto it = cfgInfo->varyingControlBranches.find(op->getBlock());
  if (it == cfgInfo->varyingControlBranches.end())
    return false;

  for (Operation *brancher : it->second) {
    Value cond = getBranchControllingCondition(brancher);
    if (!cond)
      continue;
    const Uniformity u = lookupUniformityAtLevel(solver, cond, level);
    if (!u.isUninitialized() && !u.isUniform())
      return true;
  }
  return false;
}

} // namespace mlir::spmd