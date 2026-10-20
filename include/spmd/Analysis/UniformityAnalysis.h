/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- UniformityAnalysis.h - Uniformity Analysis --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// SPMD uniformity analysis. Replaces the previous UniformityAnalysis and
// ReachingDefinitionAnalysis pair with two cooperating analyses that share
// one solver (the "uniformity solver"); alias info still comes from a
// separate, frozen alias solver as before.
//
// Architecture:
//
//   Solver 1 (alias, frozen):
//     enzyme::AliasAnalysis              (sparse, value -> alias classes)
//     enzyme::PointsToPointerAnalysis    (dense, alias class -> pointed-to)
//
//   Solver 2 (uniformity, joint fixed point):
//     DeadCodeAnalysis
//     SparseConstantPropagation
//     ParallelContextAnalysis            (dense, prog-point -> ParallelOp*
//     stack) UniformityValueAnalysis            (sparse, value -> per-level
//     Uniformity) UniformityMemoryAnalysis           (dense,  alias class ->
//     per-level Uniformity)
//
// Lattice (value) — hierarchical, N-level:
//   Each SSA value carries a DenseMap<Operation*, Uniformity> keyed by the
//   enclosing spmd::ParallelOp* for each active parallel level.  A missing
//   entry means Uniform — the structural default for any level at which the
//   value was defined outside the corresponding parallel body.  Only
//   NonUniform and Unknown entries are stored explicitly.
//
//   Three states per level: Uniform, NonUniform, Unknown.  Join of unequal
//   definite states with one being NonUniform yields NonUniform; otherwise
//   Unknown.  This matches the existing semantics — preserved so downstream
//   passes don't have to change their per-level queries.
//
// Lattice (memory):
//   Per-program-point, per-parallel-level:
//     DenseMap<Operation* /*parallelOp*/,
//              {nonUniformSlots: DenseSet<(DistinctAttr, byte offset)>,
//               unknownSlots:    DenseSet<(DistinctAttr, byte offset)>}>
//   plus a global allTop bit for fully-opaque stores.
//
//   The offset makes the state field-sensitive: writing a non-uniform rank to
//   `H->mype` no longer taints a later read of `H->nproc`, which share an alias
//   class because they share an allocation. An address whose offset within its
//   allocation is not statically known uses the kWholeObject sentinel and
//   behaves exactly as before — tainting (or observing) the entire object.
//
//   A missing level key means all alias classes are Uniform at that level —
//   matching the same missing-entry-means-Uniform convention as the value
//   lattice.  Weak updates only.
//
//   With this representation, a load from shared memory inside a thread-level
//   parallel is correctly classified as thread-uniform when the stored value
//   was thread-uniform, even if it is process-nonuniform.
//
// Control flow:
//   * scf.if/for/while: region results tainted via UniformityLattice's
//     onUpdate hook.  Loop ivs tainted in visitNonControlFlowArguments.
//     Both operations iterate over all active parallel levels.
//   * cf.br/cond_br/switch: per-Region CFGControlInfo side-table built
//     from PostDominanceInfo.  Varying-control and sync-dependence taints
//     are applied per-level so a branch that is non-uniform at the thread
//     level does not incorrectly taint values at the process level.
//
// Communication-aware seeds:
//   Ops with the spmd::ResultsNonUniform trait seed NonUniform at ALL active
//   parallel levels unconditionally.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_SPMD_ANALYSIS_UNIFORMITYANALYSIS_H
#define MLIR_DIALECT_SPMD_ANALYSIS_UNIFORMITYANALYSIS_H

#include "Enzyme/MLIR/Analysis/DataFlowAliasAnalysis.h"
#include "spmd/Analysis/ParallelContextAnalysis.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// Uniformity (per-level lattice element)
//===----------------------------------------------------------------------===//

class Uniformity {
  friend raw_ostream &operator<<(raw_ostream &, const Uniformity &);

public:
  enum class Kind { Unknown, Uniform, NonUniform };

  Uniformity() = default;
  explicit Uniformity(Kind kind) : kind(kind) {}

  bool operator==(const Uniformity &rhs) const { return kind == rhs.kind; }
  bool operator!=(const Uniformity &rhs) const { return !(*this == rhs); }

  bool isUninitialized() const { return !kind.has_value(); }

  Kind getKind() const {
    assert(!isUninitialized() && "Uniformity is not initialized");
    return *kind;
  }

  static bool isUnknown(Kind k) { return k == Kind::Unknown; }
  static bool isUniform(Kind k) { return k == Kind::Uniform; }
  static bool isNonUniform(Kind k) { return k == Kind::NonUniform; }

  bool isUnknown() const { return isUnknown(getKind()); }
  bool isUniform() const { return isUniform(getKind()); }
  bool isNonUniform() const { return isNonUniform(getKind()); }

  static Uniformity getUninitialized() { return {}; }
  static Uniformity getUnknown() { return Uniformity(Kind::Unknown); }
  static Uniformity getUniform() { return Uniformity(Kind::Uniform); }
  static Uniformity getNonUniform() { return Uniformity(Kind::NonUniform); }

  static Uniformity join(const Uniformity &lhs, const Uniformity &rhs) {
    if (lhs.isUninitialized())
      return rhs;
    if (rhs.isUninitialized())
      return lhs;
    if (lhs == rhs)
      return lhs;
    if (lhs.isNonUniform() || rhs.isNonUniform())
      return getNonUniform();
    return getUnknown();
  }

  void print(raw_ostream &os) const;

private:
  std::optional<Kind> kind;
};

raw_ostream &operator<<(raw_ostream &os, const Uniformity &uniformity);

//===----------------------------------------------------------------------===//
// UniformityLattice — sparse, per SSA value, N-level
//
// Holds a DenseMap<Operation*, Uniformity> keyed by enclosing ParallelOp*.
// A missing entry means Uniform — the structural default for any parallel
// level at which the value was defined before the corresponding parallel
// body was entered.  Only NonUniform and Unknown entries are stored
// explicitly to keep the representation compact.
//
// An uninitialized lattice (isInitialized() == false) means the analysis has
// not yet visited the defining operation; downstream analyses should defer.
//
// The onUpdate hook taints RegionBranchOpInterface results by joining the
// branch operands' per-level uniformities, preserving the scf.if / scf.for
// varying-control mechanism at every parallel level.
//===----------------------------------------------------------------------===//

class UniformityLattice : public dataflow::AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  bool isInitialized() const { return initialized; }

  /// Query uniformity at a specific parallel level.
  /// `parallelOp` is the spmd::ParallelOp* whose executor set is of interest.
  /// Returns Uniform for any level not present in the map (structural default:
  /// value was defined before that parallel region was entered).
  /// Returns Uninitialized if the lattice has not been computed yet.
  Uniformity getUniformityAtLevel(Operation *parallelOp) const {
    if (!initialized)
      return Uniformity::getUninitialized();
    auto it = perLevel.find(parallelOp);
    return (it != perLevel.end()) ? it->second : Uniformity::getUniform();
  }

  /// Join a specific level's uniformity with `u`.
  /// Uniform entries are not stored (missing == Uniform is the convention),
  /// so passing Uniform is a no-op.
  /// Asserts that the lattice is already initialized.
  ChangeResult joinAtLevel(Operation *parallelOp, const Uniformity &u);

  /// Mark this lattice as initialized (empty map ⟹ Uniform at all levels).
  /// Used to signal that the defining op has been visited and no level has
  /// been shown to diverge yet.
  ChangeResult markInitialized();

  /// Join all per-level entries from another UniformityLattice.
  ChangeResult join(const AbstractSparseLattice &other) override;

  /// Hook: when a value defined by a RegionBranchOpInterface op updates,
  /// join with the branch op's operand uniformities at every parallel level
  /// so that varying-control taints results hierarchically.
  void onUpdate(DataFlowSolver *solver) const override;

  void print(raw_ostream &os) const override;

private:
  /// Per-level uniformity map.  Missing key ⟺ Uniform at that level.
  llvm::DenseMap<Operation *, Uniformity> perLevel;
  bool initialized = false;
};

//===----------------------------------------------------------------------===//
// UniformityMemoryState — dense, per program point, N-level
//
// Tracks, for each alias class and each active parallel level, the uniformity
// of the data currently stored in that class at that level.  Weak updates
// only — stores join into the per-class, per-level lattice, never remove.
//
// Representation:
//   perLevel : DenseMap<Operation* /*parallelOp*/,
//                       LevelState {nonUniformClasses, unknownClasses}>
//
// Missing level key  ⟺  all alias classes are Uniform at that level.
// Missing class key within a LevelState  ⟺  that class is Uniform.
// allTop  ⟺  every class at every level is at least NonUniform (global
//            conservative fallback for fully-opaque stores or unknown
//            alias sets).
//
// With this representation, a load inside a thread-level parallel from
// memory stored uniformly at the thread level is correctly classified as
// thread-uniform, even if the same alias class is process-nonuniform.
//===----------------------------------------------------------------------===//

class UniformityMemoryState : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  ChangeResult join(const AbstractDenseLattice &other) override;

  /// Byte offset within an alias class, used to keep distinct fields of one
  /// allocation apart. `kWholeObject` means "offset not statically known":
  /// as a write it taints the whole object, as a read it observes every write
  /// to the object. Both directions are the conservative choice, so an
  /// unanalysable address degrades exactly to the old whole-object behaviour.
  static constexpr int64_t kWholeObject = std::numeric_limits<int64_t>::min();

  /// Mark alias class `cls` at byte `offset` with uniformity `u` at parallel
  /// level `level`. `level == nullptr` means "outside all parallel regions".
  /// Uniform and Uninitialized are no-ops (missing entry == Uniform).
  ChangeResult mark(DistinctAttr cls, Uniformity u, Operation *level,
                    int64_t offset);

  /// Mark every class in `classes` at `offset` with `u` at parallel level
  /// `level`. If `classes` is unknown, conservatively promotes to allTop.
  ChangeResult mark(const enzyme::AliasClassSet &classes, Uniformity u,
                    Operation *level, int64_t offset);

  /// Conservative top: every alias class at every level is NonUniform.
  /// Triggered by fully-opaque writes (unknown address or unknown alias set).
  ChangeResult markAllTop();

  /// Read uniformity of a single alias class at `level`, restricted to byte
  /// `offset` (or the whole object for `kWholeObject`).
  /// Returns Uniform for levels or slots not in the map (optimistic default).
  Uniformity getUniformity(DistinctAttr cls, Operation *level,
                           int64_t offset) const;

  /// Join uniformity over a set of alias classes at `level` and `offset`.
  /// Unknown alias class set  → NonUniform (worst case).
  /// Undefined alias class set → Uniform  (no aliases yet, optimistic).
  Uniformity getUniformity(const enzyme::AliasClassSet &classes,
                           Operation *level, int64_t offset) const;

  bool isAllTop() const { return allTop; }

  void print(raw_ostream &os) const override;

private:
  /// (alias class, byte offset) — one field of one allocation.
  using Slot = std::pair<DistinctAttr, int64_t>;

  /// Per-level state for a single parallel level.
  /// Missing slot key within a LevelState means that slot is Uniform.
  struct LevelState {
    /// Slots for which a NonUniform write has been observed.
    llvm::DenseSet<Slot> nonUniformSlots;
    /// Slots for which an Unknown write has been observed (and not also
    /// in nonUniformSlots — NonUniform subsumes Unknown).
    llvm::DenseSet<Slot> unknownSlots;

    /// Per-class summaries of the two sets above: a class appears here iff any
    /// of its slots does. They answer a read at an unknown offset — which must
    /// see every write to the object — in O(1) instead of scanning the slots.
    llvm::DenseSet<DistinctAttr> nonUniformClasses;
    llvm::DenseSet<DistinctAttr> unknownClasses;

    bool empty() const {
      return nonUniformSlots.empty() && unknownSlots.empty();
    }
  };

  /// Per-level memory uniformity map.
  /// Key: spmd::ParallelOp* (or nullptr for code outside all parallel regions).
  llvm::DenseMap<Operation *, LevelState> perLevel;

  /// Global top: every alias class at every level is treated as NonUniform.
  bool allTop = false;
};

//===----------------------------------------------------------------------===//
// UniformityValueAnalysis — sparse forward, N-level
//===----------------------------------------------------------------------===//

class UniformityValueAnalysis
    : public dataflow::SparseForwardDataFlowAnalysis<UniformityLattice> {
public:
  using Base = dataflow::SparseForwardDataFlowAnalysis<UniformityLattice>;

  /// Construct with a reference to the alias solver. Alias state is queried
  /// in a frozen, read-only fashion via `aliasSolver.lookupState`.
  UniformityValueAnalysis(DataFlowSolver &solver, DataFlowSolver &aliasSolver)
      : Base(solver), solver(solver), aliasSolver(aliasSolver) {}

  ~UniformityValueAnalysis();

  void setToEntryState(UniformityLattice *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               ArrayRef<const UniformityLattice *> operands,
                               ArrayRef<UniformityLattice *> results) override;

  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ValueRange nonSuccessorInputs,
      ArrayRef<UniformityLattice *> argLattices) override;

  /// Read alias state from the frozen alias solver. Returns nullptr if not
  /// yet available.
  const enzyme::AliasClassLattice *getAliasState(Value v) const {
    return aliasSolver.lookupState<enzyme::AliasClassLattice>(v);
  }

  /// DataLayout covering `op`, built once and reused. Field-sensitive offsets
  /// need type sizes, and rebuilding a DataLayout per memory access is far too
  /// expensive to do inline.
  const DataLayout &getDataLayout(Operation *op);

  /// Per-Region cf control side-table.
  struct CFGControlInfo;

  static std::unique_ptr<CFGControlInfo> computeCFGControlInfo(Region &region);

private:
  CFGControlInfo &getCFGControlInfo(Region *region);

  /// Apply sync-dependence taint to block arguments at reconvergence points
  /// for all active parallel levels in the current ParallelContextState.
  void applySyncDependenceTaint(Operation *firstOp);

  /// Apply varying-control taint to `results` at a specific parallel `level`.
  void applyVaryingControlTaintAtLevel(Operation *op,
                                       ArrayRef<UniformityLattice *> results,
                                       Operation *level);

  /// Classify memory effects and propagate uniformity of `op`'s results at
  /// `level`. Queries the per-level UniformityMemoryState for read effects,
  /// using `level` as the key so that thread-uniform loads from shared memory
  /// are correctly classified even when the same memory is process-nonuniform.
  void analyzeMemoryEffectsAtLevel(Operation *op,
                                   ArrayRef<const UniformityLattice *> operands,
                                   ArrayRef<UniformityLattice *> results,
                                   Operation *level);

  /// Classify the uniformity of a read effect on `addr` at parallel `level`
  /// using the per-level dense memory state. Returns nullopt to signal
  /// "defer" (alias info or memory state not yet available).
  std::optional<Uniformity> classifyReadEffect(Operation *op, Value addr,
                                               Operation *level);

  /// Propagate `uniformity` at `level` into all result lattices.
  void propagateAllAtLevel(ArrayRef<UniformityLattice *> results,
                           Operation *level, const Uniformity &uniformity);

  /// Propagate `uniformity` at `level` into lattices for all `values`.
  void propagateAllAtLevel(ValueRange values, Operation *level,
                           const Uniformity &uniformity);

  llvm::DenseMap<Region *, std::unique_ptr<CFGControlInfo>> cfgCache;
  DataFlowSolver &solver;
  DataFlowSolver &aliasSolver;
  std::unique_ptr<DataLayout> cachedDataLayout;
};

//===----------------------------------------------------------------------===//
// CFGControlInfo
//===----------------------------------------------------------------------===//

struct UniformityValueAnalysis::CFGControlInfo {
  /// Block -> conditional terminators whose varying-condition implies this
  /// block is conditionally reached (some path from the branch reaches
  /// elsewhere). Values defined here / writes done here are conditionally
  /// executed.
  llvm::DenseMap<Block *, llvm::SmallVector<Operation *, 2>>
      varyingControlBranches;

  /// Block -> conditional terminators where ≥2 successors of the branch
  /// reach this block (it's a reconvergence / sync-dependence point). Block
  /// args are at a join and must be tainted when the upstream branch is
  /// varying — even if all forwarded values are uniform.
  llvm::DenseMap<Block *, llvm::SmallVector<Operation *, 2>>
      syncDependentBranches;
};

//===----------------------------------------------------------------------===//
// UniformityMemoryAnalysis — dense forward, N-level
//
// Propagates UniformityMemoryState through the program. For each write effect
// at an op, classifies the stored value's uniformity at *every* active parallel
// level (from ParallelContextState) and records the result in the per-level
// map of the successor state. Reads are classified by UniformityValueAnalysis
// via classifyReadEffect(op, addr, level), which queries the per-level map
// with the specific level being analysed.
//===----------------------------------------------------------------------===//

class UniformityMemoryAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<UniformityMemoryState> {
public:
  using Base = dataflow::DenseForwardDataFlowAnalysis<UniformityMemoryState>;

  UniformityMemoryAnalysis(DataFlowSolver &solver, DataFlowSolver &aliasSolver)
      : Base(solver), aliasSolver(aliasSolver) {}

  void setToEntryState(UniformityMemoryState *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               const UniformityMemoryState &before,
                               UniformityMemoryState *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const UniformityMemoryState &before,
                                    UniformityMemoryState *after) override;

  const enzyme::AliasClassLattice *getAliasState(Value v) const {
    return aliasSolver.lookupState<enzyme::AliasClassLattice>(v);
  }

  /// DataLayout covering `op`, built once and reused (see the value analysis).
  const DataLayout &getDataLayout(Operation *op);

private:
  std::unique_ptr<DataLayout> cachedDataLayout;

  /// Compute uniformity of the data being written by `op` at `level`.
  /// Joins over all non-address operands' uniformity at that parallel level.
  /// Conservative: over-approximates for ops with metadata operands.
  Uniformity classifyStoredValue(Operation *op, Value addr, Operation *level);

  DataFlowSolver &aliasSolver;
};

//===----------------------------------------------------------------------===//
// Public queries
//===----------------------------------------------------------------------===//

/// Returns the uniformity of `value` at a specific parallel level.
/// `parallelOp` is the spmd::ParallelOp whose executor set is of interest.
/// Consults the UniformityLattice from `solver` via a sparse lookup.
Uniformity getUniformityAtLevel(Value value, Operation *parallelOp,
                                DataFlowSolver &solver);

/// Returns true iff `op` is enclosed in a divergent (NonUniform) branch or
/// loop at the innermost active parallel level, checking both the scf
/// (RegionBranchOpInterface) path and the cf (CondBranch/Switch) path.
bool isDivergent(Operation *op, DataFlowSolver &solver);

} // namespace mlir::spmd

#endif // MLIR_DIALECT_SPMD_ANALYSIS_UNIFORMITYANALYSIS_H