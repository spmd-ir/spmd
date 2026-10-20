/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef SPMD_ANALYSIS_EXECUTIONCONTEXTANALYSIS_H
#define SPMD_ANALYSIS_EXECUTIONCONTEXTANALYSIS_H

#include "spmd/Analysis/UniformityAnalysis.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <optional>

namespace mlir::spmd {

/// Describes which ranks execute a given operation.
enum class ExecKind {
  All,       ///< All ranks.
  Static,    ///< A statically known subset of ranks.
  Dynamic,   ///< An unknown dynamic subset.
  AllBut,    ///< All ranks except a statically known set.
  One,       ///< Exactly one (uniform) rank.
  AllButOne, ///< All ranks except one (uniform) rank.
};

inline raw_ostream &operator<<(raw_ostream &os, ExecKind kind) {
  switch (kind) {
  case ExecKind::All:
    return os << "All";
  case ExecKind::Static:
    return os << "Static";
  case ExecKind::Dynamic:
    return os << "Dynamic";
  case ExecKind::AllBut:
    return os << "AllBut";
  case ExecKind::One:
    return os << "One";
  case ExecKind::AllButOne:
    return os << "AllButOne";
  }
  llvm_unreachable("unhandled ExecKind");
}

//===----------------------------------------------------------------------===//
// LevelInfo — execution state for one parallelism level.
//===----------------------------------------------------------------------===//
struct LevelInfo {
  int modelGroup = -1; // -1 = implicit entry level
  bool isProcessLevel = true;
  ExecKind execKind = ExecKind::All;
  SmallVector<int32_t> executors;
  /// The spmd::ParallelOp this level was pushed for.
  /// nullptr only for the implicit entry level (modelGroup == -1).
  /// Stored here so callers can form the UniformityLattice key without
  /// a structural IR walk (which breaks interprocedurally).
  Operation *parallelOp = nullptr;

  bool operator==(const LevelInfo &rhs) const {
    return modelGroup == rhs.modelGroup &&
           isProcessLevel == rhs.isProcessLevel && execKind == rhs.execKind &&
           executors == rhs.executors && parallelOp == rhs.parallelOp;
  }
  bool operator!=(const LevelInfo &rhs) const { return !(*this == rhs); }

  bool hasKnownExecutors() const {
    return execKind == ExecKind::Static || execKind == ExecKind::AllBut;
  }
  bool isStaticExecution() const { return execKind == ExecKind::Static; }

  void print(raw_ostream &os) const {
    os << "{mg=" << modelGroup << " proc=" << isProcessLevel
       << " kind=" << execKind;
    if (!executors.empty()) {
      os << " execs=[";
      llvm::interleaveComma(executors, os);
      os << "]";
    }
    os << "}";
  }
};

inline raw_ostream &operator<<(raw_ostream &os, const LevelInfo &l) {
  l.print(os);
  return os;
}

//===----------------------------------------------------------------------===//
// ExecutionContext — stack of LevelInfo, outermost first.
//===----------------------------------------------------------------------===//
struct ExecutionContext {
  SmallVector<LevelInfo> levels;
  bool initialized = false;

  /// Every rank that executes this point by `levels` reaches it the same
  /// number of times: no loop bound, branch condition or call site that is not
  /// uniform lies between the entry of the program and here. A rank-dependent
  /// trip count leaves the set of executing ranks as it is, which is why
  /// ExecKind cannot express this. Scoped: leaving a loop, a conditional or a
  /// callee restores the value from before it. Joins by logical and.
  bool uniformCount = true;
  /// No rank may have stopped before reaching this point. Cleared after a call
  /// to a noreturn function, or a return, that not every rank reaches, and
  /// never restored: the ranks that left do not come back. Joins by logical
  /// and.
  bool noEarlyExit = true;

  bool operator==(const ExecutionContext &rhs) const {
    return initialized == rhs.initialized && levels == rhs.levels &&
           uniformCount == rhs.uniformCount && noEarlyExit == rhs.noEarlyExit;
  }
  bool operator!=(const ExecutionContext &rhs) const { return !(*this == rhs); }

  bool isInitialized() const { return initialized; }
  bool empty() const { return levels.empty(); }

  LevelInfo &top() {
    assert(!empty());
    return levels.back();
  }
  const LevelInfo &top() const {
    assert(!empty());
    return levels.back();
  }

  /// Entry state: one implicit process-level top entry.
  static ExecutionContext getEntry() {
    ExecutionContext ctx;
    ctx.initialized = true;
    LevelInfo top;
    top.modelGroup = -1;
    top.isProcessLevel = true;
    top.execKind = ExecKind::All;
    ctx.levels.push_back(top);
    return ctx;
  }

  //===--- Queries used by CheckThreadSafety ---===//

  /// Innermost thread-level (isProcessLevel=false) entry, or nullptr.
  const LevelInfo *innermostThreadLevel() const {
    for (const LevelInfo &l : llvm::reverse(levels))
      if (!l.isProcessLevel)
        return &l;
    return nullptr;
  }

  /// Innermost process-level entry, or nullptr.
  const LevelInfo *innermostProcessLevel() const {
    for (const LevelInfo &l : llvm::reverse(levels))
      if (l.isProcessLevel)
        return &l;
    return nullptr;
  }

  /// True if the innermost thread-level has executors == {0} (OMP master only).
  bool isMasterOnly() const {
    const LevelInfo *tl = innermostThreadLevel();
    if (!tl)
      return true;
    return tl->execKind == ExecKind::Static && tl->executors.size() == 1 &&
           tl->executors[0] == 0;
  }

  /// True if the innermost thread-level indicates concurrent execution —
  /// i.e. multiple threads may reach this point simultaneously.
  ///
  /// Safe (returns false) only when execKind implies a single executor:
  ///   ExecKind::One   — spmd.single / spmd.critical body
  ///   ExecKind::Static with exactly one known executor — scf.if rank==X guard
  bool isThreadLevelConcurrent() const {
    const LevelInfo *tl = innermostThreadLevel();
    if (!tl)
      return false;
    switch (tl->execKind) {
    case ExecKind::One:
      return false;
    case ExecKind::Static:
      return tl->executors.size() != 1;
    default:
      return true; // All / AllBut / AllButOne / Dynamic → concurrent
    }
    llvm_unreachable("unhandled ExecKind");
  }

  void print(raw_ostream &os) const {
    os << "ExecutionContext[";
    for (const LevelInfo &l : levels)
      os << l << " ";
    if (!uniformCount)
      os << "nonuniform-count ";
    if (!noEarlyExit)
      os << "early-exit ";
    os << "]";
  }
};

//===----------------------------------------------------------------------===//
// ExecutionContextLattice — holds an ExecutionContext.
//===----------------------------------------------------------------------===//
class ExecutionContextLattice : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  bool isInitialized() const { return value.isInitialized(); }

  const ExecutionContext &getValue() const {
    assert(value.initialized &&
           "reading uninitialized ExecutionContextLattice");
    return value;
  }

  ChangeResult setValue(const ExecutionContext &newValue) {
    if (value.initialized && value == newValue)
      return ChangeResult::NoChange;
    value = newValue;
    return ChangeResult::Change;
  }

  ChangeResult join(const AbstractDenseLattice &other) override;

  void print(raw_ostream &os) const override {
    if (!value.initialized)
      os << "<uninitialized>";
    else
      value.print(os);
  }

private:
  ExecutionContext value;
};

// Add near the top of the class / before it — documents the three-way result
// failure()        — uniformity lattice not yet populated; retry later
// success(nullopt) — value is provably non-uniform; no single executor
// success(v)       — single executor: concrete rank or kUniformExecutorSentinel
using ExecutorResult = FailureOr<std::optional<int32_t>>;

//===----------------------------------------------------------------------===//
// ExecutionContextAnalysis
//===----------------------------------------------------------------------===//
class ExecutionContextAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<ExecutionContextLattice> {
public:
  using Base = dataflow::DenseForwardDataFlowAnalysis<ExecutionContextLattice>;
  explicit ExecutionContextAnalysis(DataFlowSolver &solver) : Base(solver) {}

  void setToEntryState(ExecutionContextLattice *lattice) override;

  LogicalResult visitOperation(Operation *op,
                               const ExecutionContextLattice &before,
                               ExecutionContextLattice *after) override;

  void visitRegionBranchControlFlowTransfer(
      RegionBranchOpInterface branch, std::optional<unsigned> regionFrom,
      std::optional<unsigned> regionTo, const ExecutionContextLattice &before,
      ExecutionContextLattice *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const ExecutionContextLattice &before,
                                    ExecutionContextLattice *after) override;

  static ExecutionContext joinContext(const ExecutionContext &lhs,
                                      const ExecutionContext &rhs);
  static LevelInfo joinLevelInfo(const LevelInfo &lhs, const LevelInfo &rhs);
  static LevelInfo refineLevelInfo(const LevelInfo &base,
                                   const LevelInfo &constraint);

  /// Query uniformity of `value` at explicit parallel `level` (a ParallelOp*,
  /// nullptr = outside all parallels → Uniform by missing-entry convention).
  LogicalResult getUniformity(Operation *dependentOp, Value value,
                              Operation *level, Uniformity &out);

  /// Determine if `value` is a known single executor at `level`.
  ExecutorResult getExecutor(Value value, Operation *level);

  void getCmpOpsNonReversed(SmallVectorImpl<Operation *> &cmpOps,
                            Operation *condOp) const;
  void getCmpOps(SmallVectorImpl<Operation *> &cmpOps, Operation *condOp,
                 bool &reversedCase) const;

  FailureOr<SmallVector<int32_t>> getExecutorsForIf(scf::IfOp ifOp,
                                                    bool &reversedCase,
                                                    const ExecutionContext &ctx,
                                                    Operation *&targetLevel);

private:
  struct IfBranchInfo {
    bool valid = false;
    LevelInfo thenRefinement;
    LevelInfo elseRefinement;
    /// The level whose executors this refinement narrows (the level matching
    /// the guard's rank op, or the innermost non-uniform level for non-rank
    /// conditions). nullptr ⟹ fall back to top().
    Operation *targetLevel = nullptr;
  };

  /// Returns true iff the branch condition is non-uniform at any level in
  /// `ctx`.
  bool isConditionNonUniform(RegionBranchOpInterface branchOp,
                             const ExecutionContext &ctx);

  /// Innermost level in `ctx` at which `ifOp`'s condition is non-uniform.
  Operation *innermostNonUniformLevel(scf::IfOp ifOp,
                                      const ExecutionContext &ctx);

  IfBranchInfo getIfBranchInfo(scf::IfOp ifOp,
                               const ExecutionContextLattice &before);

  /// Match the rank op's model group against `ctx.levels` to find its
  /// ParallelOp*.  Asserts on mismatch (malformed IR).
  Operation *getRankLevel(Value rankVal, const ExecutionContext &ctx);

  void propagateContext(const ExecutionContext &ctx,
                        ExecutionContextLattice *after);

  /// propagateContext for the point after `op`, keeping an early exit taken
  /// inside `op`'s regions when the framework does not flow them there.
  void propagateAfter(Operation *op, ExecutionContext ctx,
                      ExecutionContextLattice *after);

  /// `value` is Uniform at every level of `ctx`. A lattice not computed yet
  /// counts as uniform for now; `dependent` is revisited once it is.
  bool isUniformAtEveryLevel(Value value, const ExecutionContext &ctx,
                             ProgramPoint *dependent);

  /// Control reaches the ops of `block` on every rank that reaches the entry
  /// of its region: no cf branch whose condition is not uniform decides
  /// whether `block` runs.
  bool isReachedUniformly(Block *block, const ExecutionContext &ctx,
                          ProgramPoint *dependent);

  /// What decides how often `branch` runs its regions - a loop's bounds, a
  /// conditional's condition - is uniform. False for a construct not known
  /// here.
  bool entersUniformly(RegionBranchOpInterface branch,
                       const ExecutionContext &ctx, ProgramPoint *dependent);

  /// CFGControlInfo per region, computed on first use.
  llvm::DenseMap<Region *,
                 std::unique_ptr<UniformityValueAnalysis::CFGControlInfo>>
      cfgControlInfo;
};

//===----------------------------------------------------------------------===//
// Queries on a solved ExecutionContextAnalysis
//===----------------------------------------------------------------------===//

/// True when every rank executes `op`, each the same number of times, as far
/// as the ExecutionContextAnalysis and UniformityValueAnalysis solved in
/// `solver` can show: every parallel level active at `op` is ExecKind::All, no
/// loop bound, branch condition or call site on the way there is non-uniform
/// or unknown, and no rank may have stopped earlier. An optimization needs this
/// before it lets an op establish a fact that a rewrite on all ranks relies on.
/// Errs towards false, including for an op the analysis never reached.
bool isUniformlyExecuted(Operation *op, DataFlowSolver &solver);

/// True when `value` is Uniform at every parallel level active at `op`: every
/// rank holds the same value there. Unknown and not computed are not uniform.
bool isUniformValue(Value value, Operation *op, DataFlowSolver &solver);

} // namespace mlir::spmd
#endif