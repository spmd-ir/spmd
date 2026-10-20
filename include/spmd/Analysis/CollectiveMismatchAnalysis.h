/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef SPMD_ANALYSIS_COLLECTIVEMISMATCHANALYSIS_H
#define SPMD_ANALYSIS_COLLECTIVEMISMATCHANALYSIS_H

#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/Analysis/HandleFacts.h"
#include "spmd/Analysis/HandleSlotAnalysis.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// CollectiveMismatchAnalysis — support types
//===----------------------------------------------------------------------===//

struct CollectiveMismatchErrorInfo {
  SmallVector<Operation *> conflictOprs;
  SmallVector<Operation *> causingOprs;

  CollectiveMismatchErrorInfo() = default;
  CollectiveMismatchErrorInfo(SmallVector<Operation *> conflicts,
                              SmallVector<Operation *> causes)
      : conflictOprs(std::move(conflicts)), causingOprs(std::move(causes)) {}
};

/// Key identifying a collective operation type for ordering checks.
struct CollectiveOpKey {
  OperationName name;
  bool isBlocking;

  /// Model group of the op's `usedModel` attribute. Part of the key because
  /// collectives of different programming models never match each other: an
  /// MPI barrier does not synchronise with a SHMEM one even though both lower
  /// to `spmd.barrier`. Within a per-model partition this is constant and the
  /// comparison is a no-op; it earns its keep in the cross-model ordering
  /// check, where sequences from several models are compared side by side and
  /// two same-named collectives would otherwise be treated as partners:
  ///     if (rank % 2) { mpi barrier; shmem barrier }
  ///     else          { shmem barrier; mpi barrier }
  /// is a deadlock that positional comparison misses entirely without this.
  int modelGroup;

  // When true, two keys are only equal if they wrap the same Operation*
  // (pointer identity). This models OMP barrier semantics: every thread
  // must reach this exact instruction, not just any barrier at this position.
  bool isUniqueMatching;
  Operation *uniqueIdentity; // non-null iff isUniqueMatching == true

  bool operator==(const CollectiveOpKey &other) const {
    if (name != other.name || isBlocking != other.isBlocking)
      return false;
    if (modelGroup != other.modelGroup)
      return false;
    if (isUniqueMatching || other.isUniqueMatching)
      return uniqueIdentity == other.uniqueIdentity;
    return true;
  }

  // Constructor for normal (MPI-style) collectives.
  CollectiveOpKey(OperationName name, bool isBlocking, int modelGroup = -1)
      : name(name), isBlocking(isBlocking), modelGroup(modelGroup),
        isUniqueMatching(false), uniqueIdentity(nullptr) {}

  // Constructor for unique-matching (OMP-style) collectives.
  CollectiveOpKey(OperationName name, bool isBlocking, bool uniqueMatching,
                  Operation *op, int modelGroup = -1)
      : name(name), isBlocking(isBlocking), modelGroup(modelGroup),
        isUniqueMatching(uniqueMatching),
        uniqueIdentity(uniqueMatching ? op : nullptr) {}
};

struct CollectiveOpKeyHash {
  size_t operator()(const CollectiveOpKey &key) const {
    size_t h = llvm::hash_combine(hash_value(key.name),
                                  std::hash<bool>()(key.isBlocking),
                                  std::hash<int>()(key.modelGroup));
    if (key.isUniqueMatching)
      h = llvm::hash_combine(h,
                             reinterpret_cast<uintptr_t>(key.uniqueIdentity));
    return h;
  }
};

/// A single collective operation discovered during traversal.
struct CollectiveRecord {
  Operation *op;
  CollectiveOpKey key;

  size_t executionSlot = 0;
  uint64_t cumulativeIterations = 1;

  /// Model group of the op's `usedModel` attribute.
  int modelGroup = -1;
  /// Process- vs thread-level of the binding parallel (innermost active level
  /// with matching modelGroup). nullopt = kind not resolvable (conservative).
  std::optional<bool> isProcessLevel;

  CollectiveRecord(Operation *op, CollectiveOpKey key, size_t slot,
                   uint64_t iters, int modelGroup,
                   std::optional<bool> isProcessLevel)
      : op(op), key(key), executionSlot(slot), cumulativeIterations(iters),
        modelGroup(modelGroup), isProcessLevel(isProcessLevel) {}
};

/// Ordered sequence of collective records in traversal order.
/// At a fork point, one FlatSequence per branch is produced and compared.
using FlatSequence = SmallVector<CollectiveRecord>;

/// State threaded through the recursive traversal.
struct TraversalContext {
  /// Ever-increasing slot counter — gives each discovered collective
  /// a unique position in the execution order.
  size_t slotCounter = 0;
  /// Accumulated product of constant loop trip counts on the path
  /// from the traversal root to the current op.
  uint64_t cumulativeIterations = 1;
  /// Call ops on the current path from main to the current function.
  /// Used as a recursion guard (no re-entering the same function).
  SmallVector<Operation *> callStack;

  /// We are inside a branch that splits the physical ranks (any process-level
  /// model). Persists across process-level parallels.
  bool inProcessDivergent = false;
  /// We are inside a branch that splits the current thread team (any thread-
  /// level model). Reset on entry to a new thread-level parallel.
  bool inThreadDivergent = false;
};

//===----------------------------------------------------------------------===//
// CollectiveMismatchAnalysis
//===----------------------------------------------------------------------===//

class CollectiveMismatchAnalysis {
public:
  explicit CollectiveMismatchAnalysis(DataFlowSolver &solver, uint64_t numRanks)
      : solver(solver), numRanks(numRanks) {}

  LogicalResult run(Operation *top);

  const SmallVector<CollectiveMismatchErrorInfo> &getErrors() const {
    return errorInfos;
  }

private:
  // ── Communicator resolution ────────────────────────────────────────────
  HandleFactsValue resolveCommFacts(Value value, Operation *contextOp) const;
  /// True if `commVal` may name a communicator in `commClass`. An unresolvable
  /// value counts as a member (conservative).
  bool commValueInClass(Value commVal, Operation *contextOp,
                        const llvm::DenseSet<HandleAtom> &commClass) const;
  bool isSearchedComm(Operation *opr,
                      const llvm::DenseSet<HandleAtom> &commClass) const;

  // ── Traversal ─────────────────────────────────────────────────────────

  /// Dispatch to single-block or multi-block traversal.
  FlatSequence
  traverseRegion(Region &region, TraversalContext &ctx,
                 const llvm::DenseSet<HandleAtom> &commClass) const;

  /// Process a single-block (SCF-style) region op by op.
  FlatSequence
  traverseSingleBlock(Block &block, TraversalContext &ctx,
                      const llvm::DenseSet<HandleAtom> &commClass) const;

  /// Process a multi-block (cf-dialect) region in RPO order.
  FlatSequence
  traverseCFRegion(Region &region, TraversalContext &ctx,
                   const llvm::DenseSet<HandleAtom> &commClass) const;

  /// Handle a non-loop RegionBranchOpInterface fork (scf.if, scf.switch…):
  /// traverse each branch, compare sequences, report mismatches.
  FlatSequence handleFork(RegionBranchOpInterface forkOp, TraversalContext &ctx,
                          const llvm::DenseSet<HandleAtom> &commClass) const;

  /// Handle a fork the collective does not sit inside.
  ///
  /// When two collective calls differ only in the communicator, the branch
  /// between them does not survive to the checker: clang if-converts them into
  /// one call, and the pick becomes a value — either an `arith.select`, or an
  /// `scf.if` that yields the communicator when an arm is not speculatable.
  ///     if (rank % 2) ncclAllGather(..., commWorld, ...);
  ///     else          ncclAllGather(..., newComm,   ...);
  /// becomes `spmd.allgather(arith.select %c, %commWorld, %newComm, ...)`;
  ///     MPI_Barrier(rank % 2 ? MPI_COMM_WORLD : splitComm);
  /// becomes a barrier on the result of an `scf.if` whose regions hold no
  /// collective at all. Either way `handleFork` finds nothing to compare —
  /// there is no branch around the collective, only around its argument.
  ///
  /// Both arms being the same instruction, the synthesized branches are one
  /// record long each and the only thing separating them is whether that arm's
  /// communicator belongs to the class under test. For the examples above and
  /// the class of `MPI_COMM_WORLD` that is `[collective]` against `[]` — the
  /// members that took the other arm never call it — which is the verdict the
  /// equivalent `scf.if`-around-the-call shape produces.
  void checkCommValueFork(Operation *opr, const TraversalContext &ctx,
                          const llvm::DenseSet<HandleAtom> &commClass) const;

  FlatSequence
  handleSections(Operation *sectionsOp, TraversalContext &ctx,
                 const llvm::DenseSet<HandleAtom> &commClass) const;

  /// Handle a loop-like RegionBranchOpInterface (scf.for, scf.while…):
  /// traverse the body once, fold constant trip counts.
  FlatSequence handleLoop(RegionBranchOpInterface loopOp, TraversalContext &ctx,
                          const llvm::DenseSet<HandleAtom> &commClass) const;

  /// Handle a call op: recurse into the callee if not already on the stack.
  FlatSequence handleCall(CallOpInterface call, TraversalContext &ctx,
                          const llvm::DenseSet<HandleAtom> &commClass) const;

  FlatSequence handleSingle(Operation *singleOp, TraversalContext &ctx,
                            const llvm::DenseSet<HandleAtom> &commClass) const;

  FlatSequence
  handleCritical(Operation *critOp, TraversalContext &ctx,
                 const llvm::DenseSet<HandleAtom> &commClass) const;

  FlatSequence
  handleParallel(Operation *parallelOp, TraversalContext &ctx,
                 const llvm::DenseSet<HandleAtom> &commClass) const;

  /// Worksharing loop (spmd.wsloop / spmd.taskloop): iteration-space thread
  /// partition. Modeled as a structural thread fork [body], [empty].
  FlatSequence
  handleWorksharingLoop(Operation *wsOp, TraversalContext &ctx,
                        const llvm::DenseSet<HandleAtom> &commClass) const;

  // ── Mismatch detection (add to the private section) ─────────────────────
  /// Partition branch sequences by modelGroup and run checkAndReportMismatch
  /// on each partition only if the fork splits that collective's kind.
  /// `structuralThreadFork` is true for single/critical/sections.
  void checkBranchesByKind(ArrayRef<FlatSequence> branchSeqs, Operation *forkOp,
                           const TraversalContext &ctx,
                           bool structuralThreadFork) const;

  // ── Mismatch detection ────────────────────────────────────────────────

  /// Compare branch sequences position-by-position.
  /// Emit one CollectiveMismatchErrorInfo per mismatching position,
  /// listing all conflicting ops at that position and forkOp as the cause.
  void checkAndReportMismatch(ArrayRef<FlatSequence> branchSeqs,
                              Operation *forkOp) const;

  /// Return the union of all branch sequences (used to give the parent
  /// a conservative view of what collectives may have executed).
  static FlatSequence mergeFlatSequences(ArrayRef<FlatSequence> seqs);

  // ── State ─────────────────────────────────────────────────────────────

  DataFlowSolver &solver;
  uint64_t numRanks = 0;
  mutable SmallVector<CollectiveMismatchErrorInfo> errorInfos;

  /// Set for the extra whole-module traversal that records every collective
  /// regardless of communicator class, so that ordering can be compared across
  /// programming models. While set, checkBranchesByKind runs ONLY its
  /// cross-model comparison — the per-model ones already ran, per class, in the
  /// main traversals.
  mutable bool crossModelOnly = false;
};

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_COLLECTIVEMISMATCHANALYSIS_H