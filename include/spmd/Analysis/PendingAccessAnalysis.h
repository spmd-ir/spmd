/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef SPMD_ANALYSIS_PENDINGACCESSANALYSIS_H
#define SPMD_ANALYSIS_PENDINGACCESSANALYSIS_H

#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/Analysis/HandleFacts.h"

#include "Enzyme/MLIR/Analysis/DataFlowAliasAnalysis.h"

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/CallInterfaces.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>
#include <unordered_set>

namespace mlir::spmd {

using AliasBucket = DistinctAttr;

//===----------------------------------------------------------------------===//
// Error reporting types
//===----------------------------------------------------------------------===//

struct ErrorInfo {
  Operation *prevOpr = nullptr;
  Operation *causingOpr = nullptr;
  Value aliasPrev;
  Value aliasCausing;
  ExecKind execKindPrev = ExecKind::All;
  ExecKind execKindCausing = ExecKind::All;
  llvm::SmallDenseSet<int32_t> overlapRanks;

  ErrorInfo() = default;
  ErrorInfo(Operation *prevOpr, Operation *causingOpr, Value aliasPrev,
            Value aliasCausing, ExecKind execKindPrev, ExecKind execKindCausing,
            llvm::SmallDenseSet<int32_t> overlapRanks);

  bool operator==(const ErrorInfo &other) const;
};

struct ErrorInfoHash {
  size_t operator()(const ErrorInfo &err) const;
};

using ErrorSet = std::unordered_set<ErrorInfo, ErrorInfoHash>;

//===----------------------------------------------------------------------===//
// Memory access tracking
//===----------------------------------------------------------------------===//

struct MemoryAccess {
  Operation *user = nullptr;
  Value alias;
  bool isWrite = false;
  ExecKind execKind = ExecKind::All;
  llvm::SmallDenseSet<int32_t> ranks;
  int32_t usedModel = -1;

  MemoryAccess() = default;
  MemoryAccess(Operation *user, Value alias, bool isWrite, ExecKind execKind,
               llvm::SmallDenseSet<int32_t> ranks, int32_t usedModel);

  bool operator==(const MemoryAccess &other) const;
};

struct MemoryAccessHash {
  size_t operator()(const MemoryAccess &memAccess) const;
};

//===----------------------------------------------------------------------===//
// Pending access records
//===----------------------------------------------------------------------===//

struct PendingAccessRecord {
  MemoryAccess access;
  HandleAtomSet oppositeAtoms;
  bool oppositeUnknown = false;

  bool operator==(const PendingAccessRecord &other) const {
    return access == other.access && oppositeAtoms == other.oppositeAtoms &&
           oppositeUnknown == other.oppositeUnknown;
  }
};

struct PendingAccessRecordHash {
  size_t operator()(const PendingAccessRecord &rec) const;
};

using PendingRecordSet =
    std::unordered_set<PendingAccessRecord, PendingAccessRecordHash>;

using AccessMapByAlias =
    llvm::SmallDenseMap<AliasBucket,
                        llvm::SmallDenseMap<HandleAtom, PendingRecordSet>>;

//===----------------------------------------------------------------------===//
// Lattice state
//===----------------------------------------------------------------------===//

struct PendingStateValue {
  llvm::SmallDenseMap<AliasBucket, HandleFactsValue> handleSlotContents;
  HandleFactsValue unknownHandleSlotContents;

  AccessMapByAlias accessMapByAlias;
  llvm::SmallDenseMap<HandleAtom, PendingRecordSet> unknownAccessesByHandle;
  PendingRecordSet unknownHandleAccesses;

  ChangeResult joinInPlace(const PendingStateValue &rhs) {
    bool changed = false;

    auto mergeRecordSet = [&](PendingRecordSet &dst,
                              const PendingRecordSet &src) {
      const size_t oldSize = dst.size();
      dst.insert(src.begin(), src.end());
      changed |= dst.size() != oldSize;
    };

    for (const auto &[bucket, rhsFacts] : rhs.handleSlotContents)
      changed |= handleSlotContents[bucket].joinInPlace(rhsFacts) ==
                 ChangeResult::Change;

    changed |= unknownHandleSlotContents.joinInPlace(
                   rhs.unknownHandleSlotContents) == ChangeResult::Change;

    for (const auto &[bucket, rhsByHandle] : rhs.accessMapByAlias) {
      auto &dstByHandle = accessMapByAlias[bucket];
      for (const auto &[handle, rhsRecords] : rhsByHandle)
        mergeRecordSet(dstByHandle[handle], rhsRecords);
    }

    for (const auto &[handle, rhsRecords] : rhs.unknownAccessesByHandle)
      mergeRecordSet(unknownAccessesByHandle[handle], rhsRecords);

    mergeRecordSet(unknownHandleAccesses, rhs.unknownHandleAccesses);
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  void print(llvm::raw_ostream &os) const {
    os << "<PendingStateValue"
       << " handleSlots=" << handleSlotContents.size()
       << " accessBuckets=" << accessMapByAlias.size()
       << " unknownAccessHandles=" << unknownAccessesByHandle.size()
       << " unknownHandleAccesses=" << unknownHandleAccesses.size() << ">";
  }
};

class PendingStateLattice : public dataflow::AbstractDenseLattice {
public:
  using AbstractDenseLattice::AbstractDenseLattice;

  PendingStateValue &getValue() { return value; }
  const PendingStateValue &getValue() const { return value; }

  ChangeResult join(const AbstractDenseLattice &rhs) override {
    return value.joinInPlace(
        static_cast<const PendingStateLattice &>(rhs).value);
  }

  void print(llvm::raw_ostream &os) const override { value.print(os); }

private:
  PendingStateValue value;
};

//===----------------------------------------------------------------------===//
// PendingAccessAnalysis
//===----------------------------------------------------------------------===//

/// Dense forward analysis that detects data races between non-blocking
/// communication operations and local memory accesses.
class PendingAccessAnalysis
    : public dataflow::DenseForwardDataFlowAnalysis<PendingStateLattice> {
public:
  using Base = dataflow::DenseForwardDataFlowAnalysis<PendingStateLattice>;

  PendingAccessAnalysis(DataFlowSolver &solver, DataFlowSolver &aliasSolver)
      : Base(solver), aliasSolver(aliasSolver) {}

  // explicit PendingAccessAnalysis(DataFlowSolver &solver);

  LogicalResult visitOperation(Operation *op, const PendingStateLattice &before,
                               PendingStateLattice *after) override;

  void visitCallControlFlowTransfer(CallOpInterface call,
                                    dataflow::CallControlFlowAction action,
                                    const PendingStateLattice &before,
                                    PendingStateLattice *after) override;

  const ErrorSet &getErrors() const { return errors; }

  struct AliasBucketSet {
    llvm::SmallDenseSet<AliasBucket> buckets;
    bool hasUnknown = false;
  };

protected:
  void setToEntryState(PendingStateLattice *lattice) override;

private:
  //-- Handle-facts resolution ---------------------------------------------
  LogicalResult resolveHandleFacts(Value value, const PendingStateValue &state,
                                   Operation *op, HandleFactsValue &out);
  LogicalResult recordHandleStore(Value storedValue, Value slotAddr,
                                  PendingStateValue &state, Operation *op);
  LogicalResult recordHandleStore(Operation *op, PendingStateValue &state);

  //-- Transfer function phases --------------------------------------------

  /// Top-level transfer: delegates to the three phases below.
  LogicalResult applyOperationTransfer(Operation *op, PendingStateValue &state,
                                       ErrorSet &errors);

  /// Phase 1: Record handle stores into slot-content tracking maps.
  LogicalResult transferHandleStores(Operation *op, PendingStateValue &state);

  /// Phase 2: Check memory accesses for conflicts with pending records and
  /// insert new pending records for SPMD communication operations.
  LogicalResult transferMemoryAccesses(Operation *op, PendingStateValue &state,
                                       ErrorSet &errors);

  /// Phase 3: Apply synchronization resets (RMA fences and non-blocking waits)
  /// that kill matching pending records.
  LogicalResult transferSyncResets(Operation *op, PendingStateValue &state);

  //-- Error checking ------------------------------------------------------
  void checkAccessPair(const MemoryAccess &prevAccess,
                       const MemoryAccess &currAccess, ErrorSet &errors);

  //-- Queries -------------------------------------------------------------
  FailureOr<llvm::SmallVector<MemoryAccess>> getMemoryAccesses(Operation *op);

  std::optional<AliasBucketSet> getAliasBucketsForValue(Operation *op,
                                                        Value value);
  HandleFactsValue lookupSparseHandleFactsForValue(Operation *op, Value value);

  // const enzyme::AliasClassLattice *getAliasStateForCurrentOp(Operation *op,
  //                                                            Value value) {
  //   return
  //   getOrCreateFor<enzyme::AliasClassLattice>(getProgramPointAfter(op),
  //                                                    value);
  // }

  const enzyme::AliasClassLattice *getAliasState(Operation *op, Value val) {
    return aliasSolver.lookupState<enzyme::AliasClassLattice>(val);
  }

  const HandleFactsLattice *getHandleFactsStateForCurrentOp(Operation *op,
                                                            Value value) {
    return getOrCreateFor<HandleFactsLattice>(getProgramPointAfter(op), value);
  }

  FailureOr<LevelInfo> getExecutionInfoForCurrentOp(Operation *op) {
    auto *lattice = getOrCreateFor<ExecutionContextLattice>(
        getProgramPointAfter(op), getProgramPointBefore(op));
    if (!lattice || !lattice->isInitialized())
      return failure();
    const ExecutionContext &ctx = lattice->getValue();
    // Data race analysis is process-level: use the innermost process-level
    // entry.
    const LevelInfo *pl = ctx.innermostProcessLevel();
    if (!pl)
      return failure();
    return *pl;
  }

  DataFlowSolver &aliasSolver;
  ErrorSet errors;
};

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_PENDINGACCESSANALYSIS_H