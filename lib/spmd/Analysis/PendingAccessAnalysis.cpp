// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/PendingAccessAnalysis.h"

#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

#include <sstream>

#define DEBUG_TYPE "check-dataRace"
#define PATTERN "main"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// Value classification utilities
//===----------------------------------------------------------------------===//

template <typename Range, typename Predicate>
static Value findFirstValue(Range values, Predicate predicate) {
  for (Value value : values)
    if (predicate(value))
      return value;
  return {};
}

static bool isPointerLike(Type type) {
  return isa<LLVM::LLVMPointerType, MemRefType>(type);
}

static bool isReqCarrierType(Type type) {
  if (isa<spmd::ReqType>(type))
    return true;
  if (auto memrefType = dyn_cast<MemRefType>(type))
    return isa<spmd::ReqType>(memrefType.getElementType());
  return false;
}

static bool isNonBlocking(Operation *op) {
  if (auto attr = op->getAttrOfType<BoolAttr>("isBlocking"))
    return !attr.getValue();
  return false;
}

static bool isRMASyncOp(Operation *op) {
  return op->hasTrait<spmd::RMA>() && op->hasTrait<spmd::Sync>();
}

static bool isNonBlockingSyncOp(Operation *op) {
  return op->hasTrait<spmd::NonBlockingSync>();
}

//===----------------------------------------------------------------------===//
// Operand/result accessors for handle types
//===----------------------------------------------------------------------===//

static bool hasWindowOperand(Operation *op) {
  return static_cast<bool>(findFirstValue(op->getOperands(), [](Value v) {
    return isa<spmd::WinType>(v.getType());
  }));
}

static Value getWindowOperand(Operation *op) {
  Value w = findFirstValue(op->getOperands(), [](Value v) {
    return isa<spmd::WinType>(v.getType());
  });
  assert(w && "expected window operand");
  return w;
}

static bool hasReqHandleResult(Operation *op) {
  return static_cast<bool>(findFirstValue(op->getResults(), [](Value v) {
    return isa<spmd::ReqType>(v.getType());
  }));
}

static Value getReqHandleResult(Operation *op) {
  Value r = findFirstValue(op->getResults(), [](Value v) {
    return isa<spmd::ReqType>(v.getType());
  });
  assert(r && "expected req result");
  return r;
}

static bool hasReqHandleOperand(Operation *op) {
  return static_cast<bool>(findFirstValue(op->getOperands(), [](Value v) {
    return isReqCarrierType(v.getType());
  }));
}

static Value getReqHandleOperand(Operation *op) {
  Value r = findFirstValue(
      op->getOperands(), [](Value v) { return isReqCarrierType(v.getType()); });
  assert(r && "expected req operand");
  return r;
}

static bool isMemoryInstruction(Operation *op) {
  return llvm::any_of(op->getOperands(),
                      [](Value v) { return isPointerLike(v.getType()); });
}

//===----------------------------------------------------------------------===//
// Rank-set helpers
//===----------------------------------------------------------------------===//

static llvm::SmallDenseSet<int32_t> toRankSet(ArrayRef<int32_t> ranks) {
  return {ranks.begin(), ranks.end()};
}

static llvm::SmallDenseSet<int32_t>
calculateOverlapRanks(const llvm::SmallDenseSet<int32_t> &lhs,
                      const llvm::SmallDenseSet<int32_t> &rhs) {
  llvm::SmallDenseSet<int32_t> overlap;
  const auto &smaller = lhs.size() < rhs.size() ? lhs : rhs;
  const auto &larger = lhs.size() < rhs.size() ? rhs : lhs;
  for (int32_t rank : smaller)
    if (larger.count(rank))
      overlap.insert(rank);
  return overlap;
}

static bool bothStatic(ExecKind lhs, ExecKind rhs) {
  return lhs == ExecKind::Static && rhs == ExecKind::Static;
}

static bool oneStaticOneAllBut(ExecKind lhs, ExecKind rhs) {
  return (lhs == ExecKind::Static && rhs == ExecKind::AllBut) ||
         (lhs == ExecKind::AllBut && rhs == ExecKind::Static);
}

//===----------------------------------------------------------------------===//
// Record set helpers
//===----------------------------------------------------------------------===//

static bool overlaps(const HandleAtomSet &lhs, const HandleAtomSet &rhs) {
  const auto &smaller = lhs.size() < rhs.size() ? lhs : rhs;
  const auto &larger = lhs.size() < rhs.size() ? rhs : lhs;
  for (HandleAtom atom : smaller)
    if (larger.count(atom))
      return true;
  return false;
}

static bool shouldKillRecord(HandleAtom keyHandle,
                             const PendingAccessRecord &record,
                             const HandleFactsValue &syncFacts) {
  if (syncFacts.unknown)
    return false;
  if (syncFacts.atoms.count(keyHandle))
    return true;
  return overlaps(record.oppositeAtoms, syncFacts.atoms);
}

template <typename Pred>
static void eraseMatching(PendingRecordSet &records, Pred pred) {
  for (auto it = records.begin(); it != records.end();) {
    if (pred(*it))
      it = records.erase(it);
    else
      ++it;
  }
}

template <typename Pred>
static void eraseMatchingByHandle(
    llvm::SmallDenseMap<HandleAtom, PendingRecordSet> &byHandle, Pred pred) {
  for (auto it = byHandle.begin(); it != byHandle.end();) {
    HandleAtom key = it->first;
    eraseMatching(it->second, [&](const PendingAccessRecord &rec) {
      return pred(key, rec);
    });
    if (it->second.empty()) {
      auto toErase = it++;
      byHandle.erase(toErase);
    } else {
      ++it;
    }
  }
}

template <typename Pred>
static void eraseMatchingByAlias(AccessMapByAlias &accessMap, Pred pred) {
  for (auto it = accessMap.begin(); it != accessMap.end();) {
    eraseMatchingByHandle(it->second, pred);
    if (it->second.empty()) {
      auto toErase = it++;
      accessMap.erase(toErase);
    } else {
      ++it;
    }
  }
}

static void resetBySyncHandleFacts(PendingStateValue &state,
                                   const HandleFactsValue &syncFacts) {
  if (syncFacts.unknown)
    return;

  auto pred = [&](HandleAtom key, const PendingAccessRecord &rec) {
    return shouldKillRecord(key, rec, syncFacts);
  };
  eraseMatchingByAlias(state.accessMapByAlias, pred);
  eraseMatchingByHandle(state.unknownAccessesByHandle, pred);
}

static void resetByUsedModel(PendingStateValue &state, int32_t usedModel) {
  auto sameModel = [&](const PendingAccessRecord &rec) {
    return rec.access.usedModel == usedModel;
  };
  auto sameModelKeyed = [&](HandleAtom, const PendingAccessRecord &rec) {
    return sameModel(rec);
  };

  eraseMatchingByAlias(state.accessMapByAlias, sameModelKeyed);
  eraseMatchingByHandle(state.unknownAccessesByHandle, sameModelKeyed);
  eraseMatching(state.unknownHandleAccesses, sameModel);
}

//===----------------------------------------------------------------------===//
// Candidate iteration
//===----------------------------------------------------------------------===//

template <typename Fn>
static void forEachCandidateRecordForAlias(
    const PendingStateValue &state,
    const std::optional<PendingAccessAnalysis::AliasBucketSet> &buckets,
    Fn &&fn) {
  std::unordered_set<PendingAccessRecord, PendingAccessRecordHash> seen;

  auto tryEmit = [&](const PendingAccessRecord &rec) {
    if (seen.insert(rec).second)
      fn(rec);
  };

  if (!buckets || buckets->hasUnknown) {
    for (const auto &[bucket, byHandle] : state.accessMapByAlias)
      for (const auto &[handle, records] : byHandle)
        for (const auto &rec : records)
          tryEmit(rec);
  } else {
    for (AliasBucket bucket : buckets->buckets) {
      auto it = state.accessMapByAlias.find(bucket);
      if (it == state.accessMapByAlias.end())
        continue;
      for (const auto &[handle, records] : it->second)
        for (const auto &rec : records)
          tryEmit(rec);
    }
  }

  for (const auto &[handle, records] : state.unknownAccessesByHandle)
    for (const auto &rec : records)
      tryEmit(rec);

  for (const auto &rec : state.unknownHandleAccesses)
    tryEmit(rec);
}

//===----------------------------------------------------------------------===//
// Struct implementations
//===----------------------------------------------------------------------===//

static llvm::hash_code hashRankSet(const llvm::SmallDenseSet<int32_t> &set) {
  llvm::hash_code result(0);
  for (int32_t v : set)
    result = llvm::hash_combine(result, v);
  return result;
}

MemoryAccess::MemoryAccess(Operation *user, Value alias, bool isWrite,
                           ExecKind execKind,
                           llvm::SmallDenseSet<int32_t> ranks,
                           int32_t usedModel)
    : user(user), alias(alias), isWrite(isWrite), execKind(execKind),
      ranks(std::move(ranks)), usedModel(usedModel) {}

bool MemoryAccess::operator==(const MemoryAccess &other) const {
  return user == other.user && alias == other.alias &&
         isWrite == other.isWrite && execKind == other.execKind &&
         ranks == other.ranks && usedModel == other.usedModel;
}

size_t MemoryAccessHash::operator()(const MemoryAccess &m) const {
  return llvm::hash_combine(std::hash<Operation *>()(m.user),
                            hash_value(m.alias), std::hash<bool>()(m.isWrite),
                            std::hash<int>()(static_cast<int>(m.execKind)),
                            hashRankSet(m.ranks),
                            std::hash<int32_t>()(m.usedModel));
}

ErrorInfo::ErrorInfo(Operation *prevOpr, Operation *causingOpr, Value aliasPrev,
                     Value aliasCausing, ExecKind execKindPrev,
                     ExecKind execKindCausing,
                     llvm::SmallDenseSet<int32_t> overlapRanks)
    : prevOpr(prevOpr), causingOpr(causingOpr), aliasPrev(aliasPrev),
      aliasCausing(aliasCausing), execKindPrev(execKindPrev),
      execKindCausing(execKindCausing), overlapRanks(std::move(overlapRanks)) {}

bool ErrorInfo::operator==(const ErrorInfo &other) const {
  auto locA1 = findLocToShowWrapper(prevOpr->getLoc());
  auto locA2 = findLocToShowWrapper(causingOpr->getLoc());
  auto locB1 = findLocToShowWrapper(other.prevOpr->getLoc());
  auto locB2 = findLocToShowWrapper(other.causingOpr->getLoc());
  return (locA1 == locB1 && locA2 == locB2) ||
         (locA1 == locB2 && locA2 == locB1);
}

size_t ErrorInfoHash::operator()(const ErrorInfo &err) const {
  auto h1 = hash_value(findLocToShowWrapper(err.prevOpr->getLoc()));
  auto h2 = hash_value(findLocToShowWrapper(err.causingOpr->getLoc()));
  return llvm::hash_combine(std::min(h1, h2), std::max(h1, h2));
}

size_t
PendingAccessRecordHash::operator()(const PendingAccessRecord &rec) const {
  return llvm::hash_combine(MemoryAccessHash{}(rec.access),
                            llvm::hash_combine_range(rec.oppositeAtoms.begin(),
                                                     rec.oppositeAtoms.end()),
                            std::hash<bool>()(rec.oppositeUnknown));
}

//===----------------------------------------------------------------------===//
// PendingAccessAnalysis — infrastructure
//===----------------------------------------------------------------------===//

// PendingAccessAnalysis::PendingAccessAnalysis(DataFlowSolver &solver)
//     : Base(solver) {}

void PendingAccessAnalysis::setToEntryState(PendingStateLattice *lattice) {
  (void)lattice;
}

std::optional<PendingAccessAnalysis::AliasBucketSet>
PendingAccessAnalysis::getAliasBucketsForValue(Operation *op, Value query) {
  const auto *aliasState = getAliasState(op, query);
  if (!aliasState || aliasState->isUndefined())
    return std::nullopt;

  AliasBucketSet result;
  if (aliasState->isUnknown()) {
    result.hasUnknown = true;
    return result;
  }
  for (DistinctAttr aliasClass : aliasState->getAliasClasses())
    result.buckets.insert(aliasClass);
  return result;
}

HandleFactsValue
PendingAccessAnalysis::lookupSparseHandleFactsForValue(Operation *op,
                                                       Value query) {
  if (!query || !isHandleType(query.getType()))
    return {};

  if (const auto *lat = getHandleFactsStateForCurrentOp(op, query)) {
    HandleFactsValue facts = lat->getValue();
    if (!facts.unknown && facts.atoms.empty())
      return HandleFactsValue::getAtom(query);
    return facts;
  }
  return HandleFactsValue::getAtom(query);
}

LogicalResult PendingAccessAnalysis::resolveHandleFacts(
    Value value, const PendingStateValue &state, Operation *op,
    HandleFactsValue &out) {
  out = {};

  auto resolveFromLoadSlot = [&](Value slotAddr) -> LogicalResult {
    auto slotBuckets = getAliasBucketsForValue(op, slotAddr);
    if (!slotBuckets)
      return failure();

    if (slotBuckets->hasUnknown)
      (void)out.joinInPlace(state.unknownHandleSlotContents);

    for (AliasBucket bucket : slotBuckets->buckets) {
      auto it = state.handleSlotContents.find(bucket);
      if (it != state.handleSlotContents.end())
        (void)out.joinInPlace(it->second);
    }
    return success();
  };

  if (!value)
    return success();

  if (auto load = value.getDefiningOp<memref::LoadOp>())
    return resolveFromLoadSlot(load.getMemRef());
  if (auto load = value.getDefiningOp<LLVM::LoadOp>())
    return resolveFromLoadSlot(load.getAddr());

  out = lookupSparseHandleFactsForValue(op, value);
  return success();
}

LogicalResult PendingAccessAnalysis::recordHandleStore(Value storedValue,
                                                       Value slotAddr,
                                                       PendingStateValue &state,
                                                       Operation *op) {
  if (!isHandleType(storedValue.getType()))
    return success();

  HandleFactsValue facts;
  if (failed(resolveHandleFacts(storedValue, state, op, facts)))
    return failure();

  auto buckets = getAliasBucketsForValue(op, slotAddr);
  if (!buckets)
    return failure();

  if (buckets->hasUnknown)
    (void)state.unknownHandleSlotContents.joinInPlace(facts);

  for (AliasBucket bucket : buckets->buckets)
    (void)state.handleSlotContents[bucket].joinInPlace(facts);

  return success();
}

LogicalResult
PendingAccessAnalysis::recordHandleStore(Operation *op,
                                         PendingStateValue &state) {
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return recordHandleStore(store.getValueToStore(), store.getMemRef(), state,
                             op);
  if (auto store = dyn_cast<LLVM::StoreOp>(op))
    return recordHandleStore(store.getValue(), store.getAddr(), state, op);
  return success();
}

FailureOr<llvm::SmallVector<MemoryAccess>>
PendingAccessAnalysis::getMemoryAccesses(Operation *op) {
  llvm::SmallVector<MemoryAccess> accesses;
  llvm::SmallDenseSet<Value> seen;

  FailureOr<LevelInfo> execInfo = getExecutionInfoForCurrentOp(op);
  if (failed(execInfo))
    return failure();

  int32_t usedModel = getUsedModel(op).value_or(-1);
  ExecKind execKind = execInfo->execKind;
  llvm::SmallDenseSet<int32_t> ranks = toRankSet(execInfo->executors);

  auto addAccess = [&](Value alias, bool isWrite) {
    if (!seen.insert(alias).second)
      return;
    accesses.emplace_back(op, alias, isWrite, execKind, ranks, usedModel);
  };

  auto memEffectOp = dyn_cast<MemoryEffectOpInterface>(op);

  for (Value operand : op->getOperands()) {
    if (!isPointerLike(operand.getType()))
      continue;

    if (!memEffectOp) {
      addAccess(operand, /*isWrite=*/true);
      continue;
    }

    llvm::SmallVector<MemoryEffects::EffectInstance> effects;
    memEffectOp.getEffectsOnValue(operand, effects);

    bool hasWrite = false, hasRead = false;
    for (const auto &effect : effects) {
      hasWrite |= isa<MemoryEffects::Write>(*effect.getEffect());
      hasRead |= isa<MemoryEffects::Read>(*effect.getEffect());
      if (hasWrite)
        break;
    }

    if (hasRead || hasWrite)
      addAccess(operand, hasWrite);
  }

  return accesses;
}

//===----------------------------------------------------------------------===//
// Core error checking
//===----------------------------------------------------------------------===//

void PendingAccessAnalysis::checkAccessPair(const MemoryAccess &prevAccess,
                                            const MemoryAccess &currAccess,
                                            ErrorSet &errors) {
  const auto *prevState = getAliasState(currAccess.user, prevAccess.alias);
  const auto *currState = getAliasState(currAccess.user, currAccess.alias);

  if (!prevState || !currState || prevState->isUndefined() ||
      currState->isUndefined())
    return;

  if (prevState->alias(*currState) == AliasResult::NoAlias)
    return;

  if (!(prevAccess.isWrite || currAccess.isWrite))
    return;

  auto makeError = [&](llvm::SmallDenseSet<int32_t> overlap) {
    errors.insert(ErrorInfo(prevAccess.user, currAccess.user, prevAccess.alias,
                            currAccess.alias, prevAccess.execKind,
                            currAccess.execKind, std::move(overlap)));
  };

  if (bothStatic(prevAccess.execKind, currAccess.execKind)) {
    auto overlap = calculateOverlapRanks(prevAccess.ranks, currAccess.ranks);
    if (!overlap.empty())
      makeError(std::move(overlap));
    return;
  }

  if (oneStaticOneAllBut(prevAccess.execKind, currAccess.execKind)) {
    auto overlap = calculateOverlapRanks(prevAccess.ranks, currAccess.ranks);
    if (overlap.empty())
      makeError({});
    return;
  }

  makeError({});
}

//===----------------------------------------------------------------------===//
// Transfer function — decomposed into three phases
//===----------------------------------------------------------------------===//

LogicalResult
PendingAccessAnalysis::transferHandleStores(Operation *op,
                                            PendingStateValue &state) {
  return recordHandleStore(op, state);
}

LogicalResult PendingAccessAnalysis::transferMemoryAccesses(
    Operation *op, PendingStateValue &state, ErrorSet &errors) {
  if (!isMemoryInstruction(op))
    return success();

  auto accesses = getMemoryAccesses(op);
  if (failed(accesses))
    return failure();

  for (const MemoryAccess &memAccess : *accesses) {
    auto buckets = getAliasBucketsForValue(op, memAccess.alias);

    // Check against all existing pending records for aliasing conflicts.
    forEachCandidateRecordForAlias(
        state, buckets, [&](const PendingAccessRecord &record) {
          checkAccessPair(record.access, memAccess, errors);
        });

    if (!isSpmdOp(op))
      continue;

    // Resolve window and request handle facts for SPMD operations.
    HandleFactsValue winFacts, reqFacts;

    if (op->hasTrait<spmd::RMA>() && op->hasTrait<spmd::Communication>() &&
        hasWindowOperand(op)) {
      if (failed(resolveHandleFacts(getWindowOperand(op), state, op, winFacts)))
        return failure();
    }

    if (op->hasTrait<spmd::Communication>() &&
        (isNonBlocking(op) || op->hasTrait<spmd::Persistent>()) &&
        hasReqHandleResult(op)) {
      if (failed(
              resolveHandleFacts(getReqHandleResult(op), state, op, reqFacts)))
        return failure();
    }

    // Insert pending-access records keyed by handle atoms.
    auto addRecord = [&](const HandleFactsValue &facts,
                         const HandleFactsValue &opposite) -> LogicalResult {
      if (!facts.unknown && facts.emptyKnown())
        return success();

      PendingAccessRecord record{memAccess, opposite.atoms, opposite.unknown};

      if (facts.unknown || facts.atoms.empty()) {
        state.unknownHandleAccesses.insert(record);
        return success();
      }

      auto memBuckets = getAliasBucketsForValue(op, memAccess.alias);
      if (!memBuckets)
        return failure();

      if (memBuckets->hasUnknown) {
        for (HandleAtom atom : facts.atoms)
          state.unknownAccessesByHandle[atom].insert(record);
        return success();
      }

      for (AliasBucket bucket : memBuckets->buckets)
        for (HandleAtom atom : facts.atoms)
          state.accessMapByAlias[bucket][atom].insert(record);

      return success();
    };

    if (failed(addRecord(winFacts, reqFacts)) ||
        failed(addRecord(reqFacts, winFacts)))
      return failure();
  }

  return success();
}

LogicalResult
PendingAccessAnalysis::transferSyncResets(Operation *op,
                                          PendingStateValue &state) {
  // spmd.parallel and spmd.finalize is always a full sync point
  if (isa<spmd::FinalizeOp, spmd::ParallelOp>(op)) {
    state.accessMapByAlias.clear();
    state.unknownAccessesByHandle.clear();
    state.unknownHandleAccesses.clear();
    return success();
  }
  if (isRMASyncOp(op)) {
    if (hasWindowOperand(op)) {
      HandleFactsValue syncFacts;
      if (failed(
              resolveHandleFacts(getWindowOperand(op), state, op, syncFacts)))
        return failure();
      resetBySyncHandleFacts(state, syncFacts);
    } else if (auto model = getUsedModel(op)) {
      resetByUsedModel(state, *model);
    }
  }

  if (isNonBlockingSyncOp(op) && hasReqHandleOperand(op)) {
    HandleFactsValue syncFacts;
    if (failed(
            resolveHandleFacts(getReqHandleOperand(op), state, op, syncFacts)))
      return failure();
    resetBySyncHandleFacts(state, syncFacts);
  }

  return success();
}

LogicalResult PendingAccessAnalysis::applyOperationTransfer(
    Operation *op, PendingStateValue &state, ErrorSet &errors) {

  // Phase 1: Record handle stores into slot-content maps.
  if (failed(transferHandleStores(op, state)))
    return failure();

  // Phase 2: Check memory accesses against pending records and insert new ones.
  if (failed(transferMemoryAccesses(op, state, errors)))
    return failure();

  // Phase 3: Apply synchronization resets (RMA fence / non-blocking wait).
  if (failed(transferSyncResets(op, state)))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// Visitor entry points
//===----------------------------------------------------------------------===//

static size_t countRecords(const PendingStateValue &state) {
  size_t n = state.unknownHandleAccesses.size();
  for (auto &[b, byHandle] : state.accessMapByAlias)
    for (auto &[h, recs] : byHandle)
      n += recs.size();
  for (auto &[h, recs] : state.unknownAccessesByHandle)
    n += recs.size();
  return n;
}

LogicalResult
PendingAccessAnalysis::visitOperation(Operation *op,
                                      const PendingStateLattice &before,
                                      PendingStateLattice *after) {
  PendingStateValue next = before.getValue();

  if (failed(applyOperationTransfer(op, next, errors)))
    return success();

  // gpu.launch has no RegionBranchOpInterface, so the framework never
  // feeds the body's exit state back to the op's successors in main.
  // Manually join it here. Using getProgramPointAfter(op) as the anchor
  // registers a dependency: when the terminator state changes (fixpoint
  // iteration), this op will be re-visited and re-propagated.
  for (Region &region : op->getRegions()) {
    // ModuleOp is the solver's root; its region is not a sequential
    // code path relative to any caller.
    if (isa<ModuleOp>(op))
      break;

    // Functions are handled via visitCallControlFlowTransfer — joining
    // their body exit state here would double-count and misattribute.
    if (isa<FunctionOpInterface>(op))
      break;
    // For spmd.parallel the join point IS the sync — body state is reset
    // by transferSyncResets, do not propagate pending records outward.
    if (isa<spmd::ParallelOp>(op))
      break;

    if (region.empty())
      continue;
    Block &body = region.front();

    // ── NEW: push current state INTO the body's entry ──────────────────
    // gpu.launch has no RegionBranchOpInterface, so the framework never
    // seeds the body's entry from the outer state. Do it manually here so
    // that pending records (e.g. a spmd.get in main) are visible when
    // analyzing ops inside the launch body (e.g. spmd.put in the kernel).
    Operation *firstOp = &body.front();
    auto *entryLattice =
        getOrCreate<PendingStateLattice>(getProgramPointBefore(firstOp));
    PendingStateLattice entryTmp(entryLattice->getAnchor());
    entryTmp.getValue() = next; // state after processing the launch op itself
    propagateIfChanged(entryLattice, entryLattice->join(entryTmp));
    // ────────────────────────────────────────────────────────────────────

    if (body.empty() || !body.back().mightHaveTrait<OpTrait::IsTerminator>())
      continue;
    Operation *terminator = &body.back();

    const auto *exitState = getOrCreateFor<PendingStateLattice>(
        getProgramPointAfter(op), // anchor → re-visits this op when changed
        getProgramPointAfter(terminator) // query the exit state of the body
    );
    if (exitState)
      (void)next.joinInPlace(exitState->getValue());
  }

  PendingStateLattice tmp(after->getAnchor());
  tmp.getValue() = std::move(next);
  propagateIfChanged(after, after->join(tmp));
  return success();
}

void PendingAccessAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const PendingStateLattice &before, PendingStateLattice *after) {
  if (action != dataflow::CallControlFlowAction::ExternalCallee) {
    Base::visitCallControlFlowTransfer(call, action, before, after);
    return;
  }

  PendingStateValue next = before.getValue();

  if (failed(applyOperationTransfer(call.getOperation(), next, errors)))
    return;

  PendingStateLattice tmp(after->getAnchor());
  tmp.getValue() = std::move(next);
  propagateIfChanged(after, after->join(tmp));
}

} // namespace mlir::spmd