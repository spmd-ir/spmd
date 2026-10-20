// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/CommCompletions.h"

#include "spmd/Analysis/CommBuffers.h"
#include "spmd/SPMDOps.h"
#include "spmd/SPMDTypes.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/ErrorHandling.h"

namespace mlir::spmd {

namespace {

/// True when none of `op`'s results is observed.
bool resultsUnobserved(Operation *op) {
  return llvm::all_of(op->getResults(),
                      [](Value result) { return result.use_empty(); });
}

/// The request `op` produces, or null when it produces none.
Value getRequest(Operation *op) {
  for (Value result : op->getResults())
    if (isa<ReqType>(result.getType()))
      return result;
  return {};
}

/// A value of `type` can hold a request: the request type itself, or the i32
/// the raiser casts a request to before storing it.
bool canCarryRequest(Type type) {
  return isa<ReqType>(type) || type.isInteger(32);
}

/// The value `store` stores and the address it stores it to, when `op` is a
/// store.
std::optional<std::pair<Value, Value>> getStore(Operation *op) {
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return std::make_pair(store.getValueToStore(), store.getMemRef());
  if (auto store = dyn_cast<LLVM::StoreOp>(op))
    return std::make_pair(store.getValue(), store.getAddr());
  return std::nullopt;
}

/// The value `op` loads, when it is a load.
Value getLoaded(Operation *op) {
  if (isa<memref::LoadOp, LLVM::LoadOp>(op))
    return op->getResult(0);
  return {};
}

/// Follows requests through the IR, collecting everything they reach.
///
/// Forwards, every use of a request must be understood: a completion, a cast
/// or select passing it on, or a store into a request array. Backwards, every
/// request that meets one of those must come from a communication op: a
/// completion of a request array completes whatever was stored into it. What
/// is stored into an array is whatever any load of it may yield, since slots
/// are not told apart. Anything else fails.
class RequestClosure {
public:
  explicit RequestClosure(const DataLayout &dl) : dl(dl) {}

  LogicalResult followUses(Value request);

  CommCompletions result;

private:
  LogicalResult followOrigin(Value value);
  LogicalResult followPassOn(Operation *op);
  LogicalResult followArray(Value address);
  LogicalResult addCompletion(Operation *op);

  const DataLayout &dl;
  llvm::DenseSet<Value> forwardVisited;
  llvm::DenseSet<Value> backwardVisited;
  llvm::DenseSet<Value> arrays;
};

LogicalResult RequestClosure::addCompletion(Operation *op) {
  // An inspected error code, status, flag or index makes the completion
  // observable.
  if (!resultsUnobserved(op))
    return failure();
  result.completions.insert(op);
  return success();
}

LogicalResult RequestClosure::followUses(Value request) {
  if (!forwardVisited.insert(request).second)
    return success();

  for (OpOperand &use : request.getUses()) {
    Operation *user = use.getOwner();

    if (isa<WaitOp, ReqFreeOp>(user)) {
      if (failed(addCompletion(user)))
        return failure();
      continue;
    }

    // A cast or select passes the request on. A select's condition is no
    // request.
    if (isa<UnrealizedConversionCastOp>(user) ||
        (isa<arith::SelectOp>(user) && use.getOperandNumber() != 0)) {
      if (failed(followPassOn(user)))
        return failure();
      continue;
    }

    if (std::optional<std::pair<Value, Value>> store = getStore(user)) {
      if (store->first != request)
        return failure(); // the request used as an address
      result.carriers.insert(user);
      if (failed(followArray(store->second)))
        return failure();
      continue;
    }

    // spmd.test, waitAny, waitSome, a comparison, a call, a return: the
    // request goes somewhere whose effect on it is not all-or-nothing.
    return failure();
  }
  return success();
}

LogicalResult RequestClosure::followPassOn(Operation *op) {
  if (op->getNumResults() != 1 || !canCarryRequest(op->getResult(0).getType()))
    return failure();
  result.carriers.insert(op);
  // Whatever else flows into the result goes where the result goes.
  for (OpOperand &operand : op->getOpOperands()) {
    if (isa<arith::SelectOp>(op) && operand.getOperandNumber() == 0)
      continue;
    if (failed(followOrigin(operand.get())))
      return failure();
  }
  return followUses(op->getResult(0));
}

LogicalResult RequestClosure::followOrigin(Value value) {
  if (!backwardVisited.insert(value).second)
    return success();

  Operation *def = value.getDefiningOp();
  if (!def || !canCarryRequest(value.getType()))
    return failure(); // an argument, or no request at all

  if (value == getRequest(def) && isSpmdOp(def)) {
    result.initiations.insert(def);
    return followUses(value);
  }
  if (isa<UnrealizedConversionCastOp, arith::SelectOp>(def))
    return followPassOn(def);
  if (value == getLoaded(def)) {
    result.carriers.insert(def);
    if (failed(followArray(def->getOperand(0))))
      return failure();
    return followUses(value);
  }
  return failure();
}

LogicalResult RequestClosure::followArray(Value address) {
  BufferFacts facts;
  if (failed(computeBufferFacts(address, dl, facts)))
    return failure();
  if (!arrays.insert(facts.allocation).second)
    return success();

  for (Operation *reader : facts.readers) {
    if (Value loaded = getLoaded(reader)) {
      result.carriers.insert(reader);
      if (failed(followUses(loaded)))
        return failure();
      continue;
    }
    if (isa<WaitAllOp>(reader)) {
      if (failed(addCompletion(reader)))
        return failure();
      continue;
    }
    return failure();
  }

  for (Operation *writer : facts.writers) {
    if (std::optional<std::pair<Value, Value>> store = getStore(writer)) {
      result.carriers.insert(writer);
      if (failed(followOrigin(store->first)))
        return failure();
      continue;
    }
    if (isa<WaitAllOp>(writer))
      continue; // already a completion
    return failure();
  }
  return success();
}

} // namespace

FailureOr<CommCompletions> getCommCompletions(Operation *op,
                                              const DataLayout &dl) {
  RequestClosure closure(dl);
  closure.result.initiations.insert(op);
  if (Value request = getRequest(op))
    if (failed(closure.followUses(request)))
      return failure();
  return std::move(closure.result);
}

llvm::SetVector<Operation *> collectRemovals(SmallVectorImpl<Operation *> &ops,
                                             const DataLayout &dl) {
  SmallVector<std::optional<CommCompletions>> closures;
  for (Operation *op : ops) {
    FailureOr<CommCompletions> closure = getCommCompletions(op, dl);
    closures.push_back(succeeded(closure) ? std::optional(std::move(*closure))
                                          : std::nullopt);
  }

  // An op goes only if every initiation its requests share a completion with
  // goes too. Dropping one op can strand another, so shrink to a fixpoint.
  llvm::SmallPtrSet<Operation *, 8> kept(ops.begin(), ops.end());
  for (bool shrunk = true; shrunk;) {
    shrunk = false;
    for (auto [op, closure] : llvm::zip(ops, closures)) {
      if (!kept.contains(op))
        continue;
      if (closure && llvm::all_of(closure->initiations, [&](Operation *other) {
            return kept.contains(other);
          }))
        continue;
      kept.erase(op);
      shrunk = true;
    }
  }

  llvm::SetVector<Operation *> removals;
  SmallVector<Operation *> remaining;
  for (auto [op, closure] : llvm::zip(ops, closures)) {
    if (!kept.contains(op))
      continue;
    remaining.push_back(op);
    removals.insert(closure->initiations.begin(), closure->initiations.end());
    removals.insert(closure->completions.begin(), closure->completions.end());
    removals.insert(closure->carriers.begin(), closure->carriers.end());
  }
  ops = std::move(remaining);
  return removals;
}

void eraseOps(RewriterBase &rewriter, llvm::SetVector<Operation *> ops) {
  while (!ops.empty()) {
    SmallVector<Operation *> unused;
    for (Operation *op : ops)
      if (resultsUnobserved(op))
        unused.push_back(op);
    if (unused.empty())
      llvm::reportFatalInternalError(
          "eraseOps: a result of an op to erase is used outside the set");
    for (Operation *op : unused) {
      ops.remove(op);
      rewriter.eraseOp(op);
    }
  }
}

} // namespace mlir::spmd
