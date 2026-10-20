// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- AvailableCollectives.cpp - Available-collective dataflow -*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "spmd/Analysis/AvailableCollectives.h"

#include "spmd/Analysis/CommBuffers.h"
#include "spmd/Analysis/ExecutionContextAnalysis.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/MemoryOffsets.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// CommKey
//===----------------------------------------------------------------------===//

void CommKey::print(raw_ostream &os) const {
  os << kind << "(model=" << usedModel << ", comm=" << comm
     << ", send=" << sendAlloc << ", recv=" << recvAlloc << ", shape=[";
  llvm::interleaveComma(shape, os);
  os << "])";
}

/// The buffer's allocation, or a null Value when the buffer does not denote
/// that allocation's base address.
static Value getBaseAllocation(Value buffer, const DataLayout &dl) {
  BufferSlice slice = resolveBufferSlice(buffer, dl);
  bool preservesBase = slice.offset && *slice.offset == ByteOffset();
  return preservesBase ? slice.base : Value{};
}

std::optional<CommKey> getCommKey(Operation *op, const DataLayout &dl) {
  // Stream-ordered collectives complete relative to a stream rather than to
  // program order, so nothing about their timing can be read off the CFG.  A
  // non-blocking collective's write lands at its completion, not here, so the
  // fact would be attached to the wrong program point.
  if (!isDataMovingCollective(op) || hasStreamOperand(op) || isNonBlocking(op))
    return std::nullopt;

  auto commOp = cast<CommBufferOpInterface>(op);
  Value sendBuf = commOp.getSendBuffer();
  Value recvBuf = commOp.getRecvBuffer();
  if (!sendBuf || !recvBuf)
    return std::nullopt;

  Value sendAlloc = getBaseAllocation(sendBuf, dl);
  Value recvAlloc = getBaseAllocation(recvBuf, dl);
  if (!sendAlloc || !recvAlloc)
    return std::nullopt;

  // MPI_IN_PLACE: the collective reads what it writes, so repeating it reduces
  // the already-reduced value.  Never available.
  if (sendAlloc == recvAlloc)
    return std::nullopt;

  std::optional<int> model = getUsedModel(op);
  if (!model)
    return std::nullopt;

  CommKey key;
  key.kind = op->getName().getIdentifier();
  key.usedModel = static_cast<int32_t>(*model);
  key.sendAlloc = sendAlloc;
  key.recvAlloc = recvAlloc;

  // Split the operands: the communicator, the buffers (already accounted for by
  // their allocations) and everything else, which describes the shape of the
  // transfer and has to agree operand for operand.
  for (Value operand : op->getOperands()) {
    if (isa<CommType>(operand.getType())) {
      if (key.comm)
        return std::nullopt; // more than one communicator: not a shape we model
      key.comm = operand;
      continue;
    }
    if (isAddressType(operand.getType()))
      continue;
    key.shape.push_back(operand);
  }
  if (!key.comm)
    return std::nullopt;

  return key;
}

//===----------------------------------------------------------------------===//
// The lattice
//===----------------------------------------------------------------------===//

bool AvailableCollectives::contains(const CommKey &key) const {
  return llvm::is_contained(available, key);
}

ChangeResult AvailableCollectives::join(const AbstractDenseLattice &lattice) {
  const auto &rhs = static_cast<const AvailableCollectives &>(lattice);

  // Nothing has come down that path yet, so it constrains nothing.
  if (rhs.bottom)
    return ChangeResult::NoChange;

  if (bottom) {
    bottom = false;
    available = rhs.available;
    return ChangeResult::Change;
  }

  SmallVector<CommKey> kept;
  for (const CommKey &key : available)
    if (rhs.contains(key))
      kept.push_back(key);
  if (kept.size() == available.size())
    return ChangeResult::NoChange;
  available = std::move(kept);
  return ChangeResult::Change;
}

ChangeResult AvailableCollectives::killAll() {
  if (!bottom && available.empty())
    return ChangeResult::NoChange;
  bottom = false;
  available.clear();
  return ChangeResult::Change;
}

ChangeResult AvailableCollectives::kill(
    llvm::function_ref<bool(const CommKey &)> predicate) {
  SmallVector<CommKey> kept;
  for (const CommKey &key : available)
    if (!predicate(key))
      kept.push_back(key);
  if (kept.size() == available.size())
    return ChangeResult::NoChange;
  available = std::move(kept);
  return ChangeResult::Change;
}

ChangeResult AvailableCollectives::gen(const CommKey &key) {
  if (!bottom && contains(key))
    return ChangeResult::NoChange;
  bottom = false;
  if (!contains(key))
    available.push_back(key);
  return ChangeResult::Change;
}

void AvailableCollectives::print(raw_ostream &os) const {
  if (bottom) {
    os << "<bottom>";
    return;
  }
  os << "{";
  llvm::interleaveComma(available, os,
                        [&](const CommKey &key) { key.print(os); });
  os << "}";
}

//===----------------------------------------------------------------------===//
// The analysis
//===----------------------------------------------------------------------===//

void AvailableCollectivesAnalysis::setToEntryState(
    AvailableCollectives *lattice) {
  propagateIfChanged(lattice, lattice->killAll());
}

void AvailableCollectivesAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const AvailableCollectives &before, AvailableCollectives *after) {
  propagateIfChanged(after, after->killAll());
}

ChangeResult
AvailableCollectivesAnalysis::killWrites(Value written,
                                         AvailableCollectives *after) {
  if (!written)
    return after->killAll();

  // Compare against the allocation, not the written pointer: a store through a
  // view of the buffer has to kill the fact about the buffer.
  Value allocation = resolveBufferSlice(written, dl).base;
  return after->kill([&](const CommKey &key) {
    return !aliasAnalysis.alias(allocation, key.sendAlloc).isNo() ||
           !aliasAnalysis.alias(allocation, key.recvAlloc).isNo();
  });
}

bool AvailableCollectivesAnalysis::isLocalBuffer(Value allocation) {
  auto [it, inserted] = localBuffers.try_emplace(allocation, false);
  if (inserted) {
    BufferFacts facts;
    it->second = succeeded(computeBufferFacts(allocation, dl, facts)) &&
                 !isAccessedAwayFromOps(facts);
  }
  return it->second;
}

LogicalResult
AvailableCollectivesAnalysis::visitOperation(Operation *op,
                                             const AvailableCollectives &before,
                                             AvailableCollectives *after) {
  ChangeResult changed = after->join(before);

  //===--------------------------------------------------------------------===//
  // KILL - unconditional.  A write invalidates a fact no matter which ranks
  // perform it: a fact that only some ranks lose is a fact no rank may use.
  //===--------------------------------------------------------------------===//

  // The framework flows a RegionBranchOpInterface op's regions itself and never
  // calls this for it. Any other op with regions - spmd.parallel, say - gets
  // here with the state from before it, and its regions start from the entry
  // state and are never flowed back out, so the writes nested in them would go
  // unseen after the op.
  if (op->getNumRegions() != 0) {
    propagateIfChanged(after, changed | after->killAll());
    return success();
  }

  // An op that does not describe its effects may touch anything.
  SmallVector<MemoryEffects::EffectInstance> effects;
  if (failed(getProgramMemoryEffects(op, effects))) {
    propagateIfChanged(after, changed | after->killAll());
    return success();
  }
  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
      continue;
    Value affected = effect.getValue();
    // An effect on program memory rather than on a value says nothing about
    // which buffer it touches.
    changed |= killWrites(
        affected && isAddressType(affected.getType()) ? affected : Value{},
        after);
  }

  //===--------------------------------------------------------------------===//
  // GEN - only from an operation every rank performs, on operands every rank
  // agrees on.  Runs after the kill so that a collective first invalidates the
  // old contents of its destination and only then publishes the new fact.
  //===--------------------------------------------------------------------===//

  if (std::optional<CommKey> key = getCommKey(op, dl)) {
    if (isLocalBuffer(key->sendAlloc) && isLocalBuffer(key->recvAlloc) &&
        isUniformlyExecuted(op, uniformity) &&
        isUniformValue(key->comm, op, uniformity) &&
        llvm::all_of(key->shape, [&](Value value) {
          return isUniformValue(value, op, uniformity);
        }))
      changed |= after->gen(*key);
  }

  propagateIfChanged(after, changed);
  return success();
}

} // namespace mlir::spmd
