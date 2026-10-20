// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/CommBuffers.h"

#include "spmd/SPMDTraits.h"
#include "spmd/SPMDTypes.h"
#include "spmd/Utils/MemoryOffsets.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::spmd {

//===----------------------------------------------------------------------===//
// Classification
//===----------------------------------------------------------------------===//

bool isDataMovingCollective(Operation *op) {
  return isa<CommBufferOpInterface>(op) && op->hasTrait<Collective>();
}

bool hasStreamOperand(Operation *op) {
  return llvm::any_of(op->getOperandTypes(),
                      [](Type type) { return isa<StreamType>(type); });
}

bool isNonBlocking(Operation *op) {
  return llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<ReqType>(type); });
}

std::optional<int64_t> getDatatypeByteSize(Value datatype,
                                           const DataLayout &dl) {
  auto constant =
      datatype ? datatype.getDefiningOp<ConstDatatypeOp>() : ConstDatatypeOp();
  if (!constant)
    return std::nullopt;
  return getFixedTypeByteSize(constant.getTypeAttr(), dl);
}

//===----------------------------------------------------------------------===//
// Memory effects
//===----------------------------------------------------------------------===//

LogicalResult getProgramMemoryEffects(
    Operation *op, SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  auto effectOp = dyn_cast<MemoryEffectOpInterface>(op);
  if (!effectOp)
    return failure();

  SmallVector<MemoryEffects::EffectInstance> all;
  effectOp.getEffects(all);
  // Program memory is MLIR's DefaultResource itself, which is what MemRead and
  // MemWrite on an operand and the LLVM and memref ops declare. Compared by
  // identity: isa<DefaultResource> would also accept every SPMD resource, all
  // of which have DefaultResource as their parent.
  for (const MemoryEffects::EffectInstance &effect : all)
    if (effect.getResource() == SideEffects::DefaultResource::get())
      effects.push_back(effect);
  return success();
}

LogicalResult collectWrittenAddresses(Operation *op,
                                      SmallVectorImpl<Value> &written) {
  SmallVector<MemoryEffects::EffectInstance> effects;
  if (failed(getProgramMemoryEffects(op, effects)))
    return failure();

  for (const MemoryEffects::EffectInstance &effect : effects) {
    if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
      continue;
    Value address = effect.getValue();
    if (!address || !isAddressType(address.getType()))
      return failure();
    if (!llvm::is_contained(written, address))
      written.push_back(address);
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Buffers
//===----------------------------------------------------------------------===//

bool isPrivateAllocation(Value value) {
  return isa_and_nonnull<LLVM::AllocaOp, memref::AllocaOp, memref::AllocOp>(
      value.getDefiningOp());
}

bool isPointerAnnotation(Operation *op) {
  return isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp, LLVM::InvariantStartOp,
             LLVM::InvariantEndOp, LLVM::DbgDeclareOp, LLVM::DbgValueOp>(op);
}

LogicalResult computeBufferFacts(Value buffer, const DataLayout &dl,
                                 BufferFacts &facts) {
  Value allocation = resolveBufferSlice(buffer, dl).base;
  if (!isPrivateAllocation(allocation))
    return failure();
  facts.allocation = allocation;

  SmallVector<Value> worklist{allocation};
  while (!worklist.empty()) {
    Value address = worklist.pop_back_val();
    if (!facts.aliases.insert(address))
      continue;

    for (OpOperand &use : address.getUses()) {
      Operation *user = use.getOwner();

      // An address computed from this one: follow it.
      std::optional<AddressStep> step = getAddressStep(user, dl);
      if (step && step->source == address) {
        for (Value result : user->getResults())
          if (isAddressType(result.getType()))
            worklist.push_back(result);
        continue;
      }

      if (isPointerAnnotation(user))
        continue;

      SmallVector<MemoryEffects::EffectInstance> effects;
      if (failed(getProgramMemoryEffects(user, effects)))
        return failure();

      bool described = false;
      for (const MemoryEffects::EffectInstance &effect : effects) {
        Value affected = effect.getValue();
        // An effect on some other address says nothing about this one, but
        // one with no address at all may be on it.
        if (affected && affected != address)
          continue;
        described |= affected == address;
        if (isa<MemoryEffects::Read>(effect.getEffect()))
          facts.readers.insert(user);
        if (isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
          facts.writers.insert(user);
      }
      // The address is an operand without a declared effect on it, so the op
      // may keep it - a store of the pointer itself, a call argument.
      if (!described)
        return failure();
    }
  }
  return success();
}

bool readsBuffer(Operation *op, const BufferFacts &facts) {
  return llvm::any_of(facts.readers, [&](Operation *reader) {
    return op == reader || op->isProperAncestor(reader);
  });
}

bool writesBuffer(Operation *op, const BufferFacts &facts) {
  return llvm::any_of(facts.writers, [&](Operation *writer) {
    return op == writer || op->isProperAncestor(writer);
  });
}

//===----------------------------------------------------------------------===//
// Accesses away from the op
//===----------------------------------------------------------------------===//

bool accessesMemoryAwayFromOp(Operation *op) {
  return isNonBlocking(op) || op->hasTrait<Persistent>() || op->hasTrait<RMA>();
}

bool isAccessedAwayFromOps(const BufferFacts &facts) {
  return llvm::any_of(facts.readers, accessesMemoryAwayFromOp) ||
         llvm::any_of(facts.writers, accessesMemoryAwayFromOp);
}

bool isInThreadTeam(Operation *op) {
  for (auto parallel = op->getParentOfType<ParallelOp>(); parallel;
       parallel = parallel->getParentOfType<ParallelOp>())
    if (!parallel.getIsProcessLevel())
      return true;
  return false;
}

} // namespace mlir::spmd
