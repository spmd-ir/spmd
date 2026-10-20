// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/HandleSlotAnalysis.h"

#include "spmd/SPMDTypes.h"

#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::spmd;

//===----------------------------------------------------------------------===//
// Carriers and slot keys
//===----------------------------------------------------------------------===//

bool mlir::spmd::isHandleCarrierType(Type type, HandleKind kind) {
  switch (kind) {
  case HandleKind::Communicator:
    if (isa<CommType, LLVM::LLVMPointerType>(type))
      return true;
    if (auto memref = dyn_cast<MemRefType>(type))
      return isa<LLVM::LLVMPointerType>(memref.getElementType());
    return false;
  case HandleKind::Request:
    if (isa<ReqType, WinType, LLVM::LLVMPointerType>(type) ||
        type.isInteger(32))
      return true;
    if (auto memref = dyn_cast<MemRefType>(type)) {
      Type element = memref.getElementType();
      return isa<ReqType, WinType, LLVM::LLVMPointerType>(element) ||
             element.isInteger(32);
    }
    return false;
  }
  llvm_unreachable("unhandled HandleKind");
}

/// Casts and bridges that hand the same handle or address on unchanged.
static bool isTransparentForSlots(Operation *op) {
  return isa<enzymexla::Pointer2MemrefOp, enzymexla::Memref2PointerOp,
             UnrealizedConversionCastOp>(op);
}

static Value normalizeSlotKey(Value val, HandleKind kind) {
  while (Operation *defOp = val.getDefiningOp()) {
    if (!isTransparentForSlots(defOp))
      break;
    bool found = false;
    for (Value operand : defOp->getOperands()) {
      if (isHandleCarrierType(operand.getType(), kind)) {
        val = operand;
        found = true;
        break;
      }
    }
    if (!found)
      break;
  }
  return val;
}

//===----------------------------------------------------------------------===//
// HandleSlotValue
//===----------------------------------------------------------------------===//

bool HandleSlotValue::operator==(const HandleSlotValue &other) const {
  if (unknownSlotContents != other.unknownSlotContents)
    return false;
  if (slotContents.size() != other.slotContents.size())
    return false;
  for (const auto &[key, val] : slotContents) {
    auto it = other.slotContents.find(key);
    if (it == other.slotContents.end() || it->second != val)
      return false;
  }
  return true;
}

ChangeResult HandleSlotValue::joinInPlace(const HandleSlotValue &other) {
  ChangeResult changed = ChangeResult::NoChange;
  for (const auto &[slot, otherFacts] : other.slotContents) {
    auto &myFacts = slotContents[slot];
    bool prevUnknown = myFacts.unknown;
    size_t prevSize = myFacts.atoms.size();
    (void)myFacts.joinInPlace(otherFacts);
    if (myFacts.unknown != prevUnknown || myFacts.atoms.size() != prevSize)
      changed = ChangeResult::Change;
  }
  {
    bool prevUnknown = unknownSlotContents.unknown;
    size_t prevSize = unknownSlotContents.atoms.size();
    (void)unknownSlotContents.joinInPlace(other.unknownSlotContents);
    if (unknownSlotContents.unknown != prevUnknown ||
        unknownSlotContents.atoms.size() != prevSize)
      changed = ChangeResult::Change;
  }
  return changed;
}

//===----------------------------------------------------------------------===//
// HandleSlotAnalysis
//===----------------------------------------------------------------------===//

HandleFactsValue
HandleSlotAnalysis::resolveValueFacts(Operation *op, Value val,
                                      const HandleSlotValue &slotState) {
  // Step 1: sparse lattice.
  if (const auto *lat =
          getOrCreateFor<HandleFactsLattice>(getProgramPointAfter(op), val)) {
    const HandleFactsValue &facts = lat->getValue();
    if (facts.unknown || !facts.atoms.empty())
      return facts;
  }

  // Step 2: transparent SSA ops.
  if (Operation *defOp = val.getDefiningOp()) {
    if (isTransparentForSlots(defOp) || isa<arith::SelectOp>(defOp)) {
      HandleFactsValue result;
      for (Value operand : defOp->getOperands())
        if (isHandleCarrierType(operand.getType(), kind))
          (void)result.joinInPlace(resolveValueFacts(op, operand, slotState));
      return result;
    }
  }

  // Step 3: load → slot lookup.
  Value slotAddr;
  if (auto load = val.getDefiningOp<memref::LoadOp>())
    slotAddr = load.getMemRef();
  else if (auto load = val.getDefiningOp<LLVM::LoadOp>())
    slotAddr = load.getAddr();
  if (!slotAddr)
    return {};

  Value normalizedSlot = normalizeSlotKey(slotAddr, kind);
  HandleFactsValue result;
  (void)result.joinInPlace(slotState.unknownSlotContents);
  auto it = slotState.slotContents.find(normalizedSlot);
  if (it == slotState.slotContents.end())
    it = slotState.slotContents.find(slotAddr);
  if (it != slotState.slotContents.end())
    (void)result.joinInPlace(it->second);
  return result;
}

LogicalResult HandleSlotAnalysis::visitOperation(
    Operation *op, const HandleSlotLattice &before, HandleSlotLattice *after) {
  propagateIfChanged(after, after->join(before));

  Value storedVal, slotAddr;
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    storedVal = store.getValueToStore();
    slotAddr = store.getMemRef();
  } else if (auto store = dyn_cast<LLVM::StoreOp>(op)) {
    storedVal = store.getValue();
    slotAddr = store.getAddr();
  } else {
    return success();
  }

  if (!isHandleCarrierType(storedVal.getType(), kind))
    return success();

  HandleFactsValue facts = resolveValueFacts(op, storedVal, before.getValue());
  if (!facts.unknown && facts.atoms.empty())
    return success();

  Value normalizedSlot = normalizeSlotKey(slotAddr, kind);
  HandleSlotValue updated = before.getValue();
  if (facts.unknown)
    (void)updated.unknownSlotContents.joinInPlace(facts);
  else
    (void)updated.slotContents[normalizedSlot].joinInPlace(facts);

  HandleSlotLattice tmp(after->getAnchor());
  tmp.getValue() = std::move(updated);
  propagateIfChanged(after, after->join(tmp));
  return success();
}

void HandleSlotAnalysis::visitCallControlFlowTransfer(
    CallOpInterface call, dataflow::CallControlFlowAction action,
    const HandleSlotLattice &before, HandleSlotLattice *after) {

  if (action == dataflow::CallControlFlowAction::ExternalCallee ||
      action == dataflow::CallControlFlowAction::ExitCallee) {
    propagateIfChanged(after, after->join(before));
    return;
  }

  // EnterCallee: remap caller slot keys → callee block argument keys.
  assert(action == dataflow::CallControlFlowAction::EnterCallee);

  auto sym = call.getCallableForCallee().dyn_cast<SymbolRefAttr>();
  if (!sym) {
    propagateIfChanged(after, after->join(before));
    return;
  }

  auto func = dyn_cast_or_null<FunctionOpInterface>(
      SymbolTable::lookupNearestSymbolFrom(call.getOperation(), sym));
  if (!func || func.isExternal() || func.getFunctionBody().empty()) {
    propagateIfChanged(after, after->join(before));
    return;
  }

  Block &entryBlock = func.getFunctionBody().front();
  ArrayRef<BlockArgument> calleeArgs = entryBlock.getArguments();
  OperandRange callArgs = call.getArgOperands();

  HandleSlotValue remapped;
  remapped.unknownSlotContents = before.getValue().unknownSlotContents;

  for (auto [callArg, calleeArg] : llvm::zip(callArgs, calleeArgs)) {
    if (!isHandleCarrierType(callArg.getType(), kind))
      continue;
    Value normalizedCallArg = normalizeSlotKey(callArg, kind);
    for (const auto &[slotKey, slotFacts] : before.getValue().slotContents)
      if (normalizeSlotKey(slotKey, kind) == normalizedCallArg)
        (void)remapped.slotContents[calleeArg].joinInPlace(slotFacts);
  }

  HandleSlotLattice tmp(after->getAnchor());
  tmp.getValue() = std::move(remapped);
  propagateIfChanged(after, after->join(tmp));
}

//===----------------------------------------------------------------------===//
// Resolution after solving
//===----------------------------------------------------------------------===//

HandleFactsValue mlir::spmd::resolveHandleFacts(DataFlowSolver &solver,
                                                Value val, Operation *contextOp,
                                                HandleKind kind) {
  // Step 1: sparse lattice.
  if (auto *lat = solver.lookupState<HandleFactsLattice>(val)) {
    const HandleFactsValue &facts = lat->getValue();
    if (facts.unknown || !facts.atoms.empty())
      return facts;
  }

  // Step 2: transparent SSA ops.
  if (Operation *defOp = val.getDefiningOp()) {
    if (isTransparentForSlots(defOp) || isa<arith::SelectOp>(defOp)) {
      HandleFactsValue result;
      for (Value operand : defOp->getOperands())
        if (isHandleCarrierType(operand.getType(), kind))
          (void)result.joinInPlace(
              resolveHandleFacts(solver, operand, contextOp, kind));
      return result;
    }
  }

  // Step 3: load → HandleSlotLattice at contextOp.
  Value slotAddr;
  if (auto load = val.getDefiningOp<memref::LoadOp>())
    slotAddr = load.getMemRef();
  else if (auto load = val.getDefiningOp<LLVM::LoadOp>())
    slotAddr = load.getAddr();
  if (!slotAddr)
    return {};

  const auto *slotLattice = solver.lookupState<HandleSlotLattice>(
      solver.getProgramPointBefore(contextOp));
  if (!slotLattice)
    return {};

  const HandleSlotValue &slotVal = slotLattice->getValue();
  Value normalizedSlot = normalizeSlotKey(slotAddr, kind);

  HandleFactsValue result;
  (void)result.joinInPlace(slotVal.unknownSlotContents);
  auto it = slotVal.slotContents.find(normalizedSlot);
  if (it == slotVal.slotContents.end())
    it = slotVal.slotContents.find(slotAddr);
  if (it != slotVal.slotContents.end()) {
    (void)result.joinInPlace(it->second);
  } else if (isHandleCarrierType(normalizedSlot.getType(), kind)) {
    // Double-indirection: the slot address is itself a loaded value.
    (void)result.joinInPlace(
        resolveHandleFacts(solver, normalizedSlot, contextOp, kind));
  }
  return result;
}
