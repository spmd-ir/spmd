// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/CommOriginsAnalysis.h"
#include "spmd/SPMDTypes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::spmd {
namespace {

/// Types that can carry a comm identity through the cast/memref chain:
///   - !spmd.comm  (direct)
///   - !llvm.ptr   (boxed comm)
///   - memref<N x !llvm.ptr>  (pointer2memref view of a boxed comm)
static bool isCommCarrierType(Type type) {
  if (isa<spmd::CommType>(type))
    return true;
  if (isa<LLVM::LLVMPointerType>(type))
    return true;
  if (auto mr = dyn_cast<MemRefType>(type))
    return isa<LLVM::LLVMPointerType>(mr.getElementType());
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() == 32;
  return false;
}

static bool createsFreshComm(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && dialect->getNamespace() == "spmd";
}

/// Merge facts from all operands whose type is a comm carrier into \p out.
static HandleFactsValue
mergeCarrierOperands(Operation *op,
                     ArrayRef<const HandleFactsLattice *> operands) {
  HandleFactsValue merged;
  for (auto [idx, operand] : llvm::enumerate(op->getOperands())) {
    if (!isCommCarrierType(operand.getType()))
      continue;
    (void)merged.joinInPlace(operands[idx]->getValue());
  }
  return merged;
}

} // namespace

void CommOriginsAnalysis::setToEntryState(HandleFactsLattice *lattice) {
  // Non-comm-carrier anchors need no initialisation.
  if (!isCommCarrierType(lattice->getAnchor().getType()))
    return;
  // Conservatively mark as unknown until the defining op is visited.
  propagateIfChanged(lattice, lattice->join(HandleFactsValue::getUnknown()));
}

LogicalResult CommOriginsAnalysis::visitOperation(
    Operation *op, ArrayRef<const HandleFactsLattice *> operands,
    ArrayRef<HandleFactsLattice *> results) {

  // ── Case 1: transparent propagation ops ──────────────────────────────────
  // unrealized_conversion_cast  (!spmd.comm <-> !llvm.ptr round-trips)
  // enzymexla.pointer2memref    (!llvm.ptr -> memref<N x !llvm.ptr>)
  // enzymexla.memref2pointer    (reverse)
  // arith.select                (picks between two comm carriers at runtime)
  //
  // TODO(precision, rank-aware-handles): the arith.select case produces a
  // flat union of origin atoms across both arms. This is sound but loses
  // per-rank attribution. See the matching TODO in HandleOriginsAnalysis.
  bool isTransparent =
      isa<UnrealizedConversionCastOp, arith::SelectOp>(op) ||
      op->getName().getStringRef() == "enzymexla.pointer2memref" ||
      op->getName().getStringRef() == "enzymexla.memref2pointer";

  if (isTransparent) {
    HandleFactsValue merged = mergeCarrierOperands(op, operands);
    for (auto [idx, result] : llvm::enumerate(op->getResults())) {
      if (!isCommCarrierType(result.getType()))
        continue;
      if (idx < results.size() && results[idx])
        propagateIfChanged(results[idx], results[idx]->join(merged));
    }
    return success();
  }

  // ── Case 2: SPMD ops that create a fresh communicator ────────────────────
  if (createsFreshComm(op)) {
    for (auto [idx, result] : llvm::enumerate(op->getResults())) {
      if (!isa<spmd::CommType>(result.getType()))
        continue;
      propagateIfChanged(results[idx],
                         results[idx]->join(HandleFactsValue::getAtom(result)));
    }
    return success();
  }

  return success();
}

void CommOriginsAnalysis::visitExternalCall(
    CallOpInterface call,
    ArrayRef<const HandleFactsLattice *> /*argumentLattices*/,
    ArrayRef<HandleFactsLattice *> resultLattices) {
  // External calls that produce a comm (e.g. ncclCommInitRank wrapper) are
  // marked unknown — conservative but sound.
  for (auto [idx, result] : llvm::enumerate(call->getResults())) {
    if (!isa<spmd::CommType>(result.getType()))
      continue;
    if (idx < resultLattices.size() && resultLattices[idx])
      propagateIfChanged(
          resultLattices[idx],
          resultLattices[idx]->join(HandleFactsValue::getUnknown()));
  }
}

} // namespace mlir::spmd