// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Analysis/HandleOriginsAnalysis.h"
#include "spmd/SPMDTypes.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::spmd {
namespace {

static bool isFactCarrierType(Type type) {
  if (isHandleType(type))
    return true;
  if (auto intType = dyn_cast<IntegerType>(type))
    return intType.getWidth() == 32;
  return false;
}

static bool createsFreshHandle(Operation *op) {
  Dialect *dialect = op->getDialect();
  return dialect && dialect->getNamespace() == "spmd";
}

/// Ops that transparently forward handle facts from operands to results.
/// arith.select is included because `%r = arith.select %c, %t, %f` produces
/// a value that may carry the facts of either arm at runtime. The flat union
/// is sound but rank-agnostic — see TODO below.
static bool isTransparentForwarder(Operation *op) {
  return isa<UnrealizedConversionCastOp, arith::SelectOp>(op) ||
         op->getName().getStringRef() == "enzymexla.pointer2memref" ||
         op->getName().getStringRef() == "enzymexla.memref2pointer";
}

} // namespace

void HandleOriginsAnalysis::setToEntryState(HandleFactsLattice *lattice) {
  Value anchor = lattice->getAnchor();
  if (!isFactCarrierType(anchor.getType()))
    return;
  propagateIfChanged(lattice, lattice->join(HandleFactsValue::getUnknown()));
}

LogicalResult HandleOriginsAnalysis::visitOperation(
    Operation *op, ArrayRef<const HandleFactsLattice *> operands,
    ArrayRef<HandleFactsLattice *> results) {

  if (isTransparentForwarder(op)) {
    // TODO(precision, rank-aware-handles):
    // The flat union of facts across all fact-carrier operands is sound but
    // loses per-rank attribution. For `arith.select %c, %t, %f`, a handle
    // that is atom A on exec set S (where %c is true) and atom B on the
    // complement currently merges to {A, B} on every rank. In
    // PendingAccessAnalysis this causes MPI_Wait-style sync resets to kill
    // both atoms globally, producing false negatives in race checks on the
    // un-synced ranks. Promote HandleFactsValue::atoms to
    // SmallDenseMap<HandleAtom, MultiValueInfo> and refine exec info at
    // select joins (getConditionOperands already accepts arith::SelectOp).
    HandleFactsValue mergedFacts;
    for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
      if (!isFactCarrierType(operand.getType()))
        continue;
      (void)mergedFacts.joinInPlace(operands[index]->getValue());
    }
    for (auto [index, result] : llvm::enumerate(op->getResults())) {
      if (!isFactCarrierType(result.getType()))
        continue;
      propagateIfChanged(results[index], results[index]->join(mergedFacts));
    }
    return success();
  }

  if (!createsFreshHandle(op))
    return success();

  for (auto [index, result] : llvm::enumerate(op->getResults())) {
    if (isHandleType(result.getType()))
      propagateIfChanged(
          results[index],
          results[index]->join(HandleFactsValue::getAtom(result)));
  }
  return success();
}

void HandleOriginsAnalysis::visitExternalCall(
    CallOpInterface call, ArrayRef<const HandleFactsLattice *> argumentLattices,
    ArrayRef<HandleFactsLattice *> resultLattices) {
  (void)argumentLattices;
  for (auto [index, result] : llvm::enumerate(call->getResults())) {
    if (isHandleType(result.getType()))
      if (index < resultLattices.size() && resultLattices[index])
        propagateIfChanged(
            resultLattices[index],
            resultLattices[index]->join(HandleFactsValue::getUnknown()));
  }
}

} // namespace mlir::spmd