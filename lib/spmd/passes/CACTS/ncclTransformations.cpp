// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ncclTransformations.h"
#include "common.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "ncclSpmdEntries.h"
#include "spmd/SPMDOps.h"
#include "spmd/TargetABI/NCCL.h"
#include "utils.h"

namespace mlir::spmd {
Value getNewStreamNCCL(Value streamArg, IRMapping &valueMapping,
                       int modelGroup) {
  Value newStream;
  Operation *defOpr = streamArg.getDefiningOp();
  if (isa<memref::LoadOp>(defOpr) || isa<LLVM::LoadOp>(defOpr)) {
    Value address = defOpr->getOperand(0);
    if (valueMapping.contains(address)) {
      newStream = valueMapping.lookup(address);
    } else {
      OpBuilder builder(defOpr);
      Operation *allocaOpr = address.getDefiningOp();
      builder.setInsertionPointAfter(allocaOpr);
      newStream = spmd::CastStreamOp::create(builder, allocaOpr->getLoc(),
                                             address, modelGroup)
                      .getResult();
      valueMapping.map(address, newStream);
    }
  } else {
    llvm::reportFatalUsageError(
        "\nNot implemented case of non-LoadOp stream usage\n");
  }
  return newStream;
}

Value getNewOpTypeNCCL(Value reduceOpArg, int modelGroup) {
  auto constantOp = reduceOpArg.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    llvm::reportFatalUsageError(
        "\nNot implemented case of non-constant reduceOp\n");
  auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!integerAttr)
    llvm::reportFatalUsageError(
        "\nNot implemented case of reduceOp replacement\n");

  std::optional<StringRef> opName =
      abi::nccl::lookupName(abi::nccl::kReduceOps, integerAttr.getInt());
  if (!opName)
    llvm::reportFatalUsageError("\nNot implemented case of reduceOp "
                                "replacement - const reduceOp unknown\n");

  OpBuilder builder(constantOp);
  return spmd::ConstOpTypeOp::create(builder, constantOp->getLoc(), *opName,
                                     modelGroup)
      .getResult();
}

Value getNewDatatypeNCCL(Value datatypeArg, int modelGroup) {
  auto constantOp = datatypeArg.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    llvm::reportFatalUsageError(
        "\nNot implemented case of non-constant datatype\n");
  auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!integerAttr)
    llvm::reportFatalUsageError(
        "\nNot implemented case of datatype replacement\n");

  Type type =
      abi::nccl::getDatatypeType(constantOp.getContext(), integerAttr.getInt());
  if (!type)
    llvm::reportFatalUsageError("\nNot implemented case of datatype "
                                "replacement - const datatype unknown\n");

  OpBuilder builder(constantOp);
  return spmd::ConstDatatypeOp::create(builder, constantOp->getLoc(), type,
                                       modelGroup)
      .getResult();
}

static Value handleSingleCommArgNCCL(Value origCommArg, int /*modelGroup*/) {
  return origCommArg;
}

Value getNewCommNCCL(Value commArg, int modelGroup) {
  return getNewComm(commArg, modelGroup, handleSingleCommArgNCCL);
}

std::optional<Value>
handleNCCLOperand(const llvm::json::Array &params, StringRef kind,
                  const llvm::json::Object &paramObj, Value oldOperand,
                  unsigned operandIdx, LLVM::CallOp llvmCallOp,
                  OpBuilder &builder, Location loc, unsigned modelGroup) {
  if (kind == "DATATYPE") {
    return std::optional<Value>(getNewDatatypeNCCL(oldOperand, modelGroup));
  }
  if (kind == "COMMUNICATOR") {
    // Value newComm = getNewCommNCCL(oldOperand, modelGroup);
    // if (isa<spmd::CommType>(newComm.getType())) {
    //   return std::optional<Value>(newComm);
    // }
    // else if (oldOperand.getType().isInteger(32)) {//Type checking could be
    // replcaed by looking up the param_direction in json file
    //   // (newComm(here == oldOperand) is not commWorld, and was perhabs newly
    //   created e.g. by commSplit LLVM_DEBUG(DBGS() << "comm operand is int32,
    //   create unrealized conversion to commType\n"); return
    //   std::optional<Value>(
    //       UnrealizedConversionCastOp::create(
    //           builder, loc, builder.getType<spmd::CommType>(), newComm)
    //           .getResult(0));
    // }
    if (isa<LLVM::LLVMPointerType>(oldOperand.getType())) {
      LLVM_DEBUG(DBGS() << "comm operand is pointer for which is a load should "
                           "already exist, create unrealized cast\n");
      // oldOperand is already ncclComm* (the handle), not a pointer to a slot.
      // A direct type cast is all that is needed.
      return UnrealizedConversionCastOp::create(
                 builder, loc, builder.getType<spmd::CommType>(), oldOperand)
          .getResult(0);
    } else {
      LLVM_DEBUG(
          DBGS() << "comm expecting operand has not the type LLVMPointerType"
                 << "\n");
    }
  }
  if (kind == "STREAM") {
    if (isa<LLVM::LLVMPointerType>(oldOperand.getType())) {
      LLVM_DEBUG(
          DBGS()
          << "comm operand is pointer, create load from memref to commType\n");
      // return loadValueFromMemref(oldOperand,
      // builder.getType<spmd::StreamType>(), builder, loc);
      return spmd::CastStreamOp::create(builder, loc, oldOperand, modelGroup)
          .getResult();
    } else {
      LLVM_DEBUG(
          DBGS() << "comm expecting operand has not the type LLVMPointerType"
                 << "\n");
    }
  }

  if (kind == "OPERATION") {
    return std::optional<Value>(getNewOpTypeNCCL(oldOperand, modelGroup));
  }

  LLVM_DEBUG(DBGS() << "Not Implemented yet: handleNCCLOperand " << kind
                    << "\n");
  return std::nullopt;
}

} // namespace mlir::spmd