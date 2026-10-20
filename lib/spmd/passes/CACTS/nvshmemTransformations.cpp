// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "nvshmemTransformations.h"
#include "common.h"
#include "nvshmemSpmdEntries.h"
#include "spmd/SPMDOps.h"
#include "spmd/TargetABI/NVSHMEM.h"
#include "utils.h"

namespace mlir::spmd {

Value getNewReduceOpNVSHMEM(std::string reduceOp, IRMapping &valueMapping,
                            Operation *currentCallOpr, int modelGroup) {
  Value newReduceOp;
  static bool sumCreated = false;
  static Value sumReduceOp;
  static bool incCreated = false;
  static Value incReduceOp;
  if (reduceOp == "sum") {
    if (sumCreated) {
      return sumReduceOp;
    } else {
      OpBuilder builder(currentCallOpr);
      LLVM::LLVMFuncOp llvmFuncOp =
          currentCallOpr->getParentOfType<LLVM::LLVMFuncOp>();
      builder.setInsertionPointToStart(&(llvmFuncOp.getBody().front()));
      newReduceOp = spmd::ConstOpTypeOp::create(builder, llvmFuncOp->getLoc(),
                                                StringRef("SUM"), modelGroup)
                        .getResult();
      sumCreated = true;
      sumReduceOp = newReduceOp;
    }
  } else if (reduceOp == "inc") {
    if (incCreated) {
      return incReduceOp;
    } else {
      OpBuilder builder(currentCallOpr);
      LLVM::LLVMFuncOp llvmFuncOp =
          currentCallOpr->getParentOfType<LLVM::LLVMFuncOp>();
      builder.setInsertionPointToStart(&(llvmFuncOp.getBody().front()));
      newReduceOp = spmd::ConstOpTypeOp::create(builder, llvmFuncOp->getLoc(),
                                                StringRef("INC"), modelGroup)
                        .getResult();
      incCreated = true;
      incReduceOp = newReduceOp;
    }
  } else {
    llvm::reportFatalUsageError(
        "\nNot implemented case of explicit reduceOp\n");
  }
  return newReduceOp;
}

Value getNewSignalOpNVSHMEM(Value signalOpArg, IRMapping &valueMapping,
                            Operation *currentCallOpr, int modelGroup) {
  Value newSignalOp;
  if (valueMapping.contains(newSignalOp)) {
    newSignalOp = valueMapping.lookup(signalOpArg);
  } else if (isArithConstantIntX(signalOpArg, 9)) { // SHMEM_SIGNAL_SET case
    OpBuilder builder(currentCallOpr);
    LLVM::LLVMFuncOp llvmFuncOp =
        currentCallOpr->getParentOfType<LLVM::LLVMFuncOp>();
    builder.setInsertionPointToStart(&(llvmFuncOp.getBody().front()));
    newSignalOp =
        spmd::ConstSignalOpTypeOp::create(builder, llvmFuncOp->getLoc(),
                                          StringRef("SET"), modelGroup)
            .getResult();
  } else {
    llvm::reportFatalUsageError("\nNot implemented case of getNewSignalOp\n");
  }
  return newSignalOp;
}

Value getNewDatatypeNVSHMEM(Value datatypeArg, IRMapping &valueMapping,
                            int modelGroup, bool datatypeGiven = false,
                            std::string datatype = "",
                            Operation *currentCallOpr = nullptr) {
  Value newDatatype;
  static bool intCreated = false;
  static Value intDatatype;
  static bool byteCreated = false;
  static Value byteDatatype;
  if (datatypeGiven) {
    if (datatype == "int") {
      if (intCreated) {
        return intDatatype;
      } else {
        OpBuilder builder(currentCallOpr);
        LLVM::LLVMFuncOp llvmFuncOp =
            currentCallOpr->getParentOfType<LLVM::LLVMFuncOp>();
        builder.setInsertionPointToStart(&(llvmFuncOp.getBody().front()));
        newDatatype =
            spmd::ConstDatatypeOp::create(
                builder, llvmFuncOp->getLoc(),
                builder.getIntegerType(32, /*isSigned=*/true), modelGroup)
                .getResult();
        intCreated = true;
        intDatatype = newDatatype;
      }
    } else if (datatype == "byte") {
      if (byteCreated) {
        return byteDatatype;
      } else {
        OpBuilder builder(currentCallOpr);
        LLVM::LLVMFuncOp llvmFuncOp =
            currentCallOpr->getParentOfType<LLVM::LLVMFuncOp>();
        builder.setInsertionPointToStart(&(llvmFuncOp.getBody().front()));
        newDatatype =
            spmd::ConstDatatypeOp::create(builder, llvmFuncOp->getLoc(),
                                          builder.getI8Type(), modelGroup)
                .getResult();
        byteCreated = true;
        byteDatatype = newDatatype;
      }
    } else {
      llvm::reportFatalUsageError(
          "\nNot implemented case of explicit datatype\n");
    }
  } else if (valueMapping.contains(datatypeArg)) {
    newDatatype = valueMapping.lookup(datatypeArg);
  } else {
    llvm::reportFatalUsageError(
        "\nNot implemented case of datatype replacement\n");
  }
  return newDatatype;
}

static Value handleSingleCommArgNVSHMEM(Value origCommArg, int modelGroup) {
  if (isa<spmd::CommType>(origCommArg.getType()))
    return origCommArg;

  auto constantOp = origCommArg.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    llvm::reportFatalUsageError(
        "\nNot implemented case of comm replacement - unknown defining op\n");

  auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!integerAttr)
    llvm::reportFatalUsageError(
        "\nNot implemented case of comm replacement - unknown attribute\n");

  OpBuilder builder(constantOp);
  builder.setInsertionPoint(constantOp.getOperation());
  auto loc = constantOp->getLoc();

  switch (integerAttr.getInt()) {
  case abi::nvshmem::kTeamWorld:
    return getWorldComm(builder, loc, modelGroup);
  case abi::nvshmem::kTeamNode:
    return getNodeComm(builder, loc, modelGroup);
  default:
    llvm::reportFatalUsageError(
        "\nNot implemented case of comm replacement - const comm unknown\n");
  }
}

Value getNewCommNVSHMEM(Value commArg, int modelGroup) {
  return getNewComm(commArg, modelGroup, handleSingleCommArgNVSHMEM);
}

std::optional<Value>
handleNVSHMEMOperand(const llvm::json::Array &params, StringRef kind,
                     const llvm::json::Object &paramObj, Value oldOperand,
                     unsigned operandIdx, LLVM::CallOp llvmCallOp,
                     OpBuilder &builder, Location loc, unsigned modelGroup) {
  if (kind == "TEAM")
    return std::optional(getNewCommNVSHMEM(oldOperand, modelGroup));
  if (kind == "THREAD_LEVEL") {
    // NVSHMEM uses the same 0-3 encoding as MPI/SHMEM.
    return std::optional<Value>(getThreadLevelFromInt(oldOperand, modelGroup));
  }

  LLVM_DEBUG(DBGS() << "Not Implemented yet: handleSHMEMOperand " << kind
                    << "\n");

  return std::nullopt;
}

} // namespace mlir::spmd