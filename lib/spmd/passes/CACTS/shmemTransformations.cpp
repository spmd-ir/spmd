// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "shmemTransformations.h"
#include "common.h"
#include "shmemSpmdEntries.h"
#include "spmd/SPMDOps.h"
#include "utils.h"

namespace mlir::spmd {

// Value getNewReduceOpSHMEM(std::string reduceOp, int modelGroup){
//   OpBuilder builder(currentCallOpr);

//   builder.setInsertionPointToStart(&(llvmFuncOp.getBody().front()));
//   Value newReduceOp = spmd::ConstOpTypeOp::create(builder,
//   llvmFuncOp->getLoc(), reduceOp, modelGroup).getResult(); return
//   newReduceOp;
// }

Value getNewSignalOpSHMEM(Value signalOpArg, int modelGroup) {
  Value newSignalOp;
  auto defOpr = signalOpArg.getDefiningOp();
  if (defOpr && isArithConstantIntX(signalOpArg, 0)) { // SHMEM_SIGNAL_SET case
    OpBuilder builder(defOpr);
    newSignalOp = spmd::ConstSignalOpTypeOp::create(
                      builder, defOpr->getLoc(), StringRef("SET"), modelGroup)
                      .getResult();
  } else {
    llvm::reportFatalUsageError(
        "\nNot implemented case of explicit reduceOp\n");
  }
  return newSignalOp;
}

// Scans users of a memref value for a memref::StoreOp storing a spmd::CommType.
static Value findStoredCommType(Value memrefVal) {
  for (Operation *user : memrefVal.getUsers()) {
    if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
      Value stored = storeOp.getValue();

      // Direct store of CommType.
      if (isa<spmd::CommType>(stored.getType()))
        return stored;

      // CommType cast to !llvm.ptr via unrealized_conversion_cast before store.
      // This happens when handleNewResults stores into a memref<1x!llvm.ptr>.
      if (auto castOp = stored.getDefiningOp<UnrealizedConversionCastOp>()) {
        if (castOp->getNumOperands() == 1 &&
            isa<spmd::CommType>(castOp->getOperand(0).getType()))
          return castOp->getOperand(0);
      }
    }
  }
  return Value();
}

// Given an llvm.alloca, finds a pointer2memref derived from it and scans for a
// stored CommType.
static Value findStoredCommTypeFromAlloca(Value allocaVal) {
  for (Operation *user : allocaVal.getUsers()) {
    if (auto p2m = dyn_cast<enzymexla::Pointer2MemrefOp>(user)) {
      if (Value found = findStoredCommType(p2m.getResult()))
        return found;
    }
    if (auto storeOp = dyn_cast<memref::StoreOp>(user)) {
      Value stored = storeOp.getValue();
      if (isa<spmd::CommType>(stored.getType()))
        return stored;
    }
  }
  return Value();
}

static Value handleSingleCommArgSHMEM(Value origCommArg, int modelGroup) {
  if (isa<spmd::CommType>(origCommArg.getType()))
    return origCommArg;

  if (auto getGlobalOp = origCommArg.getDefiningOp<memref::GetGlobalOp>()) {
    if (getGlobalOp.getName() != "SHMEM_TEAM_WORLD")
      llvm::reportFatalUsageError(
          "\nNot implemented: comm replacement - const comm unknown\n");
    OpBuilder builder(getGlobalOp);
    LLVM::LLVMFuncOp fn = getGlobalOp->getParentOfType<LLVM::LLVMFuncOp>();
    builder.setInsertionPointToStart(&fn.getBody().front());
    return getWorldComm(builder, fn->getLoc(), modelGroup);
  }

  if (auto addrOfOp = origCommArg.getDefiningOp<LLVM::AddressOfOp>()) {
    if (addrOfOp.getGlobalName() != "SHMEM_TEAM_WORLD")
      llvm::reportFatalUsageError(
          "\nNot implemented: comm replacement - const comm unknown\n");
    OpBuilder builder(addrOfOp);
    LLVM::LLVMFuncOp fn = addrOfOp->getParentOfType<LLVM::LLVMFuncOp>();
    builder.setInsertionPointToStart(&fn.getBody().front());
    return getWorldComm(builder, fn->getLoc(), modelGroup);
  }

  // Split team loaded from alloca-backed memref.
  // storeValueIntoMemref creates a NEW
  // pointer2memref(%alloca)->memref<1xCommType>. We must peel back through
  // memrefVal's defining p2m to the shared alloca, then scan ALL pointer2memref
  // siblings for the CommType store.
  if (auto loadOp = origCommArg.getDefiningOp<memref::LoadOp>()) {
    Value memrefVal = loadOp.getMemRef();

    // Fast path: unlikely since storeValueIntoMemref creates a fresh p2m,
    // but handles the case where the store is on this exact memref.
    if (Value found = findStoredCommType(memrefVal))
      return found;

    // Peel memrefVal (not origCommArg!) through pointer2memref to the alloca.
    if (auto p2m = memrefVal.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
      if (Value found = findStoredCommTypeFromAlloca(p2m.getOperand()))
        return found;
    }

    llvm::reportFatalUsageError(
        "\nCould not find stored CommType for memref.load — "
        "was shmem_team_split_strided converted first?\n");
  }

  // pointer2memref used directly as comm arg.
  if (auto p2m = origCommArg.getDefiningOp<enzymexla::Pointer2MemrefOp>()) {
    if (Value found = findStoredCommType(origCommArg))
      return found;
    if (Value found = findStoredCommTypeFromAlloca(p2m.getOperand()))
      return found;
    llvm::reportFatalUsageError(
        "\nCould not find stored CommType via pointer2memref — "
        "was shmem_team_split_strided converted first?\n");
  }

  // Raw alloca.
  if (isa_and_nonnull<LLVM::AllocaOp>(origCommArg.getDefiningOp())) {
    if (Value found = findStoredCommTypeFromAlloca(origCommArg))
      return found;
    llvm::reportFatalUsageError(
        "\nCould not find stored CommType from alloca — "
        "was shmem_team_split_strided converted first?\n");
  }

  LLVM_DEBUG(DBGS() << "Unhandled origCommArg defining op: ");
  LLVM_DEBUG(if (Operation *d = origCommArg.getDefiningOp()) d->dump();
             else llvm::dbgs() << "<block argument>\n";);
  llvm::reportFatalUsageError("\nNot implemented case of comm replacement\n");
}

Value getNewCommSHMEM(Value commArg, int modelGroup) {
  return getNewComm(commArg, modelGroup, handleSingleCommArgSHMEM);
}

std::optional<Value>
handleSHMEMOperand(const llvm::json::Array &params, StringRef kind,
                   const llvm::json::Object &paramObj, Value oldOperand,
                   unsigned operandIdx, LLVM::CallOp llvmCallOp,
                   OpBuilder &builder, Location loc, unsigned modelGroup) {
  if (kind == "TEAM")
    return std::optional(getNewCommSHMEM(oldOperand, modelGroup));

  if (kind == "SIG_OP") {
    return std::optional<Value>(getNewSignalOpSHMEM(oldOperand, modelGroup));
  }
  if (kind == "THREAD_LEVEL") {
    return std::optional<Value>(getThreadLevelFromInt(oldOperand, modelGroup));
  }

  LLVM_DEBUG(DBGS() << "Not Implemented yet: handleSHMEMOperand " << kind
                    << "\n");

  return std::nullopt;
}
} // namespace mlir::spmd