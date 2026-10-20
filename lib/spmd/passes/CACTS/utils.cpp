// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "utils.h"

#define DEBUG_TYPE "utils"

#define PATTERN "main"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE ":" << PATTERN << "] "

namespace mlir::spmd {

std::string toLowerStr(StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s)
    result.push_back(llvm::toLower(c));
  return result;
}

std::string toUpperStr(llvm::StringRef s) {
  std::string result;
  result.reserve(s.size());
  for (char c : s)
    result.push_back(llvm::toUpper(c));
  return result;
}

std::optional<std::string> getParamString(const llvm::json::Object *obj,
                                          llvm::StringRef key) {
  if (!obj)
    return std::nullopt;

  if (auto *val = obj->get(key))
    if (auto str = val->getAsString())
      return std::string(*str);

  return std::nullopt;
}

std::optional<bool> getParamBool(const llvm::json::Object *obj,
                                 llvm::StringRef key) {
  if (!obj)
    return std::nullopt;

  if (auto *val = obj->get(key))
    if (auto boolean = val->getAsBoolean())
      return *boolean;

  return std::nullopt;
}

bool isArithConstantIntX(Value value, int x) {
  // Check if the defining operation is an arith.constant
  if (auto constOp = value.getDefiningOp<arith::ConstantOp>()) {
    // Get the constant value
    auto attr = constOp.getValue();

    // Check if it's an IntegerAttr
    if (auto intAttr = dyn_cast<IntegerAttr>(attr)) {
      // Compare the integer value to 0
      return intAttr.getValue() == x;
    }

    // Check if it's a FloatAttr
    // if (auto floatAttr = attr.dyn_cast<mlir::FloatAttr>()) {
    //     // Compare the float value to 0.0
    //     return floatAttr.getValue().isZero();
    // }
  }
  return false;
}

void getOriginalCommArgImpl(Value commArg, SmallVector<Value> &origCommArgs,
                            SmallVector<Operation *> &branchLikeOps,
                            SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(commArg).second)
    return;

  // Already-transformed: the defining op is an spmd op producing a CommType.
  // This happens when an earlier MPI call in the same walk (e.g. commSplit)
  // was already converted before we reach this call. Just return the value.
  if (isa<spmd::CommType>(commArg.getType())) {
    origCommArgs.push_back(commArg);
    return;
  }

  Operation *defOpr = commArg.getDefiningOp();
  if (!defOpr) { // block argument
    origCommArgs.push_back(commArg);
    return;
  }

  // ── Transparent: chase through the pointer/memref/cast chain ─────────────
  if (auto loadOp = dyn_cast<LLVM::LoadOp>(defOpr)) {
    getOriginalCommArgImpl(loadOp.getAddr(), origCommArgs, branchLikeOps,
                           visited);
  } else if (auto loadOp = dyn_cast<memref::LoadOp>(defOpr)) {
    getOriginalCommArgImpl(loadOp.getMemRef(), origCommArgs, branchLikeOps,
                           visited);
  } else if (isa<enzymexla::Pointer2MemrefOp, enzymexla::Memref2PointerOp>(
                 defOpr)) {
    getOriginalCommArgImpl(defOpr->getOperand(0), origCommArgs, branchLikeOps,
                           visited);
  } else if (auto gepOp = dyn_cast<LLVM::GEPOp>(defOpr)) {
    getOriginalCommArgImpl(gepOp.getBase(), origCommArgs, branchLikeOps,
                           visited);
  } else if (isa<LLVM::BitcastOp, LLVM::AddrSpaceCastOp>(defOpr)) {
    getOriginalCommArgImpl(defOpr->getOperand(0), origCommArgs, branchLikeOps,
                           visited);
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(defOpr)) {
    branchLikeOps.push_back(defOpr);
    getOriginalCommArgImpl(selectOp.getTrueValue(), origCommArgs, branchLikeOps,
                           visited);
    getOriginalCommArgImpl(selectOp.getFalseValue(), origCommArgs,
                           branchLikeOps, visited);
  }

  // ── Terminal: opaque producers — push as-is and stop ─────────────────────
  else if (isa<LLVM::AddressOfOp, LLVM::AllocaOp, arith::ConstantOp>(defOpr)) {
    origCommArgs.push_back(commArg);
  }
  // Any call (malloc, MPI helper, etc.) is an opaque allocation source.
  // We cannot trace further; treat the call result as the origin pointer.
  // at least for calls to malloc etc this behaviour is correct
  // otherwise we could jump into the call
  else if (isa<LLVM::CallOp>(defOpr)) {
    origCommArgs.push_back(commArg);
  }
  // Undef/poison: treat as opaque rather than crashing.
  else if (isa<LLVM::UndefOp, LLVM::PoisonOp>(defOpr)) {
    origCommArgs.push_back(commArg);
  } else {
    defOpr->dump();
    llvm::reportFatalUsageError(
        "\nNot implemented case of getOriginalCommArg!\n");
  }
}

// Todo exchange this with commOriginAnalysis
//  Public entry point keeps the existing signature
void getOriginalCommArg(Value commArg, SmallVector<Value> &origCommArgs,
                        SmallVector<Operation *> &branchLikeOps) {
  SmallPtrSet<Value, 8> visited;
  getOriginalCommArgImpl(commArg, origCommArgs, branchLikeOps, visited);
}

Value convertOldToSPMDComms(SmallVector<Value> origCommArgs,
                            SmallVector<Operation *> branchLikeOps,
                            int modelGroup,
                            const SingleCommHandler &handleSingle) {
  if (origCommArgs.size() == 1)
    return handleSingle(origCommArgs[0], modelGroup);

  if (origCommArgs.size() == 2) {
    auto selectOp = dyn_cast<arith::SelectOp>(branchLikeOps[0]);
    if (!selectOp)
      llvm::reportFatalUsageError(
          "\nNot implemented case of convertOldToSPMDComms!\n");

    Value newComm1 = handleSingle(origCommArgs[0], modelGroup);
    Value newComm2 = handleSingle(origCommArgs[1], modelGroup);
    OpBuilder builder(selectOp);
    builder.setInsertionPoint(selectOp);
    return arith::SelectOp::create(builder, selectOp->getLoc(),
                                   builder.getType<spmd::CommType>(),
                                   selectOp.getCondition(), newComm1, newComm2)
        .getResult();
  }

  llvm::reportFatalUsageError(
      "\nNot implemented case of convertOldToSPMDComms (>2 args)!\n");
}

Value getNewComm(Value commArg, int modelGroup,
                 const SingleCommHandler &handleSingle) {
  SmallVector<Value> origCommArgs;
  SmallVector<Operation *> branchLikeOps;
  getOriginalCommArg(commArg, origCommArgs, branchLikeOps);
  return convertOldToSPMDComms(origCommArgs, branchLikeOps, modelGroup,
                               handleSingle);
}

} // namespace mlir::spmd