// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "mpiTransformations.h"
#include "mpiSpmdEntries.h"
#include "spmd/SPMDOps.h"
#include "spmd/TargetABI/MPICH.h"
#include "spmd/Utils/PassUtils.h"

#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"

#include "mlir/IR/IRMapping.h"

namespace mlir::spmd {

Value getNewOpTypeMPI(Value reduceOpArg, int modelGroup) {
  auto constantOp = reduceOpArg.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    llvm::reportFatalUsageError(
        "\nNot implemented case of reduceOp replacement\n");
  auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!integerAttr)
    llvm::reportFatalUsageError(
        "\nNot implemented case of reduceOp replacement\n");

  std::optional<StringRef> opName =
      abi::mpich::lookupName(abi::mpich::kReduceOps, integerAttr.getInt());
  if (!opName)
    llvm::reportFatalUsageError("\nNot implemented case of reduceOp "
                                "replacement - const reduceOp unknown\n");

  OpBuilder builder(constantOp);
  return spmd::ConstOpTypeOp::create(builder, constantOp->getLoc(), *opName,
                                     modelGroup)
      .getResult();
}

Value getNewLockTypeMPI(Value lockTypeArg, int modelGroup) {
  auto constantOp = lockTypeArg.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    llvm::reportFatalUsageError(
        "\nNot implemented case of non-constant lock type\n");
  auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!integerAttr)
    llvm::reportFatalUsageError(
        "\nNot implemented case of lock type replacement\n");

  std::optional<StringRef> lockName =
      abi::mpich::lookupName(abi::mpich::kLockTypes, integerAttr.getInt());
  if (!lockName)
    llvm::reportFatalUsageError("\nNot implemented case of lock type "
                                "replacement - const lock type unknown\n");

  OpBuilder builder(constantOp);
  return spmd::ConstLockTypeOp::create(builder, constantOp->getLoc(), *lockName,
                                       modelGroup)
      .getResult();
}

Value getNewDatatypeMPI(Value datatypeArg, int modelGroup) {
  auto constantOp = datatypeArg.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    llvm::reportFatalUsageError(
        "\nNot implemented case of datatype replacement\n");
  auto integerAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!integerAttr)
    llvm::reportFatalUsageError(
        "\nNot implemented case of datatype replacement\n");

  Type type = abi::mpich::getDatatypeType(constantOp.getContext(),
                                          integerAttr.getInt());
  if (!type)
    llvm::reportFatalUsageError("\nNot implemented case of datatype "
                                "replacement - const datatype unknown\n");

  OpBuilder builder(constantOp);
  return spmd::ConstDatatypeOp::create(builder, constantOp->getLoc(), type,
                                       modelGroup)
      .getResult();
}

// Keep only the model-specific part:
static Value handleSingleCommArgMPI(Value origCommArg, int modelGroup) {
  auto constantOp = origCommArg.getDefiningOp<arith::ConstantOp>();
  if (!constantOp)
    return origCommArg; // non-world, already-split communicator

  auto intAttr = dyn_cast<IntegerAttr>(constantOp.getValueAttr());
  if (!intAttr)
    llvm::reportFatalUsageError("\nNot implemented case of comm replacement\n");

  if (intAttr.getInt() != kMpiCommWorldHandle)
    llvm::reportFatalUsageError(
        "\nNot implemented case of comm replacement - const comm unknown\n");

  OpBuilder builder(constantOp);
  builder.setInsertionPoint(constantOp.getOperation());
  return getWorldComm(builder, constantOp->getLoc(), modelGroup);
}

// Public entry point for MPI:
Value getNewCommMPI(Value commArg, int modelGroup) {
  return getNewComm(commArg, modelGroup, handleSingleCommArgMPI);
}

bool isStatusIgnore(Value statusArg) {
  // Check if the operation is an LLVM `inttoptr`
  auto intToPtrOp = statusArg.getDefiningOp<LLVM::IntToPtrOp>();
  if (!intToPtrOp)
    return false;

  // Get the operand of the `inttoptr` operation
  Value operand = intToPtrOp.getOperand();

  // Check if the operand is defined by an `arith.constant` operation
  auto constantOp = operand.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constantOp)
    return false;

  // Check if the constant value is `1` and its type is `i64`
  auto intAttr = dyn_cast<mlir::IntegerAttr>(constantOp.getValue());
  if (!intAttr || intAttr.getValue() != 1)
    return false;

  // if (!intAttr.getType().isSignlessInteger(64))
  //   return false;

  return true;
}

// Detects MPI_STATUSES_IGNORE (NULL / inttoptr(0)).
// isStatusIgnore already covers the scalar MPI_STATUS_IGNORE (inttoptr(1)).
static bool isNullPtr(Value v) {
  if (auto itp = v.getDefiningOp<LLVM::IntToPtrOp>()) {
    if (auto cst = itp.getOperand().getDefiningOp<arith::ConstantOp>())
      if (auto attr = dyn_cast<IntegerAttr>(cst.getValue()))
        return attr.getValue() == 0;
  }
  if (v.getDefiningOp<LLVM::ZeroOp>())
    return true;
  return false;
}

bool isInPlace(Value sendBuf) {
  // Check if the operation is an LLVM `inttoptr`
  auto intToPtrOp = sendBuf.getDefiningOp<LLVM::IntToPtrOp>();
  if (!intToPtrOp)
    return false;

  // Get the operand of the `inttoptr` operation
  Value operand = intToPtrOp.getOperand();

  // Check if the operand is defined by an `arith.constant` operation
  auto constantOp = operand.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constantOp)
    return false;

  // Check if the constant value is `-1` and its type is `i64`
  auto intAttr = dyn_cast<mlir::IntegerAttr>(constantOp.getValue());
  if (!intAttr || intAttr.getValue() != -1)
    return false;

  // if (!intAttr.getType().isSignlessInteger(64))
  //   return false;

  return true;
}

bool checkForSendAndRecvBufAndGetNonRootIdx(const llvm::json::Array &params,
                                            unsigned rootBufIdx,
                                            unsigned &nonRootIdx) {
  bool hasSendBuf = false;
  bool hasRecvBuf = false;
  unsigned idx = 0;
  for (const auto &param : params) {
    if (auto *paramObj = param.getAsObject()) {
      auto name = getParamString(paramObj, "name");
      if (!name)
        llvm::reportFatalUsageError("json is not accessible or parameter "
                                    "attribute not specified for this call");

      if (*name == "sendbuf") {
        hasSendBuf = true;
        if (idx != rootBufIdx) {
          nonRootIdx = idx;
        }
      } else if (*name == "recvbuf") {
        hasRecvBuf = true;
        if (idx != rootBufIdx) {
          nonRootIdx = idx;
        }
      }
    }
    idx++;
  }
  return hasSendBuf && hasRecvBuf;
}

Value checkAndGetInPlace(Value rootBuf, unsigned rootBufIdx,
                         const llvm::json::Array &params,
                         LLVM::CallOp &llvmCallOp, int modelGroup) {
  unsigned nonRootIdx = 0;
  if (!checkForSendAndRecvBufAndGetNonRootIdx(params, rootBufIdx, nonRootIdx))
    return rootBuf;

  if (isInPlace(rootBuf)) {
    return llvmCallOp.getOperand(nonRootIdx);
  }
  return rootBuf;
}

std::optional<Value>
handleMPIOperand(const llvm::json::Array &params, StringRef kind,
                 const llvm::json::Object &paramObj, Value oldOperand,
                 unsigned operandIdx, LLVM::CallOp llvmCallOp,
                 OpBuilder &builder, Location loc, unsigned modelGroup) {

  if (kind == "DATATYPE") {
    return std::optional<Value>(getNewDatatypeMPI(oldOperand, modelGroup));
  }

  if (kind == "LOCK_TYPE") {
    IRMapping valueMapping;
    return std::optional<Value>(getNewLockTypeMPI(oldOperand, modelGroup));
  }

  if (kind == "OPERATION") {
    return std::optional<Value>(getNewOpTypeMPI(oldOperand, modelGroup));
  }

  if (kind == "BUFFER") {
    auto rootOnly = getParamBool(&paramObj, "root_only");
    if (!rootOnly)
      llvm::reportFatalUsageError("missing root_only");

    if (*rootOnly)
      return std::optional<Value>(checkAndGetInPlace(
          oldOperand, operandIdx, params, llvmCallOp, modelGroup));

    return std::optional<Value>(oldOperand);
  }

  if (kind == "STATUS") {
    // Pass absent Optional for both ignore sentinels:
    //   MPI_STATUS_IGNORE   = inttoptr(1)   — scalar wait/test ops
    //   MPI_STATUSES_IGNORE = null pointer  — array waitall/testall ops
    if (isStatusIgnore(oldOperand) || isNullPtr(oldOperand))
      return std::optional<Value>(Value());

    if (!isa<LLVM::LLVMPointerType>(oldOperand.getType()))
      return std::nullopt; // unexpected type, fall through to shared handler

    // "length" in the JSON param means this is an array of statuses
    // (MPI_Waitall, MPI_Waitsome, MPI_Testall, MPI_Testsome).
    // Mirror exactly what the REQUEST case does for array requests.
    auto length = getParamString(&paramObj, "length");
    if (length.has_value()) {
      auto statusMemrefType = MemRefType::get(
          {ShapedType::kDynamic}, builder.getType<spmd::StatusType>());
      return std::optional<Value>(
          UnrealizedConversionCastOp::create(builder, loc, statusMemrefType,
                                             oldOperand)
              .getResult(0));
    }

    // Scalar status as operand (unusual; scalar status is normally a
    // result via R() not an operand via O(), but handle gracefully).
    return loadValueFromMemref(oldOperand, builder.getType<spmd::StatusType>(),
                               builder, loc);
  }

  if (kind == "COMMUNICATOR") {
    // Value newComm = getNewCommMPI(oldOperand, modelGroup);
    // if (isa<spmd::CommType>(newComm.getType())) {
    //   return std::optional<Value>(newComm);
    // }
    // else
    if (oldOperand.getType().isInteger(
            32)) { // Type checking could be replcaed by looking up the
                   // param_direction in json file
      // (newComm(here == oldOperand) is not commWorld, and was perhabs newly
      // created e.g. by commSplit
      LLVM_DEBUG(DBGS() << "comm operand is int32, create unrealized "
                           "conversion to commType\n");
      return std::optional<Value>(
          UnrealizedConversionCastOp::create(
              builder, loc, builder.getType<spmd::CommType>(), oldOperand)
              .getResult(0));
    } else if (isa<LLVM::LLVMPointerType>(oldOperand.getType())) {
      LLVM_DEBUG(
          DBGS()
          << "comm operand is pointer, create load from memref to commType\n");
      auto paramDirection = getParamString(&paramObj, "direction");
      if (paramDirection.has_value()) {
        // Communicator was used by address, no load needed only cast
        return std::optional<Value>(
            UnrealizedConversionCastOp::create(
                builder, loc, builder.getType<spmd::CommType>(), oldOperand)
                .getResult(0));
      } else { // null
        // Communicator was used by address, no load needed only cast
        return loadValueFromMemref(
            oldOperand, builder.getType<spmd::CommType>(), builder, loc);
      }
    } else {
      LLVM_DEBUG(DBGS() << "comm expecting operand has not the type int32, "
                           "LLVMPointerType, spmd.commType"
                        << "\n");
    }
  }

  if (kind == "WINDOW") {
    // Type checking could be replcaed by looking up the param_direction in json
    // file
    if (oldOperand.getType().isInteger(32))
      return std::optional<Value>(
          UnrealizedConversionCastOp::create(
              builder, loc, builder.getType<spmd::WinType>(), oldOperand)
              .getResult(0));

    if (isa<LLVM::LLVMPointerType>(oldOperand.getType()))
      return loadValueFromMemref(oldOperand, builder.getType<spmd::WinType>(),
                                 builder, loc);

    LLVM_DEBUG(
        DBGS()
        << "window expecting operand has not the type int32 or LLVMPointerType"
        << "\n");
  }

  if (kind == "REQUEST") {
    // Type checking could be replcaed by looking up the param_direction in json
    // file
    if (oldOperand.getType().isInteger(32))
      return std::optional<Value>(
          UnrealizedConversionCastOp::create(
              builder, loc, builder.getType<spmd::ReqType>(), oldOperand)
              .getResult(0));
    if (isa<LLVM::LLVMPointerType>(oldOperand.getType())) {
      auto length = getParamString(&paramObj, "length");
      if (length.has_value()) {
        // Array (e.g. Waitall): cast to typed memref for data race analysis.
        auto reqMemrefType = MemRefType::get({ShapedType::kDynamic},
                                             builder.getType<spmd::ReqType>());
        return std::optional<Value>(UnrealizedConversionCastOp::create(
                                        builder, loc, reqMemrefType, oldOperand)
                                        .getResult(0));
      } else { // null
        // Scalar: load the single request written by Irecv/Isend.
        return loadValueFromMemref(oldOperand, builder.getType<spmd::ReqType>(),
                                   builder, loc);
      }
    }
    LLVM_DEBUG(DBGS() << "request expecting operand has not the type "
                         "int32 or LLVMPointerType\n");
  }

  if (kind == "MESSAGE") {
    // Type checking could be replcaed by looking up the param_direction in json
    // file
    if (oldOperand.getType().isInteger(32))
      return std::optional<Value>(
          UnrealizedConversionCastOp::create(
              builder, loc, builder.getType<spmd::MsgType>(), oldOperand)
              .getResult(0));

    if (isa<LLVM::LLVMPointerType>(oldOperand.getType())) {
      return loadValueFromMemref(oldOperand, builder.getType<spmd::MsgType>(),
                                 builder, loc);
    }

    LLVM_DEBUG(
        DBGS()
        << "request expecting operand has not the type int32 or LLVMPointerType"
        << "\n");
  }

  if (kind == "THREAD_LEVEL") {
    return std::optional<Value>(getThreadLevelFromInt(oldOperand, modelGroup));
  }

  return std::nullopt;
}

void replaceReqNull(PatternRewriter &rewriter, Operation *spmdOp, Value reqArg,
                    int modelGroup, LogicalResult &logicRes) {

  auto usedModelOpt = getUsedModel(spmdOp);
  if (!usedModelOpt || usedModelOpt.value() != modelGroup)
    return;

  for (Operation *user : reqArg.getUsers()) {

    auto castOp = dyn_cast<UnrealizedConversionCastOp>(user);
    if (!castOp)
      continue;

    if (!castOp.getResult(0).getType().isInteger(32))
      continue;

    Value i32Val = castOp.getResult(0);

    // Step 1: find memref.store of i32Val
    for (Operation *i32User : i32Val.getUsers()) {

      auto firstStore = dyn_cast<memref::StoreOp>(i32User);
      if (!firstStore)
        continue;

      LLVM_DEBUG(DBGS() << "Found memref.store of i32Val: " << *firstStore
                        << "\n");

      Value memrefA = firstStore.getMemRef();
      // Step 2: memrefA must come from Pointer2MemrefOp
      auto ptr2memA = memrefA.getDefiningOp<enzymexla::Pointer2MemrefOp>();
      if (!ptr2memA)
        continue;

      Value llvmPtr = ptr2memA.getOperand(); // underlying llvm.ptr

      // Step 3: find users of llvmPtr
      for (Operation *ptrUser : llvmPtr.getUsers()) {

        auto ptr2memB = dyn_cast<enzymexla::Pointer2MemrefOp>(ptrUser);
        if (!ptr2memB)
          continue;

        LLVM_DEBUG(DBGS() << "Found second Pointer2MemrefOp: " << *ptr2memB
                          << "\n");
        Value memrefB = ptr2memB.getResult();
        // Step 4: find magic memref.store
        for (Operation *memUser : memrefB.getUsers()) {

          auto magicStore = dyn_cast<memref::StoreOp>(memUser);
          if (!magicStore)
            continue;

          auto constOp =
              magicStore.getValue().getDefiningOp<arith::ConstantOp>();
          if (!constOp)
            continue;

          auto attr = dyn_cast<IntegerAttr>(constOp.getValue());
          if (!attr || attr.getInt() != abi::mpich::kRequestNull)
            continue;

          LLVM_DEBUG(DBGS()
                     << "Found magic memref.store: " << *magicStore << "\n");

          auto loc = magicStore.getLoc();
          rewriter.setInsertionPoint(magicStore);

          auto nullReq = rewriter.create<spmd::RequestNullOp>(loc, modelGroup);

          auto castToI32 = rewriter.create<UnrealizedConversionCastOp>(
              loc, rewriter.getI32Type(), nullReq.getResult());

          rewriter.replaceOpWithNewOp<memref::StoreOp>(
              magicStore, castToI32.getResult(0), memrefB,
              magicStore.getIndices());

          logicRes = success();
        }
      }
    }
  }
}
struct ConvertMpiRequestNull : public OpRewritePattern<LLVM::LLVMFuncOp> {
  ConvertMpiRequestNull(MLIRContext *ctx, const ApiRegistryEntry &entry)
      : OpRewritePattern(ctx), entry(entry) {}

  LogicalResult matchAndRewrite(LLVM::LLVMFuncOp func,
                                PatternRewriter &rewriter) const override {

    MLIRContext *ctx = rewriter.getContext();
    auto reqType = spmd::ReqType::get(ctx);

    LogicalResult logicRes = failure();
    func.walk([&](Operation *opr) {
      if (opr->getDialect()->getNamespace().str() != "spmd") {
        return;
      }
      for (Value result : opr->getResults()) {
        if (result.getType() != reqType)
          continue;
        else {
          LLVM_DEBUG(DBGS()
                     << "Found spmd op with req result: " << *opr << "\n");
          replaceReqNull(rewriter, opr, result, entry.modelGroup, logicRes);
          break;
        }
      }

      for (Value operand : opr->getOperands()) {
        if (operand.getType() != reqType)
          continue;
        else {
          LLVM_DEBUG(DBGS()
                     << "Found spmd op with req operand: " << *opr << "\n");
          replaceReqNull(rewriter, opr, operand, entry.modelGroup, logicRes);
          break;
        }
      }
    });
    return succeeded(logicRes) ? success() : failure();
  }

private:
  const ApiRegistryEntry &entry;
};

struct MaterializeMpiCommWorld : public OpRewritePattern<arith::ConstantOp> {
  MaterializeMpiCommWorld(MLIRContext *ctx, int modelGroup)
      : OpRewritePattern(ctx), modelGroup(modelGroup) {}

  LogicalResult matchAndRewrite(arith::ConstantOp cst,
                                PatternRewriter &rewriter) const override {
    auto intAttr = dyn_cast<IntegerAttr>(cst.getValue());
    if (!intAttr || !intAttr.getType().isInteger(32) ||
        intAttr.getInt() != kMpiCommWorldHandle)
      return failure();

    Location loc = cst.getLoc();
    Value cw = rewriter.create<spmd::CommWorldOp>(loc, modelGroup).getResult();
    Value asI32 =
        rewriter
            .create<UnrealizedConversionCastOp>(loc, rewriter.getI32Type(), cw)
            .getResult(0);
    rewriter.replaceOp(cst, asI32);
    return success();
  }
  int modelGroup;
};

void populateMPIPatterns(RewritePatternSet &patterns,
                         const ApiRegistryEntry &entry) {
  patterns.add<ConvertMpiRequestNull>(patterns.getContext(), entry);
  patterns.add<MaterializeMpiCommWorld>(patterns.getContext(),
                                        entry.modelGroup);
}

} // namespace mlir::spmd