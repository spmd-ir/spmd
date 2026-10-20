// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ncclSpmdEntries.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir::spmd {

//   // if id is send by a collective to comm world, then rankArg represents
//   equivalently rank of commworld of both modelGroups
//   // Todo: for simplicity we just check if sizeArg stems from a commworld and
//   assume that one process is managing one gpu
#define DEBUG_TYPE "convert-apiCalls-to-spmd"
#define DBGS() ::llvm::dbgs() << "[" DEBUG_TYPE "] "

/// Tries to recognise "ncclCommInitRank whose nRanks is the size of the world
/// communicator". Returns false, with a debug note, when a precondition of
/// that pattern does not hold.
static bool ncclInitMatchesWorldSize(Operation *newOpr, Operation *oldOpr) {

  // 1. newOpr must be spmd.commWorld
  auto commWorldOp = dyn_cast<spmd::CommWorldOp>(newOpr);
  if (!commWorldOp) {
    LLVM_DEBUG(
        DBGS() << "ncclInitMatchesWorldSize: newOpr is not spmd.commWorld\n");
    return false;
  }

  // 2. oldOpr must be an llvm.call to "ncclCommInitRank"
  auto callOp = dyn_cast<LLVM::CallOp>(oldOpr);
  if (!callOp || !callOp.getCallee() ||
      callOp.getCallee()->str() != "ncclCommInitRank") {
    LLVM_DEBUG(DBGS() << "ncclInitMatchesWorldSize: oldOpr is not an llvm.call "
                         "to ncclCommInitRank\n");
    return false;
  }

  // 3. ncclCommInitRank must have exactly 4 operands
  if (callOp.getArgOperands().size() != 4) {
    LLVM_DEBUG(DBGS() << "ncclInitMatchesWorldSize: ncclCommInitRank does not "
                         "have 4 operands\n");
    return false;
  }

  // 4. Trace the second operand (nRanks, index 1) through unrealized casts /
  //    llvm or memref loads+stores (via enzymexla.pointer2memref) back to
  //    spmd.getSizeOfComm whose comm operand is defined by spmd.CommWorldOp.
  //    Dominance is checked on every load/store pair to ensure the store
  //    reliably reaches the load. If multiple dominating stores exist and
  //    neither dominates the other the origin is ambiguous and the trace fails.
  Value nRanks = callOp->getOperand(1);

  auto parentFunc = oldOpr->getParentOfType<LLVM::LLVMFuncOp>();
  if (!parentFunc) {
    LLVM_DEBUG(DBGS() << "ncclInitMatchesWorldSize: ncclCommInitRank is not "
                         "inside an LLVMFuncOp\n");
    return false;
  }

  DominanceInfo domInfo(parentFunc);

  // Helper: given a new dominating store, update `stored` or return false if
  // truly ambiguous (neither candidate dominates the other).
  auto updateStored = [&](Value &stored, LLVM::StoreOp storeOp) -> bool {
    if (!stored) {
      stored = storeOp.getValue();
      return true;
    }
    Operation *prevDef = stored.getDefiningOp();
    Operation *newDef = storeOp.getOperation();
    if (domInfo.dominates(newDef, prevDef))
      stored = storeOp.getValue(); // new store is later, prefer it
    else if (!domInfo.dominates(prevDef, newDef))
      return false; // neither dominates the other — truly ambiguous
    // else: existing stored is later, keep it
    return true;
  };

  auto updateStoredMemref = [&](Value &stored,
                                memref::StoreOp storeOp) -> bool {
    if (!stored) {
      stored = storeOp.getValue();
      return true;
    }
    Operation *prevDef = stored.getDefiningOp();
    Operation *newDef = storeOp.getOperation();
    if (domInfo.dominates(newDef, prevDef))
      stored = storeOp.getValue(); // new store is later, prefer it
    else if (!domInfo.dominates(prevDef, newDef))
      return false; // neither dominates the other — truly ambiguous
    // else: existing stored is later, keep it
    return true;
  };

  auto tracesToCommWorldSize = [&](Value val) -> bool {
    while (true) {
      Operation *defOp = val.getDefiningOp();
      if (!defOp)
        return false;

      // ── Terminal: spmd.getSizeOfComm ────────────────────────────────────
      if (auto getSizeOp = dyn_cast<spmd::GetSizeOfCommOp>(defOp)) {
        // isWorldComm peels the materialisation cast and accepts both forms
        // the comm can have here: spmd.commWorld if MPI_Comm_size was already
        // converted, or the raw MPI_COMM_WORLD constant if it was not.
        return isWorldComm(getSizeOp.getComm());
      }

      // ── Transparent: unrealized conversion cast (single input) ──────────
      if (auto cast = dyn_cast<UnrealizedConversionCastOp>(defOp)) {
        if (cast.getInputs().size() != 1)
          return false;
        val = cast.getInputs()[0];
        continue;
      }

      // ── Transparent: llvm.load – find closest dominating llvm.store ─────
      if (auto loadOp = dyn_cast<LLVM::LoadOp>(defOp)) {
        Value ptr = loadOp.getAddr();
        Value stored;
        bool ok = true;
        for (OpOperand &use : ptr.getUses()) {
          if (auto storeOp = dyn_cast<LLVM::StoreOp>(use.getOwner())) {
            if (domInfo.dominates(storeOp.getOperation(),
                                  loadOp.getOperation())) {
              if (!updateStored(stored, storeOp)) {
                ok = false;
                break;
              }
            }
          }
        }
        if (!ok || !stored)
          return false;
        val = stored;
        continue;
      }

      // ── Transparent: memref.load ─────────────────────────────────────────
      if (auto loadOp = dyn_cast<memref::LoadOp>(defOp)) {
        Value stored;
        bool ok = true;
        auto p2m = dyn_cast_or_null<enzymexla::Pointer2MemrefOp>(
            loadOp.getMemRef().getDefiningOp());

        if (p2m) {
          // Walk all pointer2memref users of the same raw pointer.
          Value ptr = p2m.getOperand();
          for (OpOperand &ptrUse : ptr.getUses()) {
            if (!ok)
              break;
            auto otherP2m =
                dyn_cast<enzymexla::Pointer2MemrefOp>(ptrUse.getOwner());
            if (!otherP2m)
              continue;
            for (OpOperand &memUse : otherP2m.getResult().getUses()) {
              if (auto storeOp = dyn_cast<memref::StoreOp>(memUse.getOwner())) {
                if (domInfo.dominates(storeOp.getOperation(),
                                      loadOp.getOperation())) {
                  if (!updateStoredMemref(stored, storeOp)) {
                    ok = false;
                    break;
                  }
                }
              }
            }
          }
        } else {
          // Direct stores to the same memref value.
          for (OpOperand &use : loadOp.getMemRef().getUses()) {
            if (auto storeOp = dyn_cast<memref::StoreOp>(use.getOwner())) {
              if (domInfo.dominates(storeOp.getOperation(),
                                    loadOp.getOperation())) {
                if (!updateStoredMemref(stored, storeOp)) {
                  ok = false;
                  break;
                }
              }
            }
          }
        }

        if (!ok || !stored)
          return false;
        val = stored;
        continue;
      }

      return false;
    }
  };

  if (!tracesToCommWorldSize(nRanks)) {
    LLVM_DEBUG(DBGS() << "ncclInitMatchesWorldSize: nRanks does not trace back "
                         "to getSizeOfComm(commWorld)\n");
    return false;
  }

  return true;
}

// if id is sent by a collective to comm world, then rankArg represents
// equivalently the rank in commWorld of both modelGroups.
// Todo: for simplicity we just check that sizeArg stems from a commWorld and
// assume that one process manages one gpu.
//
// Emits the synthetic spmd.init for an ncclCommInitRank matching that pattern;
// a call that does not match is left alone.
//
// Worth knowing: every precondition above used to be an llvm_unreachable. The
// shipping build is -c opt, which defines NDEBUG, so llvm_unreachable expanded
// to __builtin_unreachable() -- the optimiser deleted each check and the init
// below was emitted for *every* ncclCommInitRank. That masked two real bugs in
// the match, which is why it went unnoticed that it never once succeeded:
// the communicator behind spmd.getSizeOfComm was compared without peeling the
// materialisation cast, and only spmd.commWorld was accepted, never the raw
// MPI_COMM_WORLD constant still present when MPI_Comm_size has not been
// converted yet. Both are fixed in isWorldComm(); the pattern now matches all
// 32 ncclCommInitRank sites in the collective suite, so this is a real gate.
void handleNcclCommInitRank(Operation *newOpr, Operation *oldOpr) {
  if (!ncclInitMatchesWorldSize(newOpr, oldOpr))
    return;
  // 5. Everything checks out – emit spmd.init just before newOpr.  The call
  // both initialises NCCL and hands out the communicator, and the dialect
  // splits the two; the initialisation has to come first, or the
  // communicator read from spmd.commWorld is read before there is one.
  OpBuilder builder(newOpr);
  auto modelGroup = getUsedModel(newOpr);
  assert(modelGroup.has_value() && "no modelGroupAttr Value");
  Value singleLevel =
      getSingleThreadLevel(builder, newOpr->getLoc(), modelGroup.value());
  spmd::InitOp::create(
      builder, newOpr->getLoc(),
      /*resultTypes=*/TypeRange{builder.getType<spmd::ErrorType>()},
      /*usedModel=*/(uint32_t)modelGroup.value(),
      /*threadRequired=*/singleLevel);
}

const ApiCallToSpmdMap createNcclToSpmdOperands() {
  ApiCallToSpmdMap map = {
      // {"ncclCommInitRank", {"spmd.init", Operands(), Results(),
      // newAttributes(), postProcessing()}},
      // {"ncclCommInitRankConfig", {"spmd.init", Operands(), Results(),
      // newAttributes(), postProcessing()}},
      {"ncclCommDestroy",
       {"spmd.commDestroy", Operands(O("comm", 0)), Results(), newAttributes(),
        postProcessing()}},
      {"ncclCommSplit",
       {"spmd.commSplit",
        Operands(O("parentComm", 0), O("color", 1), O("key", 2)
                 // O("commConfig", 4)
                 ),
        Results(R("newComm", 3)), newAttributes(), postProcessing()}},
      {"ncclCommUserRank",
       {"spmd.getRankInComm", Operands(O("comm", 0)), Results(R("rank", 1)),
        newAttributes(), postProcessing()}},
      {"ncclCommCuDevice",
       {"spmd.getDeviceInComm", Operands(O("comm", 0)), Results(R("device", 1)),
        newAttributes(), postProcessing()}},
      {"ncclCommInitRank",
       {"spmd.commWorld", Operands(), Results(R("comm", 0)), newAttributes(),
        postProcessing(handleNcclCommInitRank),
        /*producesError=*/false}},
      {"ncclCommCount",
       {"spmd.getSizeOfComm", Operands(O("comm", 0)), Results(R("size", 1)),
        newAttributes(), postProcessing()}},
      {"ncclMemAlloc",
       {"spmd.malloc",
        Operands(D("comm", getWorldComm), O("size", 1, convertI32ToI64)),
        Results(R("address", 0)), newAttributes(), postProcessing()}},
      {"ncclMemFree",
       {"spmd.free", Operands(D("comm", getWorldComm), O("address", 0)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclSend",
       {"spmd.send",
        Operands(O("comm", 4), O("sendBuf", 0), O("dataCount", 1),
                 O("datatype", 2), O("destRank", 3), D("tag", zeroI32),
                 O("stream", 5)),
        Results(), newAttributes(getNotReadyModeAttr, getNotBufferedAttr),
        postProcessing()}},
      {"ncclRecv",
       {"spmd.recv",
        Operands(O("comm", 4), O("recvBuf", 0), O("dataCount", 1),
                 O("datatype", 2), O("srcRank", 3), D("tag", zeroI32),
                 O("stream", 5)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclAllReduce",
       {"spmd.allreduce",
        Operands(O("comm", 5), O("sendBuf", 0), O("recvBuf", 1),
                 O("dataCount", 2), O("datatype", 3), O("reduceOp", 4),
                 O("stream", 6)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclBroadcast",
       {"spmd.bcast",
        Operands(O("comm", 5), O("sendBuf", 0), O("recvBuf", 1),
                 O("dataCount", 2), O("datatype", 3), O("rootRank", 4),
                 O("stream", 6)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclBcast",
       {"spmd.bcast",
        Operands(O("comm", 4), O("sendBuf", 0), O("recvBuf", 0),
                 O("dataCount", 1), O("datatype", 2), O("rootRank", 3),
                 O("stream", 5)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclReduce",
       {"spmd.reduce",
        Operands(O("comm", 6), O("sendBuf", 0), O("recvBuf", 1),
                 O("dataCount", 2), O("datatype", 3), O("reduceOp", 4),
                 O("rootRank", 5), O("stream", 7)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclAllGather",
       {"spmd.allgather",
        Operands(O("comm", 4), O("sendBuf", 0), O("sendCount", 2),
                 O("sendType", 3), O("recvBuf", 1), O("recvCount", 2),
                 O("recvType", 3), O("stream", 5)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclReduceScatter",
       {"spmd.reduceScatter",
        Operands(O("comm", 5), O("sendBuf", 0), O("recvBuf", 1),
                 O("dataCount", 2), O("datatype", 3), O("reduceOp", 4),
                 O("stream", 6)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclAlltoAll",
       {"spmd.alltoall",
        Operands(O("comm", 4), O("sendBuf", 0), O("sendCount", 2),
                 O("sendType", 3), O("recvBuf", 1), O("recvCount", 2),
                 O("recvType", 3), O("stream", 5)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclGather",
       {"spmd.gather",
        Operands(O("comm", 5), O("sendBuf", 0), O("sendCount", 2),
                 O("sendType", 3), O("recvBuf", 1), O("recvCount", 2),
                 O("recvType", 3), O("rootRank", 4), O("stream", 6)),
        Results(), newAttributes(), postProcessing()}},
      {"ncclScatter",
       {"spmd.scatter",
        Operands(O("comm", 5), O("sendBuf", 0), O("sendCount", 2),
                 O("sendType", 3), O("recvBuf", 1), O("recvCount", 2),
                 O("recvType", 3), O("rootRank", 4), O("stream", 6)),
        Results(), newAttributes(), postProcessing()}},
      // NCCL RMA calls
      /*{"ncclPutSignal", {"spmd.putImplicitSignalOp",
        Operands(
          O("comm", 9),
          O("originAddr", 0),
          O("originCount", 1),
          O("originDatatype", 2),
          D("originStride", oneI64),
          O("targetRank", 3),
          O("targetWin", 4),
          O("targetDisp", 5),
          O("targetCount", 1),
          O("targetDatatype", 2),
          D("targetStride", oneI64),
          O("sigIdx", 6),
          O("ctx", 7),
          O("flags", 8),
          O("stream", 10)
        ),
        Results(),
        newAttributes(),
        postProcessing()
      }},
      {"ncclSignal", {"spmd.signal",
        Operands(
          O("comm", 4),
          O("targetRank", 0),
          O("sigIdx", 1),
          O("ctx", 2),
          O("flags", 3),
          O("stream", 5)
        ),
        Results(),
        newAttributes(),
        postProcessing()
      }},
      {"ncclWaitSignal", {"spmd.waitSignal",
        Operands(
          O("comm", 2),
          O("nDesc", 0),
          O("signalDescs", 1),
          O("stream", 3)
        ),
        Results(),
        newAttributes(),
        postProcessing()
      }}*/
      {"ncclGroupStart",
       {"spmd.groupStart", Operands(), Results(), newAttributes(),
        postProcessing()}},
      {"ncclGroupEnd",
       {"spmd.groupEnd", Operands(), Results(), newAttributes(),
        postProcessing()}}};
  return map;
}

const ApiCallToSpmdMap ncclToSpmdOperands = createNcclToSpmdOperands();

} // namespace mlir::spmd
