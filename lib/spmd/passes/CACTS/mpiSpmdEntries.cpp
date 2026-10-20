// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mpiSpmdEntries.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir::spmd {

void checkCommSplitEqualsWorldDup(Operation *spmdCommSplit,
                                  Operation *oldOperation) {
  if (!isa<spmd::CommSplitOp>(spmdCommSplit)) {
    return;
  }
  OpBuilder builder(spmdCommSplit);
  auto loc = spmdCommSplit->getLoc();
  Value color = spmdCommSplit->getOperand(1);
  Value key = spmdCommSplit->getOperand(2);
  Value newComm = spmdCommSplit->getResult(0);

  std::optional<int> modelGroupOpt = getUsedModel(spmdCommSplit);
  if (!modelGroupOpt) {
    LLVM_DEBUG(
        llvm::dbgs()
        << "Warning: spmd.comm_split does not have a usedModel attribute. "
           "Skipping the check for comm_split being a dup of world_comm.\n");
    return;
  }
  int modelGroup = *modelGroupOpt;

  // special case of a comm_split as a dup of world_comm.
  // Every step is null-guarded: color/key can be block arguments (e.g. an
  // MPI_Comm_split in a helper function that takes them as parameters), in
  // which case getDefiningOp() returns null.
  auto rankOp = key.getDefiningOp<spmd::GetRankInCommOp>();
  if (color.getDefiningOp<arith::ConstantOp>() && rankOp &&
      rankOp.getComm().getDefiningOp<spmd::CommWorldOp>()) {
    Value commWorld =
        spmd::CommWorldOp::create(builder, loc, modelGroup).getResult();
    newComm.replaceAllUsesWith(commWorld);
  }
}

const ApiCallToSpmdMap createMpiToSpmdOperands() {
  ApiCallToSpmdMap map = {

  // MPI_Init: $threadRequired is non-optional on spmd.init, so we inject
  // spmd.constThreadLevel("SINGLE") as a default operand.
  // MPI standard §12.4.3: MPI_Init is equivalent to MPI_Init_Thread with
  // required = MPI_THREAD_SINGLE.
  {"MPI_Init", {"spmd.init",
      Operands(/*O("argc", 0), O("argv", 1)*/D("threadRequired", getSingleThreadLevel)),
      Results(),   // $threadProvided absent — optional, not produced for MPI_Init
      newAttributes(),
      postProcessing()}},

  // MPI_Init_thread(argc[0], argv[1], required[2], provided[3])
  // argc and argv are intentionally skipped, matching MPI_Init.
  // required (2): kind "THREAD_LEVEL" in JSON → handleMPIOperand → getThreadLevelMPI
  // provided (3): kind "THREAD_LEVEL" in JSON → result type ThreadLevelType;
  //               castThreadLevelToI32ForMPI stores i32 back to int*.
  {"MPI_Init_thread", {"spmd.init",
      Operands(O("threadRequired", 2)),
      Results(R("threadProvided", 3, castThreadLevelToI32)),
      newAttributes(),
      postProcessing()}},

  {"MPI_Finalize", {"spmd.finalize",
                    Operands(),
                    Results(),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Abort", {"spmd.abort",
                  Operands(O("comm", 0), O("errorCode", 1)),
                  Results(),
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Comm_rank", {"spmd.getRankInComm",
                      Operands(O("comm", 0)),
                      Results(R("rank", 1)),
                      newAttributes(),
                      postProcessing()}},

  {"MPI_Comm_size", {"spmd.getSizeOfComm",
                      Operands(O("comm", 0)),
                      Results(R("size", 1)),
                      newAttributes(),
                      postProcessing()}},

  {"MPI_Barrier", {"spmd.barrier",
                    Operands(O("comm", 0)),
                    Results(),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Bcast", {"spmd.bcast",
                  Operands(O("comm", 4),
                          O("sendBuf", 0),
                          O("recvBuf", 0),
                          O("count", 1, convertI32ToI64),
                          O("datatype", 2),
                          O("root", 3)
                          ),
                  Results(),
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Gather", {"spmd.gather",
                  Operands(O("comm", 7),
                          O("sendBuf", 0),
                          O("sendCount", 1, convertI32ToI64),
                          O("sendType", 2),
                          O("recvBuf", 3),
                          O("recvCount", 4, convertI32ToI64),
                          O("recvType", 5),
                          O("root", 6)),
                  Results(),
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Allgather", {"spmd.allgather",
                      Operands(O("comm", 6),
                              O("sendBuf", 0),
                              O("sendCount", 1, convertI32ToI64),
                              O("sendType", 2),
                              O("recvBuf", 3),
                              O("recvCount", 4, convertI32ToI64),
                              O("recvType", 5)),
                      Results(),
                      newAttributes(),
                      postProcessing()}},

  {"MPI_Alltoall", {"spmd.alltoall",
                    Operands(O("comm", 6),
                              O("sendBuf", 0),
                              O("sendCount", 1, convertI32ToI64),
                              O("sendType", 2),
                              O("recvBuf", 3),
                              O("recvCount", 4, convertI32ToI64),
                              O("recvType", 5)),
                    Results(),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Reduce", {"spmd.reduce",
                  Operands(O("comm", 6),
                            O("sendBuf", 0),
                            O("recvBuf", 1),
                            O("count", 2, convertI32ToI64),
                            O("datatype", 3),
                            O("op", 4),
                            O("root", 5)),
                  Results(),
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Allreduce", {"spmd.allreduce",
                      Operands(O("comm", 5),
                              O("sendBuf", 0),
                              O("recvBuf", 1),
                              O("count", 2, convertI32ToI64),
                              O("datatype", 3),
                              O("op", 4)),
                      Results(),
                      newAttributes(),
                      postProcessing()}},

  {"MPI_Scan", {"spmd.scan",
                Operands(O("comm", 5),
                          O("sendBuf", 0),
                          O("recvBuf", 1),
                          O("count", 2, convertI32ToI64),
                          O("datatype", 3),
                          O("op", 4)),
                Results(),
                newAttributes(),
                postProcessing()}},

  {"MPI_Exscan", {"spmd.exscan",
                  Operands(O("comm", 5),
                            O("sendBuf", 0),
                            O("recvBuf", 1),
                            O("count", 2, convertI32ToI64),
                            O("datatype", 3),
                            O("op", 4)),
                  Results(),
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Scatter", {"spmd.scatter",
                    Operands(O("comm", 7),
                            O("sendBuf", 0),
                            O("sendCount", 1, convertI32ToI64),
                            O("sendType", 2),
                            O("recvBuf", 3),
                            O("recvCount", 4, convertI32ToI64),
                            O("recvType", 5),
                            O("root", 6)),
                    Results(),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Reduce_scatter_block", {"spmd.reduceScatter",
                                Operands(O("comm", 5),
                                          O("sendBuf", 0),
                                          O("recvBuf", 1),
                                          O("count", 2, convertI32ToI64),
                                          O("datatype", 3),
                                          O("op", 4)
                                          ),
                                Results(),
                                newAttributes(),
                                postProcessing()}},


  {"MPI_Send", {"spmd.send",
                Operands(O("comm", 5),
                          O("buf", 0),
                          O("count", 1, convertI32ToI64),
                          O("datatype", 2),
                          O("dest", 3),
                          O("tag", 4)),
                Results(),
                newAttributes(getNotReadyModeAttr, getNotBufferedAttr),
                postProcessing()}},

  {"MPI_Ssend", {"spmd.send",
                  Operands(O("comm", 5),
                          O("buf", 0),
                          O("count", 1, convertI32ToI64),
                          O("datatype", 2),
                          O("dest", 3),
                          O("tag", 4)),
                  Results(),
                  newAttributes(getNotReadyModeAttr, getNotBufferedAttr),
                  postProcessing()}},

  {"MPI_Bsend", {"spmd.send",
                  Operands(O("comm", 5),
                          O("buf", 0),
                          O("count", 1, convertI32ToI64),
                          O("datatype", 2),
                          O("dest", 3),
                          O("tag", 4)),
                  Results(),
                  newAttributes(getNotReadyModeAttr, getBufferedAttr),
                  postProcessing()}},

  {"MPI_Rsend", {"spmd.send",
                  Operands(O("comm", 5),
                          O("buf", 0),
                          O("count", 1, convertI32ToI64),
                          O("datatype", 2),
                          O("dest", 3),
                          O("tag", 4)),
                  Results(),
                  newAttributes(getReadyModeAttr, getNotBufferedAttr),
                  postProcessing()}},

  {"MPI_Recv", {"spmd.recv",
                Operands(O("comm", 5),
                          O("buf", 0),
                          O("count", 1, convertI32ToI64),
                          O("datatype", 2),
                          O("source", 3),
                          O("tag", 4)),
                Results(/*R("status", 6)*/),
                newAttributes(),
                postProcessing()}},

  {"MPI_Win_fence", {"spmd.fence",
                  Operands(O("win", 1), O("assert", 0)),
                  Results(),
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Win_allocate", {"spmd.winAlloc",
                        Operands(
                          O("comm", 3),
                          O("size", 0),
                          O("dispUnit", 1)
                          //    O("info", std::nullopt),
                        ),
                        Results(R("basePtr", 4), R("win", 5)),
                        newAttributes(),
                        postProcessing()}},

  {"MPI_Win_free", {"spmd.winFree",
                    Operands(O("win", 0)),
                    Results(),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Win_create", {"spmd.winCreate",
                      Operands(O("comm", 4),
                                O("baseAddr", 0),
                                O("size", 1),
                                O("dispUnit", 2)
                              //  O("info", std::nullopt)
                            ),
                      Results(R("win", 5)),
                      newAttributes(),
                      postProcessing()}},

  {"MPI_Wait", {"spmd.wait",
                Operands(O("req", 0)),
                Results(R("status", 1)),
                newAttributes(),
                postProcessing()}},

  {"MPI_Waitall", {"spmd.waitAll",
                  Operands(O("count", 0),
                            O("arrayOfReqs", 1),
                            O("statuses", 2)),  // arg 2 = MPI_Status[] (optional)
                  Results(),
                  newAttributes(),
                  postProcessing()}},

  // api.json gives outcount and array_of_indices no kind, so their spmd types
  // are spelled out here; the same holds for MPI_Testsome.
  {"MPI_Waitsome", {"spmd.waitSome",
                  Operands(O("incount", 0),
                            O("arrayOfReqs", 1),
                            O("arrayOfIndices", 3, viewAsI32Array), // int *array_of_indices
                            O("statuses", 4)),         // arg 4 = MPI_Status[] (optional)
                  Results(R("outcount", 2, nullptr, i32Type)), // int *outcount
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Waitany", {"spmd.waitAny",
                    Operands(O("count", 0),
                              O("arrayOfReqs", 1)),
                    Results(R("index", 2),        // arg 2 = int *indx
                            R("status", 3)),      // arg 3 = MPI_Status*
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Comm_split", {"spmd.commSplit",
                      Operands(O("comm", 0),
                                O("color", 1),
                                O("key", 2)),
                      Results(R("newcomm", 3)),
                      newAttributes(),
                      postProcessing(checkCommSplitEqualsWorldDup)}},

  {"MPI_Comm_dup", {"spmd.commDup",
                    Operands(O("comm", 0)),
                    Results(R("newcomm", 1)),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Comm_free", {"spmd.commDestroy",
                      Operands(O("comm", 0)),
                      Results(),
                      newAttributes(),
                      postProcessing()}},

  {"MPI_Alloc_mem", {"spmd.malloc",
                      Operands(D("comm", getWorldComm),
                              O("size", 0) 
                              // O("info", std::nullopt), 
                              ),
                      Results(R("baseptr", 2)),
                      newAttributes(),
                      postProcessing()}},

  {"MPI_Free_mem", {"spmd.free",
                    Operands(D("comm", getWorldComm),
                              O("baseptr", 0)),
                    Results(),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Test", {"spmd.test",
              Operands(O("req", 0)),
              Results(R("flag", 1, convertI1ToI32),
                      R("status", 2)),          // arg 2 = MPI_Status*
              newAttributes(),
              postProcessing()}},

  {"MPI_Testall", {"spmd.testAll",
                    Operands(O("count", 0),
                              O("arrayOfReqs", 1),
                              O("arrayOfStatuses", 3)),
                    Results(R("flag", 2, convertI1ToI32)),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Testsome", {"spmd.testSome",
                    Operands(O("incount", 0),
                              O("arrayOfReqs", 1),
                              O("arrayOfIndices", 3, viewAsI32Array),
                              O("arrayOfStatuses", 4)),
                    Results(R("outcount", 2, nullptr, i32Type)),
                    newAttributes(),
                    postProcessing()}},

  {"MPI_Testany", {"spmd.testAny",
                  Operands(O("count", 0),
                            O("arrayOfReqs", 1)),
                  Results(R("index", 2),
                          R("flag", 3, convertI1ToI32),
                          R("status", 4)),
                  newAttributes(),
                  postProcessing()}},

  {"MPI_Request_free", {"spmd.reqFree",
                        Operands(O("req", 0)),
                        Results(),
                        newAttributes(),
                        postProcessing()}},

  // $isAtomic is OpenSHMEM's atomic single-element store and fetch
  // (shmem_<T>_atomic_set/_fetch), which a transfer of a buffer is not.  The
  // accumulating calls below are atomic per element by the MPI standard.
  {"MPI_Put", {"spmd.put",
                Operands(O("originAddr", 0),
                        O("originCount", 1, convertI32ToI64),
                        O("originDatatype", 2),
                        D("originStride", oneI64),
                        O("targetRank", 3),
                        O("targetWin", 7),
                        O("targetDisp", 4),
                        O("targetCount", 5, convertI32ToI64),
                        O("targetDatatype", 6),
                        D("targetStride", oneI64)),
                Results(),
                newAttributes(getNotAtomicAttr),
                postProcessing()}},



  {"MPI_Get", {"spmd.get",
                Operands(O("originAddr", 0),
                        O("originCount", 1, convertI32ToI64),
                        O("originDatatype", 2),
                        D("originStride", oneI64),
                        O("targetRank", 3),
                        O("targetWin", 7),
                        O("targetDisp", 4),
                        O("targetCount", 5, convertI32ToI64),
                        O("targetDatatype", 6),
                        D("targetStride", oneI64)),
                Results(),
                newAttributes(getNotAtomicAttr),
                postProcessing()}},



  {"MPI_Accumulate", {"spmd.accumulate",
                      Operands(O("originAddr", 0),
                                O("originCount", 1, convertI32ToI64),
                                O("originDatatype", 2),
                                D("originStride", oneI64),
                                O("targetRank", 3),
                                O("targetWin", 8),
                                O("targetDisp", 4),
                                O("targetCount", 5, convertI32ToI64),
                                O("targetDatatype", 6),
                                D("targetStride", oneI64),
                                O("op", 7)),
                      Results(),
                      newAttributes(getAtomicAttr),
                      postProcessing()}},

  {"MPI_Get_accumulate", {"spmd.getAccumulate",
                          Operands(O("originAddr", 0),
                                    O("originCount", 1, convertI32ToI64),
                                    O("originDatatype", 2),
                                    D("originStride", oneI64),
                                    O("resultAddr", 3),
                                    O("resultCount", 4, convertI32ToI64),
                                    O("resultDatatype", 5),
                                    D("resultStride", oneI64),
                                    O("targetRank", 6),
                                    O("targetWin", 11),
                                    O("targetDisp", 7),
                                    O("targetCount", 8, convertI32ToI64),
                                    O("targetDatatype", 9),
                                    D("targetStride", oneI64),
                                    O("op", 10)),
                          Results(),
                          newAttributes(getAtomicAttr),
                          postProcessing()}},

  {"MPI_Fetch_and_op", {"spmd.fetchAndOp",
                        Operands(O("originAddr", 0),
                                  O("resultAddr", 1),
                                  O("type", 2),
                                  O("reduceOp", 5),
                                  O("targetRank", 3),
                                  O("targetWin", 6),
                                  O("targetDisposition", 4)),
                          Results(),
                          newAttributes(getAtomicAttr),
                          postProcessing()}},

  {"MPI_Compare_and_swap", {"spmd.compareAndSwap",
                        Operands(O("originAddr", 0),
                                  O("compareAddr", 1),
                                  O("resultAddr", 2),
                                  O("type", 3),
                                  O("targetRank", 4),
                                  O("targetWin", 6),
                                  O("targetDisposition", 5)),
                          Results(),
                          newAttributes(getAtomicAttr),
                          postProcessing()}},

// TODO: insert all the O(info) comments for completeness

  {"MPI_Dims_create", {
    "spmd.dimsCreate",
    Operands(O("nProc", 0),
             O("nDims", 1),
             O("dims", 2)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Recv_init", {
    "spmd.recvInit",
    Operands(O("comm", 5),
             O("recvBuf", 0),
             O("dataCount", 1, convertI32ToI64),
             O("datatype", 2),
             O("srcRank", 3),
             O("tag", 4)),
    Results(R("req", 6)),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Precv_init", {
    "spmd.recvInit",
    Operands(O("comm", 6),
             O("recvBuf", 0),
             O("partitions", 1),
             O("dataCount", 2, convertI32ToI64),
             O("datatype", 3),
             O("srcRank", 4),
             O("tag", 5)),
    Results(R("req", 8)),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Cart_create", {
    "spmd.cartCreate",
    Operands(O("oldComm", 0),
             O("nDims", 1),
             O("dims", 2),
             O("periods", 3),
             O("reorder", 4)),
    Results(R("cartComm", 5)),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Send_init", {
    "spmd.sendInit",
    Operands(O("comm", 5),
             O("sendBuf", 0),
             O("dataCount", 1, convertI32ToI64),
             O("datatype", 2),
             O("destRank", 3),
             O("tag", 4)),
    Results(R("req", 6)),
    newAttributes(getNotBufferedAttr, getNotReadyModeAttr),
    postProcessing()
  }},
  {"MPI_Ssend_init", {
    "spmd.sendInit",
    Operands(O("comm", 5),
             O("sendBuf", 0),
             O("dataCount", 1, convertI32ToI64),
             O("datatype", 2),
             O("destRank", 3),
             O("tag", 4)),
    Results(R("req", 6)),
    newAttributes(getNotBufferedAttr, getNotReadyModeAttr),
    postProcessing()
  }},
  {"MPI_Bsend_init", {
    "spmd.sendInit",
    Operands(O("comm", 5),
             O("sendBuf", 0),
             O("dataCount", 1, convertI32ToI64),
             O("datatype", 2),
             O("destRank", 3),
             O("tag", 4)),
    Results(R("req", 6)),
    newAttributes(getBufferedAttr, getNotReadyModeAttr),
    postProcessing()
  }},
  {"MPI_Rsend_init", {
    "spmd.sendInit",
    Operands(O("comm", 5),
             O("sendBuf", 0),
             O("dataCount", 1, convertI32ToI64),
             O("datatype", 2),
             O("destRank", 3),
             O("tag", 4)),
    Results(R("req", 6)),
    newAttributes(getNotBufferedAttr, getReadyModeAttr),
    postProcessing()
  }},
  {"MPI_Win_complete", {
    "spmd.winComplete",
    Operands(O("win", 0)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Mrecv", {
    "spmd.mRecv",
    Operands(O("recvBuf", 0),
             O("dataCount", 1, convertI32ToI64),
             O("datatype", 2),
             O("msg", 3)),
    Results(R("status", 4)),
    newAttributes(),
    postProcessing()
  }},
  {"MPI_Imrecv", {
    "spmd.imRecv",
    Operands(O("recvBuf", 0),
             O("dataCount", 1, convertI32ToI64),
             O("datatype", 2),
             O("msg", 3)),
    Results(R("req", 4)),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Win_flush", {
    "spmd.flush",
    Operands(O("rank", 0),
             O("win", 1)),
    Results(),
    newAttributes(getNotLocalAttr),
    postProcessing()
  }},

  {"MPI_Start", {
    "spmd.start",
    Operands(O("req", 0)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Sendrecv", {
    "spmd.sendrecv",
    Operands(O("comm", 10),
             O("sendBuf", 0),
             O("sendCount", 1, convertI32ToI64),
             O("sendDatatype", 2),
             D("sendStride", oneI64),
             O("sendRank", 3),
             O("sendTag", 4),
             O("recvBuf", 5),
             O("recvCount", 6, convertI32ToI64),
             O("recvDatatype", 7),
             D("recvStride", oneI64),
             O("recvRank", 8),
             O("recvTag", 9)),
    Results(/*R("status", 11)*/),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Sendrecv_replace", {
    "spmd.sendrecv",
    Operands(O("comm", 7),
             O("sendBuf", 0),
             O("sendCount", 1, convertI32ToI64),
             O("sendDatatype", 2),
             D("sendStride", oneI64),
             O("sendRank", 3),
             O("sendTag", 4),
             O("recvBuf", 0),
             O("recvCount", 1, convertI32ToI64),
             O("recvDatatype", 2),
             D("recvStride", oneI64),
             O("recvRank", 5),
             O("recvTag", 6)),
    Results(/*R("status", 8)*/),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Psend_init", {
    "spmd.sendInit",
    Operands(O("comm", 6),
             O("sendBuf", 0),
             O("partitions", 1),
             O("dataCount", 2),
             O("datatype", 3),
             O("destRank", 4),
             O("tag", 5)),
    Results(R("req", 8)),
    newAttributes(getNotBufferedAttr, getNotReadyModeAttr),
    postProcessing()
  }},

  {"MPI_Win_flush_all", {
    "spmd.flushAll",
    Operands(O("win", 0)),
    Results(),
    newAttributes(getNotLocalAttr),
    postProcessing()
  }},

  {"MPI_Win_flush_local", {
    "spmd.flush",
    Operands(O("rank", 0),
             O("win", 1)),
    Results(),
    newAttributes(getLocalAttr),
    postProcessing()
  }},

  {"MPI_Win_flush_local_all", {
    "spmd.flushAll",
    Operands(O("win", 0)),
    Results(),
    newAttributes(getLocalAttr),
    postProcessing()
  }},

  {"MPI_Win_lock", {
    "spmd.lock",
    Operands(O("rank", 1),
             O("win", 3),
             O("lockType", 0),
             O("assert", 2)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Win_lock_all", {
    "spmd.lockAll",
    Operands(O("win", 1),
             O("assert", 0)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Win_start", {
    "spmd.winStart",
    Operands(O("group", 0),
             O("assert", 1),
             O("win", 2)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Win_unlock", {
    "spmd.unlock",
    Operands(O("rank", 0),
             O("win", 1)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Win_unlock_all", {
    "spmd.unlockAll",
    Operands(O("win", 0)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Pready", {
    "spmd.pReady",
    Operands(O("partition", 0),
             O("request", 1)),
    Results(),
    newAttributes(),
    postProcessing()
  }},

  {"MPI_Mprobe", {
    "spmd.mProbe",
    Operands(O("comm", 2),
             O("srcRank", 0),
             O("tag", 1)),
    Results(R("msg", 3)/*,
            R("status", 4)*/),
    newAttributes(),
    postProcessing()
  }}
};

  // StringMap::lookup value-initialises a missing entry, so a misspelled name
  // below used to derive an entry with an empty op name, which crashed the pass
  // only when a program called the non-blocking form (MPI_Sendrecv_Replace).
  // This table is built at startup, so fail there instead.
  auto blocking = [&map](llvm::StringRef name) -> const APICallEntry & {
    auto it = map.find(name);
    if (it == map.end())
      llvm::reportFatalInternalError(llvm::Twine("mpiSpmdEntries: no entry '") +
                                     name +
                                     "' to derive a non-blocking variant from");
    return it->second;
  };

  // non-blocking variants
  map["MPI_Rput"] = makeNonBlocking(blocking("MPI_Put"), /*reqArgidx=*/8);
  map["MPI_Rget"] = makeNonBlocking(blocking("MPI_Get"), /*reqArgidx=*/8);
  map["MPI_Rget_accumulate"] =
      makeNonBlocking(blocking("MPI_Get_accumulate"), /*reqArgidx=*/12);
  map["MPI_Raccumulate"] =
      makeNonBlocking(blocking("MPI_Accumulate"), /*reqArgidx=*/9);
  map["MPI_Ibarrier"] =
      makeNonBlocking(blocking("MPI_Barrier"), /*reqArgidx=*/1);
  map["MPI_Ibcast"] = makeNonBlocking(blocking("MPI_Bcast"), /*reqArgidx=*/5);

  map["MPI_Igather"] = makeNonBlocking(blocking("MPI_Gather"), /*reqArgidx=*/8);
  map["MPI_Iallgather"] =
      makeNonBlocking(blocking("MPI_Allgather"), /*reqArgidx=*/7);
  map["MPI_Ialltoall"] =
      makeNonBlocking(blocking("MPI_Alltoall"), /*reqArgidx=*/7);

  map["MPI_Ireduce"] = makeNonBlocking(blocking("MPI_Reduce"), /*reqArgidx=*/7);
  map["MPI_Iallreduce"] =
      makeNonBlocking(blocking("MPI_Allreduce"), /*reqArgidx=*/6);
  map["MPI_Iscan"] = makeNonBlocking(blocking("MPI_Scan"), /*reqArgidx=*/6);
  map["MPI_Iexscan"] = makeNonBlocking(blocking("MPI_Exscan"), /*reqArgidx=*/6);

  map["MPI_Iscatter"] =
      makeNonBlocking(blocking("MPI_Scatter"), /*reqArgidx=*/8);
  map["MPI_Ireduce_scatter_block"] =
      makeNonBlocking(blocking("MPI_Reduce_scatter_block"),
                      /*reqArgidx=*/6);

  map["MPI_Isend"] = makeNonBlocking(blocking("MPI_Send"), /*reqArgidx=*/6);
  map["MPI_Issend"] = makeNonBlocking(blocking("MPI_Ssend"), /*reqArgidx=*/6);
  map["MPI_Ibsend"] = makeNonBlocking(blocking("MPI_Bsend"), /*reqArgidx=*/6);
  map["MPI_Irsend"] = makeNonBlocking(blocking("MPI_Rsend"), /*reqArgidx=*/6);
  map["MPI_Irecv"] = makeNonBlocking(blocking("MPI_Recv"), /*reqArgidx=*/6);
  map["MPI_Isendrecv"] =
      makeNonBlocking(blocking("MPI_Sendrecv"), /*reqArgidx=*/11);
  map["MPI_Isendrecv_replace"] =
      makeNonBlocking(blocking("MPI_Sendrecv_replace"), /*reqArgidx=*/8);
  return map;
}
// errortype is always the last result, and not included here but added later in
// the transformation pass

const ApiCallToSpmdMap mpiToSpmdOperands = createMpiToSpmdOperands();

} // namespace mlir::spmd