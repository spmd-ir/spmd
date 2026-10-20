// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shmemSpmdEntries.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "spmd/SPMDOps.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include <string>

namespace mlir::spmd {

static Value createDefaultWindow(OpBuilder &builder, Location loc,
                                 Value targetAddr, int modelGroup) {
  Value newWorldComm = getWorldComm(builder, loc, modelGroup);
  Value zeroI32 =
      arith::ConstantOp::create(builder, loc, builder.getI32IntegerAttr(0))
          .getResult();
  auto winCreateOp = spmd::WinCreateOp::create(
      builder, loc, builder.getType<spmd::WinType>(),
      builder.getType<spmd::ErrorType>(), newWorldComm, targetAddr, Value(),
      zeroI32, modelGroup);
  Value newWin = winCreateOp.getResult(0);
  return newWin;
}

// The type a spmd.constDatatype records for the type name in a call's name.
// An integer is signed or unsigned when the name says which, and signless when
// it does not: C leaves the signedness of char to the platform, and the sized
// and untyped calls (shmem_put32, shmem_putmem) move words of no type at all.
inline Type getShmemElementType(OpBuilder &builder, llvm::StringRef datatype) {
  auto sint = [&](unsigned width) {
    return builder.getIntegerType(width, /*isSigned=*/true);
  };
  auto uint = [&](unsigned width) {
    return builder.getIntegerType(width, /*isSigned=*/false);
  };
  if (datatype == "char" || datatype == "byte" || datatype == "8" ||
      datatype == "mem")
    return builder.getI8Type();
  if (datatype == "schar" || datatype == "int8")
    return sint(8);
  if (datatype == "uchar" || datatype == "uint8")
    return uint(8);
  if (datatype == "16")
    return builder.getI16Type();
  if (datatype == "short" || datatype == "int16")
    return sint(16);
  if (datatype == "ushort" || datatype == "uint16")
    return uint(16);
  if (datatype == "32")
    return builder.getI32Type();
  if (datatype == "int" || datatype == "int32")
    return sint(32);
  if (datatype == "uint" || datatype == "uint32")
    return uint(32);
  if (datatype == "64")
    return builder.getI64Type();
  if (datatype == "long" || datatype == "longlong" || datatype == "int64" ||
      datatype == "ptrdiff" || datatype == "prtdiff")
    return sint(64);
  if (datatype == "ulong" || datatype == "ulonglong" || datatype == "uint64" ||
      datatype == "size")
    return uint(64);
  if (datatype == "128")
    return builder.getIntegerType(128);
  if (datatype == "float")
    return builder.getF32Type();
  if (datatype == "double")
    return builder.getF64Type();
  if (datatype == "longdouble")
    return builder.getF80Type();

  llvm::reportFatalUsageError("Not implemented SHMEM datatype");
}

// Take a raw api call name and insert the type in its name
static std::string typeApiCallName(llvm::StringRef stringType,
                                   std::string raw_name) {
  std::pair<StringRef, StringRef> parts =
      llvm::StringRef(raw_name).split("TYPENAME");
  std::string typedApiCallName =
      parts.first.str() + stringType.str() + parts.second.str();
  return typedApiCallName;
}

static std::string sizeApiCallName(llvm::StringRef stringType,
                                   std::string raw_name) {
  std::pair<StringRef, StringRef> parts =
      llvm::StringRef(raw_name).split("SIZE");
  std::string typedApiCallName =
      parts.first.str() + stringType.str() + parts.second.str();
  return typedApiCallName;
}

static std::string opApiCallName(llvm::StringRef stringOp,
                                 std::string raw_name) {
  std::pair<StringRef, StringRef> parts = llvm::StringRef(raw_name).split("OP");
  std::string typedApiCallName =
      parts.first.str() + stringOp.str() + parts.second.str();
  return typedApiCallName;
}

const ApiCallToSpmdMap createShmemToSpmdOperands() {
  // Library of allowed type lists in SHMEM
  // OpenSHMEM 1.6 Spec Table 5
  llvm::SmallVector<llvm::StringRef, 8> standardRmaTypes = {
      "float", "double",    "longdouble", "char",   "schar",  "short",
      "int",   "long",      "longlong",   "uchar",  "ushort", "uint",
      "ulong", "ulonglong", "int8",       "int16",  "int32",  "int64",
      "uint8", "uint16",    "uint32",     "uint64", "size",   "ptrdiff"};
  // TODO: rework fetch ops and types. Bitwise operations are not yet correct
  // OpenSHMEM 1.6 Spec Table 6
  llvm::SmallVector<llvm::StringRef, 8> standardAmoTypes = {
      "int",   "long",  "longlong", "uint",   "ulong", "ulonglong",
      "int32", "int64", "uint32",   "uint64", "size",  "ptrdiff"};
  // OpenSHMEM 1.6 Spec Table 7
  llvm::SmallVector<llvm::StringRef, 8> extendedAmoTypes = {
      "float",     "double", "int",   "long",   "longlong", "uint", "ulong",
      "ulonglong", "int32",  "int64", "uint32", "uint64",   "size", "ptrdiff"};
  llvm::SmallVector<llvm::StringRef, 8> sizeOptions = {"8", "16", "32", "64",
                                                       "128"};
  llvm::SmallVector<llvm::StringRef, 8> fetchOpOptions = {"inc", "add", "and",
                                                          "or", "xor"};

  // Library of all reduce operations per type as defined in OpenSHMEM 1.6 Spec
  // Table 10. Datastructure: Each pair in this vector contains a vector of
  // types and one of operations.
  //                Each of the types works with each of the operations in each
  //                pair. The cartesian product of the vectors in each pair
  //                performed for all pairs creates all combinations of types
  //                and operations that work for team based reductions.
  llvm::SmallVector<std::pair<llvm::SmallVector<llvm::StringRef, 32>,
                              llvm::SmallVector<llvm::StringRef, 8>>,
                    3>
      reductionTypesAndOps = {
          {{"char",     "schar",   "short",  "int",   "long",   "longlong",
            "ptrdiff",  "uchar",   "ushort", "uint",  "ulong",  "ulonglong",
            "int8",     "int16",   "int32",  "int64", "uint8",  "uint16",
            "uint32",   "uint64",  "size",   "float", "double", "longdouble",
            "complexd", "complexf"},
           {"sum", "prod"}},
          {{"char",    "schar",  "short",  "int",   "long",   "longlong",
            "ptrdiff", "uchar",  "ushort", "uint",  "ulong",  "ulonglong",
            "int8",    "int16",  "int32",  "int64", "uint8",  "uint16",
            "uint32",  "uint64", "size",   "float", "double", "longdouble"},
           {"sum", "prod", "max", "min"}},
          {{"uchar", "ushort", "uint", "ulong", "ulonglong", "int8", "int16",
            "int32", "int64", "uint8", "uint16", "uint32", "uint64", "size"},
           {"sum", "prod", "max", "min", "and", "or", "xor"}}};

  ApiCallToSpmdMap map = {

      // shmem_init: implied SINGLE, same as MPI_Init
      {"shmem_init",
       {"spmd.init",
        Operands(/*O("argc", 0), O("argv", 1)*/ D("threadRequired",
                                                  getSingleThreadLevel)),
        Results(), newAttributes(), postProcessing()}},

      // shmem_init_thread(int requested, int *provided):
      //   index 0 = requested (THREAD_LEVEL kind in JSON)
      //   index 1 = provided  (THREAD_LEVEL kind in JSON, output pointer)
      {"shmem_init_thread",
       {"spmd.init",
        Operands(
            O("threadRequired", 0)), // handleSHMEMOperand handles THREAD_LEVEL
        Results(R("threadProvided", 1, castThreadLevelToI32)), newAttributes(),
        postProcessing()}},

      {"shmem_finalize",
       {"spmd.finalize", Operands(), Results(), newAttributes(),
        postProcessing()}},
      {"shmem_global_exit",
       {"spmd.abort", Operands(D("comm", getWorldComm), O("errorCode", 0)),
        Results(), newAttributes(), postProcessing()}},
      {"shmem_my_pe",
       {"spmd.getRankInComm", Operands(D("comm", getWorldComm)),
        Results(RR("rank", 0)), newAttributes(), postProcessing()}},

      {"shmem_n_pes",
       {"spmd.getSizeOfComm", Operands(D("comm", getWorldComm)),
        Results(RR("size", 0)), newAttributes(), postProcessing()}},

      {"shmem_barrier_all",
       {"spmd.fence", Operands(/*M("win"),*/ D("assert", zeroI32)), Results(),
        newAttributes(), postProcessing()}},

      {"shmem_team_my_pe",
       {"spmd.getRankInComm", Operands(O("comm", 0)), Results(RR("rank", 0)),
        newAttributes(), postProcessing()}},

      {"shmem_team_n_pes",
       {"spmd.getSizeOfComm", Operands(O("comm", 0)), Results(RR("rank", 0)),
        newAttributes(), postProcessing()}},

      {"shmem_team_sync",
       {"spmd.barrier", Operands(O("comm", 0)), Results(), newAttributes(),
        postProcessing()}},

      {"shmem_sync_all",
       {"spmd.barrier", Operands(D("comm", getWorldComm)), Results(),
        newAttributes(), postProcessing()}},

      {"shmem_malloc",
       {"spmd.malloc",
        Operands(D("comm", getWorldComm), O("size", 0, convertI32ToI64)),
        Results(RR("address", 0)), newAttributes(), postProcessing()}},

      // void *shmem_align(size_t alignment, size_t size): the alignment
      // comes first, unlike spmd.malloc's operands.
      {"shmem_align",
       {"spmd.malloc",
        Operands(D("comm", getWorldComm), O("size", 1, convertI32ToI64),
                 O("alignment", 0, convertI32ToI64)),
        Results(RR("address", 0)), newAttributes(), postProcessing()}},

      {"shmem_realloc",
       {"spmd.realloc",
        Operands(D("comm", getWorldComm), O("oldAddress", 0),
                 O("size", 1, convertI32ToI64)),
        Results(RR("newAddress", 0)), newAttributes(), postProcessing()}},

      {"shmem_free",
       {"spmd.free", Operands(D("comm", getWorldComm), O("address", 0)),
        Results(), newAttributes(), postProcessing()}},

      {"shmem_team_split_strided",
       {"spmd.commSplitStrided",
        Operands(O("parentComm", 0), O("start", 1), O("stride", 2),
                 O("size", 3)),
        //  O("commConfig", 4)),
        Results(R("newComm", 6)), newAttributes(), postProcessing()}},

      {"shmem_team_destroy",
       {"spmd.commDestroy", Operands(O("comm", 0)), Results(), newAttributes(),
        postProcessing()}},

      {"shmem_quiet",
       {"spmd.flushAll", Operands(), Results(), newAttributes(getNotLocalAttr),
        postProcessing()}},

      {"shmem_fence",
       {"spmd.fenceOrdering", Operands(), Results(), newAttributes(),
        postProcessing()}}

  };

  for (auto vectorPair : reductionTypesAndOps) {
    llvm::SmallVector<llvm::StringRef, 32> types = vectorPair.first;
    llvm::SmallVector<llvm::StringRef, 8> operands = vectorPair.second;

    // iterate over cartesian product
    for (llvm::StringRef stringType : types) {
      for (llvm::StringRef reduceOp : operands) {
        std::function<Value(OpBuilder &, Location, int)> getDatatype;
        getDatatype = [stringType](OpBuilder &builder, Location loc,
                                   int modelGroup) {
          Type type = getShmemElementType(builder, stringType);
          return spmd::ConstDatatypeOp::create(builder, loc, type, modelGroup)
              .getResult();
        };
        std::function<Value(OpBuilder &, Location, int)> getFetchOp;
        getFetchOp = [reduceOp](OpBuilder &builder, Location loc,
                                int modelGroup) {
          return spmd::ConstOpTypeOp::create(builder, loc, reduceOp, modelGroup)
              .getResult();
        };

        map[opApiCallName(
            reduceOp,
            typeApiCallName(stringType, "shmem_TYPENAME_OP_reduce"))] =
            APICallEntry{"spmd.allreduce",
                         Operands(O("comm", 0), O("sendBuf", 2),
                                  O("recvBuf", 1), O("dataCount", 3),
                                  D("datatype", getDatatype),
                                  D("reduceOp", getFetchOp)),
                         Results(), newAttributes(), postProcessing()};
      }
    }
  }

  for (llvm::StringRef stringType : standardRmaTypes) {
    std::function<Value(OpBuilder &, Location, int)> getDatatype;
    getDatatype = [stringType](OpBuilder &builder, Location loc,
                               int modelGroup) {
      Type type = getShmemElementType(builder, stringType);
      return spmd::ConstDatatypeOp::create(builder, loc, type, modelGroup)
          .getResult();
    };

    map[typeApiCallName(stringType, "shmem_TYPENAME_put")] = APICallEntry{
        "spmd.put",
        Operands(O("originAddr", 1), O("originCount", 2),
                 D("originDatatype", getDatatype), D("originStride", oneI64),
                 O("targetRank", 3), O("targetWin", 0, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 2),
                 D("targetDatatype", getDatatype), D("targetStride", oneI64)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_put_signal")] =
        APICallEntry{"spmd.putSignal",
                     Operands(O("originAddr", 1), O("originCount", 2),
                              D("originDatatype", getDatatype),
                              D("originStride", oneI64), O("targetRank", 6),
                              O("targetWin", 0, createDefaultWindow),
                              D("targetDisp", zeroI64), O("targetCount", 2),
                              D("targetDatatype", getDatatype),
                              D("targetStride", oneI64),
                              O("targetSignalWin", 3, createDefaultWindow),
                              O("signal", 4), O("signalOp", 5)),
                     Results(), newAttributes(getNotAtomicAttr),
                     postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_p")] = APICallEntry{
        "spmd.put",
        Operands(O("originAddr", 1), D("originCount", oneI64),
                 D("originDatatype", getDatatype), D("originStride", oneI64),
                 O("targetRank", 2), O("targetWin", 0, createDefaultWindow),
                 D("targetDisp", zeroI64), D("targetCount", oneI64),
                 D("targetDatatype", getDatatype), D("targetStride", oneI64)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_iput")] = APICallEntry{
        "spmd.put",
        Operands(O("originAddr", 1), O("originCount", 4),
                 D("originDatatype", getDatatype), O("originStride", 3),
                 O("targetRank", 5), O("targetWin", 0, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 4),
                 D("targetDatatype", getDatatype), O("targetStride", 2)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_get")] = APICallEntry{
        "spmd.get",
        Operands(O("originAddr", 0), O("originCount", 2),
                 D("originDatatype", getDatatype), D("originStride", oneI64),
                 O("targetRank", 3), O("targetWin", 1, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 2),
                 D("targetDatatype", getDatatype), D("targetStride", oneI64)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_g")] = APICallEntry{
        "spmd.get",
        Operands(V("originAddr", 0), D("originCount", oneI64),
                 D("originDatatype", getDatatype), D("originStride", oneI64),
                 O("targetRank", 1), O("targetWin", 0, createDefaultWindow),
                 D("targetDisp", zeroI64), D("targetCount", oneI64),
                 D("targetDatatype", getDatatype), D("targetStride", oneI64)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_iget")] = APICallEntry{
        "spmd.get",
        Operands(O("originAddr", 0), O("originCount", 4),
                 D("originDatatype", getDatatype), O("originStride", 3),
                 O("targetRank", 5), O("targetWin", 1, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 4),
                 D("targetDatatype", getDatatype), O("targetStride", 2)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_alltoall")] =
        APICallEntry{"spmd.alltoall",
                     Operands(O("comm", 0), O("sendBuf", 2), O("sendCount", 3),
                              D("sendType", getDatatype), O("recvBuf", 1),
                              O("recvCount", 3), D("recvType", getDatatype)),
                     Results(), newAttributes(), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_broadcast")] =
        APICallEntry{"spmd.bcast",
                     Operands(O("comm", 0), O("sendBuf", 2), O("recvBuf", 1),
                              O("dataCount", 3), D("dataType", getDatatype),
                              O("rootRank", 4)),
                     Results(), newAttributes(), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_collect")] =
        APICallEntry{"spmd.allgather",
                     Operands(O("comm", 0), O("sendBuf", 2), O("sendCount", 3),
                              D("sendType", getDatatype), O("recvBuf", 1),
                              O("recvCount", 3), D("recvType", getDatatype)),
                     Results(), newAttributes(), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_fcollect")] =
        APICallEntry{"spmd.allgather",
                     Operands(O("comm", 0), O("sendBuf", 2), O("sendCount", 3),
                              D("sendType", getDatatype), O("recvBuf", 1),
                              O("recvCount", 3), D("recvType", getDatatype)),
                     Results(), newAttributes(), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_put_nbi")] =
        makeNonBlocking(
            map.lookup(typeApiCallName(stringType, "shmem_TYPENAME_put")));
    map[typeApiCallName(stringType, "shmem_TYPENAME_put_signal_nbi")] =
        makeNonBlocking(map.lookup(
            typeApiCallName(stringType, "shmem_TYPENAME_put_signal")));
    map[typeApiCallName(stringType, "shmem_TYPENAME_get_nbi")] =
        makeNonBlocking(
            map.lookup(typeApiCallName(stringType, "shmem_TYPENAME_get")));
  }

  for (llvm::StringRef stringType : standardAmoTypes) {
    std::function<Value(OpBuilder &, Location, int)> getDatatype;
    getDatatype = [stringType](OpBuilder &builder, Location loc,
                               int modelGroup) {
      Type type = getShmemElementType(builder, stringType);
      return spmd::ConstDatatypeOp::create(builder, loc, type, modelGroup)
          .getResult();
    };

    for (llvm::StringRef reduceOp : fetchOpOptions) {
      std::function<Value(OpBuilder &, Location, int)> getFetchOp;
      getFetchOp = [reduceOp](OpBuilder &builder, Location loc,
                              int modelGroup) {
        return spmd::ConstOpTypeOp::create(builder, loc, reduceOp, modelGroup)
            .getResult();
      };

      if (reduceOp == "inc") {
        map[opApiCallName(
            reduceOp,
            typeApiCallName(stringType, "shmem_TYPENAME_atomic_fetch_OP"))] =
            APICallEntry{
                "spmd.getAccumulate",
                Operands(
                    D("originAddr", zeroPtr), D("originCount", oneI64),
                    D("originDatatype", getDatatype), D("originStride", oneI64),
                    V("resultAddr", 0), D("resultCount", oneI64),
                    D("resultDatatype", getDatatype), D("resultStride", oneI64),
                    O("targetRank", 1), O("targetWin", 0, createDefaultWindow),
                    D("targetDisp", zeroI64), D("targetCount", oneI64),
                    D("targetDatatype", getDatatype), D("targetStride", oneI64),
                    D("accOp", getFetchOp)),
                Results(), newAttributes(getAtomicAttr), postProcessing()};

        // fetch_inc NBI
        map[opApiCallName(
            reduceOp, typeApiCallName(stringType,
                                      "shmem_TYPENAME_atomic_fetch_OP_nbi"))] =
            APICallEntry{
                "spmd.getAccumulate",
                Operands(
                    D("originAddr", zeroPtr), D("originCount", oneI64),
                    D("originDatatype", getDatatype), D("originStride", oneI64),
                    O("resultAddr", 0), // caller provides result buffer
                    D("resultCount", oneI64), D("resultDatatype", getDatatype),
                    D("resultStride", oneI64),
                    O("targetRank", 2), // shifted: fetch[0], dest[1], pe[2]
                    O("targetWin", 1, createDefaultWindow), // dest is operand 1
                    D("targetDisp", zeroI64), D("targetCount", oneI64),
                    D("targetDatatype", getDatatype), D("targetStride", oneI64),
                    D("accOp", getFetchOp)),
                Results(), newAttributes(getAtomicAttr), postProcessing()};
      } else {
        map[opApiCallName(
            reduceOp,
            typeApiCallName(stringType, "shmem_TYPENAME_atomic_fetch_OP"))] =
            APICallEntry{
                "spmd.getAccumulate",
                Operands(
                    D("originAddr", zeroPtr), D("originCount", oneI64),
                    D("originDatatype", getDatatype), D("originStride", oneI64),
                    V("resultAddr", 0), D("resultCount", oneI64),
                    D("resultDatatype", getDatatype), D("resultStride", oneI64),
                    O("targetRank", 2), O("targetWin", 0, createDefaultWindow),
                    D("targetDisp", zeroI64), D("targetCount", oneI64),
                    D("targetDatatype", getDatatype), D("targetStride", oneI64),
                    D("accOp", getFetchOp)),
                Results(), newAttributes(getAtomicAttr), postProcessing()};
        // non-inc NBI
        map[opApiCallName(
            reduceOp, typeApiCallName(stringType,
                                      "shmem_TYPENAME_atomic_fetch_OP_nbi"))] =
            APICallEntry{
                "spmd.getAccumulate",
                Operands(
                    D("originAddr", zeroPtr), D("originCount", oneI64),
                    D("originDatatype", getDatatype), D("originStride", oneI64),
                    O("resultAddr", 0), // fetch[0]
                    D("resultCount", oneI64), D("resultDatatype", getDatatype),
                    D("resultStride", oneI64),
                    O("targetRank", 3), // fetch[0], dest[1], value[2], pe[3]
                    O("targetWin", 1, createDefaultWindow),
                    D("targetDisp", zeroI64), D("targetCount", oneI64),
                    D("targetDatatype", getDatatype), D("targetStride", oneI64),
                    D("accOp", getFetchOp)),
                Results(), newAttributes(getAtomicAttr), postProcessing()};
      }
    }

    map[typeApiCallName(stringType, "shmem_TYPENAME_atomic_compare_swap")] =
        APICallEntry{"spmd.compareAndSwap",
                     Operands(O("originAddress", 2), O("compareAddr", 1),
                              V("resultAddr", 0), D("type", getDatatype),
                              O("targetRank", 3),
                              O("targetWin", 0, createDefaultWindow),
                              D("targetDisp", zeroI64)),
                     Results(), newAttributes(getAtomicAttr), postProcessing()};
    map[typeApiCallName(stringType, "shmem_TYPENAME_atomic_compare_swap_nbi")] =
        APICallEntry{"spmd.compareAndSwap",
                     Operands(O("originAddress", 3), // value[3]
                              O("compareAddr", 2),   // cond[2]
                              O("resultAddr", 0), // fetch[0] — caller provides
                              D("type", getDatatype),
                              O("targetRank", 4),                     // pe[4]
                              O("targetWin", 1, createDefaultWindow), // dest[1]
                              D("targetDisp", zeroI64)),
                     Results(), newAttributes(getAtomicAttr), postProcessing()};
  }

  for (llvm::StringRef stringType : extendedAmoTypes) {
    std::function<Value(OpBuilder &, Location, int)> getDatatype;
    getDatatype = [stringType](OpBuilder &builder, Location loc,
                               int modelGroup) {
      Type type = getShmemElementType(builder, stringType);
      return spmd::ConstDatatypeOp::create(builder, loc, type, modelGroup)
          .getResult();
    };

    map[typeApiCallName(stringType, "shmem_TYPENAME_atomic_fetch")] =
        APICallEntry{
            "spmd.get",
            Operands(
                V("originAddr", 0), D("originCount", oneI64),
                D("originDatatype", getDatatype), D("originStride", oneI64),
                O("targetRank", 1), O("targetWin", 0, createDefaultWindow),
                D("targetDisp", zeroI64), D("targetCount", oneI64),
                D("targetDatatype", getDatatype), D("targetStride", oneI64)),
            Results(), newAttributes(getAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_atomic_fetch_nbi")] =
        APICallEntry{
            "spmd.get",
            Operands(O("originAddr", 0), // fetch[0] — caller provides
                     D("originCount", oneI64), D("originDatatype", getDatatype),
                     D("originStride", oneI64), O("targetRank", 2), // pe[2]
                     O("targetWin", 1, createDefaultWindow),        // source[1]
                     D("targetDisp", zeroI64), D("targetCount", oneI64),
                     D("targetDatatype", getDatatype),
                     D("targetStride", oneI64)),
            Results(), newAttributes(getAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_atomic_set")] =
        APICallEntry{
            "spmd.put",
            Operands(
                O("originAddr", 1), D("originCount", oneI64),
                D("originDatatype", getDatatype), D("originStride", oneI64),
                O("targetRank", 2), O("targetWin", 0, createDefaultWindow),
                D("targetDisp", zeroI64), D("targetCount", oneI64),
                D("targetDatatype", getDatatype), D("targetStride", oneI64)),
            Results(), newAttributes(getAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_atomic_swap")] =
        APICallEntry{"spmd.swap",
                     Operands(O("originBuf", 1), // value[1]
                              V("resultBuf",
                                0), // dest[0] — allocate buffer, load result
                              D("datatype", getDatatype),
                              O("targetRank", 2),                     // pe[2]
                              O("targetWin", 0, createDefaultWindow), // dest[0]
                              D("targetDisp", zeroI64)),
                     Results(), newAttributes(getAtomicAttr), postProcessing()};

    map[typeApiCallName(stringType, "shmem_TYPENAME_atomic_swap_nbi")] =
        APICallEntry{"spmd.swap",
                     Operands(O("originBuf", 2), // value[2]
                              O("resultBuf", 0), // fetch[0] — caller provides
                              D("datatype", getDatatype),
                              O("targetRank", 3),                     // pe[3]
                              O("targetWin", 1, createDefaultWindow), // dest[1]
                              D("targetDisp", zeroI64)),
                     Results(), newAttributes(getAtomicAttr), postProcessing()};
  }

  for (llvm::StringRef stringType : sizeOptions) {
    std::function<Value(OpBuilder &, Location, int)> getDatatype;
    getDatatype = [stringType](OpBuilder &builder, Location loc,
                               int modelGroup) {
      Type type = getShmemElementType(builder, stringType);
      return spmd::ConstDatatypeOp::create(builder, loc, type, modelGroup)
          .getResult();
    };

    map[sizeApiCallName(stringType, "shmem_putSIZE")] = APICallEntry{
        "spmd.put",
        Operands(O("originAddr", 1), O("originCount", 2),
                 D("originDatatype", getDatatype), D("originStride", oneI64),
                 O("targetRank", 3), O("targetWin", 0, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 2),
                 D("targetDatatype", getDatatype), D("targetStride", oneI64)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[sizeApiCallName(stringType, "shmem_iputSIZE")] = APICallEntry{
        "spmd.put",
        Operands(O("originAddr", 1), O("originCount", 4),
                 D("originDatatype", getDatatype), O("originStride", 3),
                 O("targetRank", 5), O("targetWin", 0, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 4),
                 D("targetDatatype", getDatatype), O("targetStride", 2)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[sizeApiCallName(stringType, "shmem_getSIZE")] = APICallEntry{
        "spmd.get",
        Operands(O("originAddr", 0), O("originCount", 2),
                 D("originDatatype", getDatatype), D("originStride", oneI64),
                 O("targetRank", 3), O("targetWin", 1, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 2),
                 D("targetDatatype", getDatatype), D("targetStride", oneI64)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[sizeApiCallName(stringType, "shmem_igetSIZE")] = APICallEntry{
        "spmd.get",
        Operands(O("originAddr", 0), O("originCount", 4),
                 D("originDatatype", getDatatype), O("originStride", 3),
                 O("targetRank", 5), O("targetWin", 1, createDefaultWindow),
                 D("targetDisp", zeroI64), O("targetCount", 4),
                 D("targetDatatype", getDatatype), O("targetStride", 2)),
        Results(), newAttributes(getNotAtomicAttr), postProcessing()};

    map[sizeApiCallName(stringType, "shmem_putSIZE_nbi")] = makeNonBlocking(
        map.lookup(sizeApiCallName(stringType, "shmem_putSIZE")));
    map[sizeApiCallName(stringType, "shmem_getSIZE_nbi")] = makeNonBlocking(
        map.lookup(sizeApiCallName(stringType, "shmem_getSIZE")));
  }

  return map;
}

const ApiCallToSpmdMap shmemToSpmdOperands = createShmemToSpmdOperands();

} // namespace mlir::spmd
