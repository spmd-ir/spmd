//===- SPMDOps.td - SPMD dialect ops ----------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_OPS
#define SPMD_OPS

include "SPMDDialect.td"
include "SPMDTypes.td"
include "SPMDAttributes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"
// include "mlir/Dialect/LLVMIR/LLVMOpsInterfaces.td"

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/IR/Traits.td"
include "SPMDEffects.td"
include "SPMDTraits.td"
include "SPMDInterfaces.td"


def SPMD_InitOp : SPMD_Op<"init", [ProcessManagement, RuntimeAllocT<>, RuntimeWriteT<>]> {
  let summary = "";
  let description = [{
    Initialises the SPMD runtime.

    $threadRequired is always present (non-optional):
      - MPI_Init        → caller injects spmd.constThreadLevel("SINGLE")
      - MPI_Init_Thread → caller passes the converted required argument

    $threadProvided is optional:
      - MPI_Init        → absent (Single is already known from $threadRequired)
      - MPI_Init_Thread → present; carries the level actually granted by the
                          runtime; may flow into control-flow comparisons before
                          the enclosing spmd.parallel is entered.
  }];
  let arguments = (ins I32Attr              : $usedModel,
                       SPMD_ThreadLevelType : $threadRequired);
  let results = (outs Optional<SPMD_ThreadLevelType> : $threadProvided,
                      SPMD_ErrorType                 : $error);
}

def SPMD_FinalizeOp : SPMD_Op<"finalize", [ProcessManagement, RuntimeReadT<>, RuntimeFreeT<>]> {
  let arguments = (ins I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_GetRankInCommOp : SPMD_Op<"getRankInComm", [ProcessManagement, ResultsNonUniform, RuntimeReadT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]>: $comm,
                       I32Attr : $usedModel);
  let results = (outs I32 : $rank,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_GetDeviceInCommOp : SPMD_Op<"getDeviceInComm", [ProcessManagement, ResultsNonUniform, RuntimeReadT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]>: $comm,
                       I32Attr : $usedModel);
  let results = (outs I32 : $device,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_GetSizeOfCommOp : SPMD_Op<"getSizeOfComm", [ProcessManagement, RuntimeReadT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]>: $comm,
                       I32Attr : $usedModel);
  let results = (outs I32 : $size, 
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_CommWorldOp : SPMD_Op<"commWorld", [CommManagement, /*ConstantLike,*/ Pure]> {
  let arguments = (ins I32Attr : $usedModel);
  let results = (outs SPMD_CommType: $commWorld);
  // let regions = ();
  let summary = "";
  let description = [{}];
  // let hasFolder = 1;
}  

def SPMD_RequestNullOp : SPMD_Op<"requestNull", [ReqManagement,/*ConstantLike,*/ Pure]> {
  let arguments = (ins I32Attr : $usedModel);
  let results = (outs SPMD_ReqType : $reqNull);
  // let regions = ();
  let summary = "";
  let description = [{}];
  // let hasFolder = 1;
}

def SPMD_CommNodeOp : SPMD_Op<"commNode", [CommManagement, /*ConstantLike,*/ Pure]> {
  let arguments = (ins I32Attr : $usedModel);
  let results = (outs SPMD_CommType: $commNode);
  // let regions = ();
  let summary = "";
  let description = [{}];
  // let hasFolder = 1;
}


def SPMD_CommSplitStridedOp : SPMD_Op<"commSplitStrided", [CommManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "parent communicator", [CommRead<>]> : $parentComm,
                       I32 : $start,
                       I32 : $stride,
                       I32 : $size,
                       Optional<SPMD_CommConfigType> : $commConfig,
                      //  Optional<I64> : $commConfigMask, this argument is not practical as shmem has only one field in the config..
                      //  so this mask could be avoided by just passing NULL for the config or the real config when it wants to be used
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_CommType, "new communicator", [CommAlloc<>, CommWrite<>]> : $newComm,
                      SPMD_ErrorType : $error );
  let summary = "";
  let description = [{}];
}

def SPMD_CommDestroyOp : SPMD_Op<"commDestroy", [CommManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>, CommFree<>]> : $comm,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_CommSplitOp : SPMD_Op<"commSplit", [CommManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $parentComm,
                       I32 : $color,
                       I32 : $key,
                       Optional<SPMD_CommConfigType> : $commConfig,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_CommType, "", [CommAlloc<>, CommWrite<>]> : $newComm,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_CommDupOp : SPMD_Op<"commDup", [CommManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $oldComm,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_CommType, "", [CommAlloc<>, CommWrite<>]> : $newComm,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_WaitOp : SPMD_Op<"wait", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_ReqType, "", [ReqRead<>, ReqWrite<>]> : $req,
                       I32Attr : $usedModel);
  let results = (outs Optional<SPMD_StatusType> : $status,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_WaitAllOp : SPMD_Op<"waitAll", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $count,
                       Arg<MemRefOf<[SPMD_ReqType]>, "", [MemRead, MemWrite, ReqRead<>, ReqWrite<>]> : $reqs,
                       Res<Optional<MemRefOf<[SPMD_StatusType]>>, "", [MemWrite]> : $statuses,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_WaitSomeOp : SPMD_Op<"waitSome", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $inCount,
                       Arg<MemRefOf<[SPMD_ReqType]>, "", [MemRead, MemWrite, ReqRead<>, ReqWrite<>]> : $reqs,
                       Arg<MemRefOf<[I32]>, "", [MemWrite]> : $indices,
                       Arg<Optional<MemRefOf<[SPMD_StatusType]>>, "", [MemWrite]> : $statuses,
                       I32Attr : $usedModel);
  let results = (outs I32 : $outCount,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_WaitAnyOp : SPMD_Op<"waitAny", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $count,
                       Arg<MemRefOf<[SPMD_ReqType]>, "", [MemRead, MemWrite, ReqRead<>, ReqWrite<>]> : $reqs,
                       I32Attr : $usedModel);
  let results = (outs I32 : $index,
                      Optional<SPMD_StatusType> : $status,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_TestOp : SPMD_Op<"test", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_ReqType, "", [ReqRead<>]> : $req,
                       I32Attr : $usedModel);
  let results = (outs I1 : $flag,
                      Optional<SPMD_StatusType> : $status,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_TestAllOp : SPMD_Op<"testAll", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $count,
                       Arg<MemRefOf<[SPMD_ReqType]>, "", [MemRead, MemWrite, ReqRead<>]> : $reqs,
                       Res<Optional<MemRefOf<[SPMD_StatusType]>>, "", [MemWrite]> : $statuses,
                       I32Attr : $usedModel);
  let results = (outs I1             : $flag,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_TestSomeOp : SPMD_Op<"testSome", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $inCount,
                       Arg<MemRefOf<[SPMD_ReqType]>, "", [MemRead, MemWrite, ReqRead<>]> : $reqs,
                       Arg<MemRefOf<[I32]>, "", [MemWrite]>                              : $indices,
                       Res<Optional<MemRefOf<[SPMD_StatusType]>>, "", [MemWrite]>        : $statuses,
                       I32Attr : $usedModel);
  let results = (outs I32            : $outCount,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_TestAnyOp : SPMD_Op<"testAny", [NonBlockingSync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $count,
                       Arg<MemRefOf<[SPMD_ReqType]>, "", [MemRead, MemWrite, ReqRead<>]> : $reqs,
                       I32Attr : $usedModel);
  let results = (outs I32                            : $index,
                      I1                             : $flag,
                      Optional<SPMD_StatusType>      : $status,
                      SPMD_ErrorType                 : $error);
  let summary = "";
  let description = [{}];
}


def SPMD_ConstDatatypeOp : SPMD_Op<"constDatatype", [/*ConstantLike,*/ Pure]> {
  let arguments = (ins TypeAttr : $typeAttr,
                       I32Attr : $usedModel);
  let results = (outs SPMD_DatatypeType: $constDatatype);
  // let regions = ();
  let summary = "";
  let description = [{
    A predefined datatype, as the MLIR type of one element.  An integer is
    signed (`si32`) or unsigned (`ui32`) when the datatype says which --
    MPI_INT and MPI_UNSIGNED, shmem_int_* and shmem_uint_*, ncclInt32 and
    ncclUint32 -- and signless (`i32`) when it does not: C leaves the
    signedness of `char` to the platform, MPI_BYTE is no number, and
    shmem_put32 moves words.  A maximum or minimum depends on the signedness,
    so the lowerings refuse one over a signless integer where the target
    would have to guess it.

    The type describes the datatype, not a value: `si32` and `ui32` are not
    LLVM types, and nothing may build an LLVM operation from it without
    dropping the signedness first.
  }];
  // let hasFolder = 1;
}

def SPMD_ConstOpTypeOp : SPMD_Op<"constOpType", [/*ConstantLike,*/ Pure]> {
  let arguments = (ins StrAttr : $stringAttr,
                       I32Attr : $usedModel);
  let results = (outs SPMD_OpType: $constOpType);
  // let regions = ();
  let summary = "";
  let description = [{}];
  // let hasFolder = 1;
}

def SPMD_ConstSignalOpTypeOp : SPMD_Op<"constSignalOpType", [/*ConstantLike,*/ Pure]> {
  let arguments = (ins StrAttr : $stringAttr,
                       I32Attr : $usedModel);
  let results = (outs SPMD_SignalOpType: $constSignalOpType);
  // let regions = ();
  let summary = "";
  let description = [{}];
  // let hasFolder = 1;
}

def SPMD_ConstThreadLevelOp : SPMD_Op<"constThreadLevel", [Pure]> {
  let summary = "Produces a constant !spmd.threadLevel value";
  let description = [{
    Maps a string literal ("SINGLE", "FUNNELED", "SERIALIZED", "MULTIPLE")
    to the corresponding `!spmd.threadLevel` SSA value.
    Used when converting the `required` argument of `MPI_Init_Thread`, and
    when injecting the implied SINGLE level for plain `MPI_Init`.
  }];
  let arguments = (ins StrAttr : $level,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ThreadLevelType : $threadLevel);
}

def SPMD_CastThreadLevelToI32Op : SPMD_Op<"castThreadLevelToI32", [Pure]> {
  let summary = "Casts !spmd.threadLevel to i32";
  let description = [{
    Lowers a thread-level SSA value to a plain integer so it can be stored
    back into an `int *provided` pointer or used in integer comparisons
    such as `if (provided < MPI_THREAD_FUNNELED)`.
  }];
  let arguments = (ins SPMD_ThreadLevelType : $threadLevel);
  let results = (outs I32 : $result);
}

def SPMD_CastStreamOp : SPMD_Op<"castStream", [Pure/*, ViewLikeOpInterface*/]> {
  let arguments = (ins SPMD_AnyAddressLike : $oldStream,
                       I32Attr : $usedModel);
  let results = (outs SPMD_StreamType: $newStream);
  // let regions = ();
  let summary = "";
  let description = [{}];
}

// possibly also RuntimeAllocT
def SPMD_MallocOp : SPMD_Op<"malloc", [DataManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       I64 : $size,
                       Optional<I64> : $alignment,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_AnyAddressLike, "", [MemAlloc]> : $address,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

// possibly also RuntimeAllocT
def SPMD_ReallocOp : SPMD_Op<"realloc", [DataManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "", [MemFree]>:$oldAddress,
                       I64 : $size,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_AnyAddressLike, "", [MemAlloc]> : $newAddress,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

// possibly also RuntimeFreeT
def SPMD_FreeOp : SPMD_Op<"free", [DataManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "", [MemFree]>:$address,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

// memory effects specified in SPMDOPS.cpp
// has a write effect, but not tied to any operand
// Todo: is this really necessary to have nontied write effect? can it be just a normal effect tied one of the operands?
def SPMD_SendOp : SPMD_Op<"send", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, P2P, Communication,  RuntimeWriteT<>, RuntimeReadT<> ]> { //MemoryEffects<[RuntimeRead<>, RuntimeWrite<>]>,
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "", [MemRead]> : $sendBuf, 
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       I32 : $destRank,
                       I32 : $tag,
                       Arg<Optional<SPMD_StreamType>, "stream", [StreamRead<>, StreamWrite<>]> : $stream,  
                       BoolAttr : $isBlocking,
                       BoolAttr : $isBuffered,
                       BoolAttr : $isReadyMode,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req, // MemAlloc
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}


def SPMD_RecvOp : SPMD_Op<"recv", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, P2P, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "", [MemWrite]> : $recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       I32 : $srcRank,
                       I32 : $tag,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      // SPMD_StatusType: status,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_SendrecvOp : SPMD_Op<"sendrecv", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, P2P, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "send address", [MemRead]> : $sendBuf,
                       I64 : $sendCount,
                       SPMD_DatatypeType : $sendDatatype,
                       I64 : $sendStride,
                       I32 : $sendRank,
                       I32 : $sendTag,
                       Arg<SPMD_AnyAddressLike, "recv address", [MemWrite]> : $recvBuf,
                       I64 : $recvCount,
                       SPMD_DatatypeType : $recvDatatype,
                       I64 : $recvStride,
                       I32 : $recvRank,
                       I32 : $recvTag,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  // $req is present exactly for the non-blocking form (MPI_Isendrecv,
  // MPI_Isendrecv_replace), as on send and recv.
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      // SPMD_StatusType: $status,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_BarrierOp : SPMD_Op<"barrier", [Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel,
                       // NEW: OMP barrier semantics — every thread must reach THIS op instance.
                       // When true the analysis will refuse to match this record against any
                       // other op instance, even one with an identical key.
                       OptionalAttr<BoolAttr>:$isUniqueMatching);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_BcastOp : SPMD_Op<"bcast", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                      Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]>:$sendBuf,
                      Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       I32 : $rootRank,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_ReduceOp : SPMD_Op<"reduce", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                      Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]>:$sendBuf,
                      Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       SPMD_OpType : $reduceOp,
                       I32 : $rootRank,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_AllreduceOp : SPMD_Op<"allreduce", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                      Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]>:$sendBuf,
                      Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       SPMD_OpType : $reduceOp,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_ReduceScatterOp : SPMD_Op<"reduceScatter", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                      Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]> : $sendBuf,
                      Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]> : $recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       SPMD_OpType : $reduceOp,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_ScanOp : SPMD_Op<"scan", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                      Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]>:$sendBuf,
                      Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       SPMD_OpType : $reduceOp,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_ExscanOp : SPMD_Op<"exscan", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                      Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]>:$sendBuf,
                      Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       SPMD_OpType : $reduceOp,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_ScatterOp : SPMD_Op<"scatter", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                      Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]>:$sendBuf,
                       I64 : $sendCount,
                       SPMD_DatatypeType : $sendType,
                      Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $recvCount,
                       SPMD_DatatypeType : $recvType,
                       I32 : $rootRank,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_GatherOp : SPMD_Op<"gather", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "send buffer",
                             [MemRead]>:$sendBuf,
                       I64 : $sendCount,
                       SPMD_DatatypeType : $sendType,
                       Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $recvCount,
                       SPMD_DatatypeType : $recvType,
                       I32 : $rootRank,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_AllgatherOp : SPMD_Op<"allgather", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "send buffer",
                            [MemRead]>:$sendBuf,
                       I64 : $sendCount,
                       SPMD_DatatypeType : $sendType,
                       Arg<SPMD_AnyAddressLike, "recv buffer",
                             [MemWrite]>:$recvBuf,
                       I64 : $recvCount,
                       SPMD_DatatypeType : $recvType,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_AlltoallOp : SPMD_Op<"alltoall", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, Collective, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommWrite<>, CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "send buffer",
                            [MemRead]>:$sendBuf,
                       I64 : $sendCount,
                       SPMD_DatatypeType : $sendType,
                       Arg<SPMD_AnyAddressLike, "recv buffer",
                            [MemWrite]>:$recvBuf,
                       I64 : $recvCount,
                       SPMD_DatatypeType : $recvType,
                       Arg<Optional<SPMD_StreamType>, "stream",
                             [StreamRead<>, StreamWrite<>]> : $stream,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}


def SPMD_AbortOp : SPMD_Op<"abort", [ProcessManagement, RuntimeReadT<>, RuntimeWriteT<>, RuntimeFreeT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       I32 : $errorCode,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_ReqFreeOp : SPMD_Op<"reqFree", [ReqManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_ReqType, "", [MemFree]> : $req,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

/* ***************************************************************************************************** */
/* ------------------------------------------------ RMA ------------------------------------------------ */
/* ***************************************************************************************************** */

/* ----------------------------------------- Window Management ----------------------------------------- */
// def SPMD_WindowOp : SPMD_Op<"window", [RMA]> {
//   let arguments = (ins BoolAttr : $isAtomic,
//                        I32Attr : $usedModel);
//   let results = (outs SPMD_WinType : $win,
//                       SPMD_ErrorType : $error);
//   let summary = "";
//   let description = [{}];
// }

def SPMD_WinCreateOp : SPMD_Op<"winCreate", [RMA, DataManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "", [MemRead]> : $baseAddress,
                       Optional<I64> : $size,
                       I32 : $dispUnit,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_WinType, "", [WinAlloc<>, WinWrite<>]> : $win,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_WinAllocOp : SPMD_Op<"winAlloc", [RMA, DataManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       I64 : $size,
                       I32 : $dispUnit,
                      // Optional<I64> : $num_elems,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_AnyAddressLike, "", [MemAlloc]> : $baseAddress,
                      Res<SPMD_WinType, "", [WinAlloc<>, WinWrite<>]> : $win,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_WinFreeOp : SPMD_Op<"winFree", [RMA, DataManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_WinType, "", [WinRead<>, WinFree<>]> : $win,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

/* ------------------------------------------ Synchronisation ------------------------------------------ */
def SPMD_FenceOp : SPMD_Op<"fence", [RMA, Sync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<Optional<SPMD_WinType>, "", [WinWrite<>, WinRead<>]> : $win,
                       I32 : $assert,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_FenceOrderingOp : SPMD_Op<"fenceOrdering", [RMA, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<Optional<SPMD_WinType>, "", [WinWrite<>, WinRead<>]> : $win,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_FlushOp : SPMD_Op<"flush", [RMA, Sync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $rank,
                       Arg<SPMD_WinType, "", [WinWrite<>, WinRead<>]> : $win,
                       BoolAttr : $isLocal,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_FlushAllOp : SPMD_Op<"flushAll", [RMA, Sync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<Optional<SPMD_WinType>, "", [WinWrite<>, WinRead<>]> : $win,
                       BoolAttr : $isLocal,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_ConstLockTypeOp : SPMD_Op<"constLockTypeOp", [/*ConstantLike,*/ Pure]> {
  let arguments = (ins StrAttr : $stringAttr,
                       I32Attr : $usedModel);
  let results = (outs SPMD_LockType: $constLockTypeOp);
  // let regions = ();
  let summary = "";
  let description = [{}];
  // let hasFolder = 1;
}

def SPMD_LockOp : SPMD_Op<"lock", [RMA, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $rank,
                       Arg<SPMD_WinType, "", [WinWrite<>, WinRead<>]> : $win,
                       SPMD_LockType : $lockType,
                       I32 : $assert,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_LockAllOp : SPMD_Op<"lockAll", [RMA, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_WinType, "", [WinWrite<>, WinRead<>]> : $win,
                       I32 : $assert,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_UnlockOp : SPMD_Op<"unlock", [RMA, Sync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $rank,
                       Arg<SPMD_WinType, "", [WinWrite<>, WinRead<>]> : $win,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_UnlockAllOp : SPMD_Op<"unlockAll", [RMA, Sync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_WinType, "", [WinWrite<>, WinRead<>]> : $win,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

// NCCL RMA calls
/*
def SPMD_SignalOp : SPMD_Op<"signal", [RMA, Sync]> {
  let arguments = (ins SPMD_CommType : $comm,
                       I32 : $targetRank,
                       I32 : $sigIdx,
                       I32 : $ctx,
                       I32 : $flags,
                       SPMD_StreamType : $stream,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs Optional<SPMD_ReqType> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_WaitSignalOp : SPMD_Op<"waitSignal", [RMA, Sync]> {
  let arguments = (ins SPMD_CommType : $comm,
                       I32 : $nDesc,
                       Arg<MemRefOf<[SPMD_WaitSignalDescriptorType]>, "", []> :
                        $signalDescs,
                       SPMD_StreamType : $stream,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs Optional<SPMD_ReqType> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}
*/

def SPMD_GroupStartOp : SPMD_Op<"groupStart", [RMA, Sync]> {
  let arguments = (ins I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_GroupEndOp : SPMD_Op<"groupEnd", [RMA, Sync]> {
  let arguments = (ins I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

/* ------------------------------------------- Communication ------------------------------------------- */
def SPMD_PutOp : SPMD_Op<"put", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins
    Arg<SPMD_AnyAddressLike, "origin address", [MemRead]> : $originAddr,
    I64 : $originCount,
    SPMD_DatatypeType : $originDatatype,
    I64 : $originStride,
    I32 : $targetRank,
    Arg<SPMD_WinType, "target window", [MemRead]> : $targetWin,
    I64 : $targetDisp,
    I64 : $targetCount,
    SPMD_DatatypeType : $targetDatatype,
    I64 : $targetStride,
    BoolAttr : $isBlocking,
    BoolAttr : $isAtomic,
    I32Attr : $usedModel
  );
  let results = (outs
    Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
    SPMD_ErrorType : $error
  );
  let summary = "";
  let description = [{}];
}


def SPMD_PutSignalOp : SPMD_Op<"putSignal", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "origin data address", [MemRead]> : $originAddr,
                       I64 : $originCount,
                       SPMD_DatatypeType : $originDatatype,
                       I64 : $originStride,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "target data window", [MemRead]> : $targetWin,
                       I64 : $targetDisp,
                       I64 : $targetCount,
                       SPMD_DatatypeType : $targetDatatype,
                       I64 : $targetStride,
                       Arg<SPMD_WinType, "target signal window", [MemRead]> : $targetSignalWin,
                       I64 : $signal,
                       SPMD_SignalOpType : $signalOp,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

// NCCL RMA calls
/*def SPMD_PutImplicitSignalOp : SPMD_Op<"putImplicitSignalOp", [RMA, Communication]> {
  let arguments = (ins SPMD_CommType : $comm,
                       Arg<SPMD_AnyAddressLike, "origin data address", [MemRead]> : $originAddr,
                       I64 : $originCount,
                       SPMD_DatatypeType : $originDatatype,
                       I64 : $originStride,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "target data window", [MemWrite]> : $targetWin,
                       I64 : $targetDisp,
                       I64 : $targetCount,
                       SPMD_DatatypeType : $targetDatatype,
                       I64 : $targetStride,
                       I32 : $sigIdx,
                       I32 : $ctx,
                       I32 : $flags,
                       SPMD_StreamType : $stream,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs Optional<SPMD_ReqType> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}*/

def SPMD_GetOp : SPMD_Op<"get", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "origin address", [MemWrite]> : $originAddr,
                       I64 : $originCount,
                       SPMD_DatatypeType : $originDatatype,
                       I64 : $originStride,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "target window", [MemRead]> : $targetWin,
                       I64 : $targetDisp,
                       I64 : $targetCount,
                       SPMD_DatatypeType : $targetDatatype,
                       I64 : $targetStride,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_AccumulateOp : SPMD_Op<"accumulate", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "origin address", [MemRead]> : $originAddr,
                       I64 : $originCount,
                       SPMD_DatatypeType : $originDatatype,
                       I64 : $originStride,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "", [MemRead]> : $targetWin,
                       I64 : $targetDisp,
                       I64 : $targetCount,
                       SPMD_DatatypeType : $targetDatatype,
                       I64 : $targetStride,
                       SPMD_OpType : $accOp,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_GetAccumulateOp : SPMD_Op<"getAccumulate", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "origin address", [MemRead]> : $originAddr,
                       I64 : $originCount,
                       SPMD_DatatypeType : $originDatatype,
                       I64 : $originStride,
                       Arg<SPMD_AnyAddressLike, "result address", [MemWrite]> : $resultAddr,
                       I64 : $resultCount,
                       SPMD_DatatypeType : $resultDatatype,
                       I64 : $resultStride,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "", [MemRead]> : $targetWin,
                       I64 : $targetDisp,
                       I64 : $targetCount,
                       SPMD_DatatypeType : $targetDatatype,
                       I64 : $targetStride,
                       SPMD_OpType : $accOp,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs Res<Optional<SPMD_ReqType>, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_SwapOp : SPMD_Op<"swap", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<AnyMemRef, "origin buffer", [MemRead]> : $originBuf,
                       Arg<AnyMemRef, "result buffer", [MemRead]> : $resultBuf,
                       SPMD_DatatypeType : $datatype,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "", [MemRead]> : $targetWin,
                       I64 : $targetDisposition,
                       BoolAttr : $isAtomic,
                       BoolAttr: $isNonBlocking,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_CompareAndSwapOp : SPMD_Op<"compareAndSwap", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "origin address", [MemRead]> : $originAddr,
                       Arg<SPMD_AnyAddressLike, "compare address", [MemRead]> : $compareAddr,
                       Arg<SPMD_AnyAddressLike, "result address", [MemWrite]> : $resultAddr,
                       SPMD_DatatypeType : $type,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "", [MemRead]> : $targetWin,
                       I64 : $targetDisposition,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_FetchAndOpOp : SPMD_Op<"fetchAndOp", [RMA, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "origin address", [MemRead]> : $originAddr,
                       Arg<SPMD_AnyAddressLike, "result address", [MemWrite]> : $resultAddr,
                       SPMD_DatatypeType : $type,
                       SPMD_OpType : $reduceOp,
                       I32 : $targetRank,
                       Arg<SPMD_WinType, "", [MemRead]> : $targetWin,
                       I64 : $targetDisposition,
                       BoolAttr: $isBlocking,
                       BoolAttr : $isAtomic, 
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

/* ***************************************************************************************************** */
/* -------------------------------------------- Device Stuff ------------------------------------------- */
/* ***************************************************************************************************** */

/* ----------------------------------------- Device Memory Management ----------------------------------------- */
// later when not operating anymore on llvm ptrs, we could use gpu dialect instead, it provices similar operations

def SPMD_MemcpyOp : SPMD_Op<"memcpy", [RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "", [MemWrite]> : $dst,
                       Arg<SPMD_AnyAddressLike, "", [MemRead]> : $src,
                       I64 : $count,
                       I32 : $kind,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_MemsetOp : SPMD_Op<"memset", [RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "", [MemWrite]> : $dst,
                       I32 : $value,
                       I64 : $count,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

/* ***************************************************************************************************** */
/* -------------------------------------------- MPI Specific Stuff ------------------------------------------- */
/* ***************************************************************************************************** */
def SPMD_DimsCreateOp : SPMD_Op<"dimsCreate", []> {
  let arguments = (ins I32 : $nProc,
                       I32 : $nDims,
                       Arg<SPMD_AnyAddressLike, "", [MemRead, MemWrite]> : $dims,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

//TODO: reorder could be set to a custom type that eases static analysis
def SPMD_CartCreateOp: SPMD_Op<"cartCreate", [CommManagement, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $oldComm,
                       I32 : $nDims,
                       Arg<SPMD_AnyAddressLike, "", 
                          [MemRead]> : $dims,
                       Arg<SPMD_AnyAddressLike, "", 
                          [MemRead]> : $periods,
                       I32 : $reorder,
                       I32Attr : $usedModel);
  let results = (outs Arg<SPMD_CommType, "", [CommAlloc<>, CommWrite<>]> : $cartComm,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

//TODO: msg could be set to a custom type that eases static analysis
def SPMD_MProbeOp: SPMD_Op<"mProbe", [P2P, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       I32 : $srcRank,
                       I32 : $tag,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_MsgType, "", [MsgAlloc<>, MsgWrite<>]> : $msg,
                      Optional<SPMD_StatusType> : $status,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_MRecvOp: SPMD_Op<"mRecv", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, P2P, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "", [MemWrite]> : $recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       Arg<SPMD_MsgType, "", [MsgRead<>, MsgFree<>]> : $msg,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Optional<SPMD_StatusType> : $status,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

// created due to limitation of only having one optional result
def SPMD_IMRecvOp: SPMD_Op<"imRecv", [DeclareOpInterfaceMethods<SPMD_CommBufferOpInterface>, P2P, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_AnyAddressLike, "", [MemWrite]> : $recvBuf,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       Arg<SPMD_MsgType, "", [MsgRead<>, MsgFree<>]> : $msg,
                       BoolAttr : $isBlocking,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_ReqType, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_StartOp: SPMD_Op<"start", [P2P, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_ReqType, "", [ReqWrite<>, ReqRead<>]> : $req,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

//TODO: partition could be set to a custom type that eases static analysis
def SPMD_PReadyOp: SPMD_Op<"pReady", [P2P, Communication, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $partition,
                       Arg<SPMD_ReqType, "", [ReqWrite<>, ReqRead<>]> : $req,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}


def SPMD_SendInitOp : SPMD_Op<"sendInit", [P2P, Communication, Persistent, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "", [MemRead]> : $sendBuf,
                       Optional<I32> : $partitions,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       I32 : $destRank,
                       I32 : $tag,
                      //  Optional<SPMD_StreamType> : $stream,  // MemWrite, MemRead
                       //BoolAttr : $isBlocking, ToDo: json does not include isBlocking
                       BoolAttr : $isBuffered,
                       BoolAttr : $isReadyMode,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_ReqType, "", [ReqAlloc<>, ReqWrite<>]> : $req, //MemAlloc
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

def SPMD_RecvInitOp : SPMD_Op<"recvInit", [P2P, Communication, Persistent, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
                       Arg<SPMD_AnyAddressLike, "", [MemWrite]> : $recvBuf,
                       Optional<I32> : $partitions,
                       I64 : $dataCount,
                       SPMD_DatatypeType : $datatype,
                       I32 : $srcRank,
                       I32 : $tag,
                      //  Optional<SPMD_StreamType> : $stream,  // MemWrite, MemRead
                       //BoolAttr : $isBlocking, ToDo: json does not include isBlocking
                      //  BoolAttr : $isBuffered,
                       I32Attr : $usedModel);
  let results = (outs Res<SPMD_ReqType, "", [ReqAlloc<>, ReqWrite<>]> : $req,
                      SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}

//Todo: Incorporate Groups as typed values, and all other group ops
def SPMD_WinStartOp : SPMD_Op<"winStart", [RMA, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins I32 : $group,
                       I32 : $assert,
                       Arg<SPMD_WinType, "", [WinRead<>, WinWrite<>]> : $win,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}


def SPMD_WinCompleteOp : SPMD_Op<"winComplete", [RMA, Sync, RuntimeReadT<>, RuntimeWriteT<>]> {
  let arguments = (ins Arg<SPMD_WinType, "", [WinRead<>, WinWrite<>]> : $win,
                       I32Attr : $usedModel);
  let results = (outs SPMD_ErrorType : $error);
  let summary = "";
  let description = [{}];
}


// utility
def SPMD_CastSPMDErrorToI32Op : SPMD_Op<"castSPMDErrorToI32", [Pure]> {
  let summary = "Casts !spmd.error to i32";
  let description = [{
    Casts an SPMD error type to a plain i32.
    Useful when lowering SPMD-specific return values to standard integer codes
    (like MPI_SUCCESS).
  }];

  let arguments = (ins SPMD_ErrorType : $error);
  let results = (outs I32 : $result);
}

// --------------------------------------------------------------------------
// SPMD Operations for OpenMP/Threading support
// --------------------------------------------------------------------------

// --------------------------------------------------------------------------
// Region Terminators
// --------------------------------------------------------------------------

// we needed to change this op in order to handle the return value of the main function e.g. for MPI
def SPMD_TerminatorOp : SPMD_Op<"terminator", [Terminator]> {
  let summary = "Terminates an spmd.critical and single region";
  let description = [{
    The `spmd.terminator` operation marks the end of a control flow
    block within an `spmd.critical` and `spmd.single` region.
  }];

  let arguments = (ins);
  let results = (outs);
}

// New — dedicated to ParallelOp
// Used by: ParallelOp only
def SPMD_ParallelYieldOp : SPMD_Op<"parallel_yield", [
    Terminator,
    ReturnLike,
    ParentOneOf<["ParallelOp"]>   // enforced by type system
]> {
  let summary = "Terminates spmd.parallel, optionally yielding a process exit code";
  let description = [{
    Terminates an `spmd.parallel` region. The optional `$exit_code`
    carries the OS process exit code (i32) for process-level parallel
    regions (MPI, SHMEM, NVSHMEM). Absent for thread-level (OpenMP)
    and GPU kernel regions where exit codes are meaningless.
  }];

  let arguments = (ins Optional<I32>:$exit_code);
  let results = (outs);
}

def SPMD_YieldOp : SPMD_Op<"yield", [Pure, Terminator, ReturnLike]> {
  let summary = "Terminates an spmd.loop region";
  let description = [{
    Terminates a block in an `spmd.loop` region and optionally passes
    values to the next iteration or as the final result of the loop.
  }];

  let arguments = (ins Variadic<AnyType>:$results);

  let builders = [
    OpBuilder<(ins), [{ /* empty yield */ }]>
  ];
}

// --------------------------------------------------------------------------
// Communicator Retrieval for Orphaned Directives
// --------------------------------------------------------------------------

def SPMD_GetParallelCommOp : SPMD_Op<"getParallelComm", [Pure]> {
  let summary = "Retrieves the active communicator from the implicit context";
  let description = [{
    Retrieves the active SPMD communicator from the implicit execution
    context of the enclosing parallel region. Used to model orphaned
    OpenMP directives without modifying function signatures.
  }];

  let arguments = (ins);
  let results = (outs SPMD_CommType:$comm);
  let hasFolder = 1;
}

// its willingly not "isIsolatedFromAbove" and not implementing RegionBranchOpInterface
// instead of creating an explicit barrier, we could also make this op collective and add the attribute "isBlocking"
def SPMD_ParallelOp : SPMD_Op<"parallel", [
    // Collective,
    AutomaticAllocationScope,
    AttrSizedOperandSegments,
    RecursiveMemoryEffects,
    // SingleBlockImplicitTerminator<"ParallelYieldOp">,  <-- REMOVED:
    // only legal for single-block regions; multi-block main (cf.switch,
    // cf.br, …) needs AnyRegion + a custom verifier below.
    // RuntimeReadT<>, RuntimeWriteT<> (cant be specified, when RecursiveMemoryEffects is that is more important here)
]> {
  let summary = "Fork-join parallel region executed by a team of ranks";
  let description = [{
    Models MPI process-level and OpenMP thread-level parallelism.

    Shared variables are inferred from SSA: any value used inside the region
    but defined outside it and absent from $private_vars and $reduction_vars
    is implicitly shared. This avoids redundancy with the SSA structure.

    $private_vars covers both OpenMP private and firstprivate — the
    initialization distinction is irrelevant for data race analysis.

    $reduction_syms holds one FlatSymbolRefAttr per $reduction_var,
    identifying the operator needed to emit spmd.allreduce at the join point.

    With an explicit barrier before the ParallelYieldOp similar to omp parallel.

    $threadLevel records the thread-support level that was requested at
    spmd.init time, propagated by wrapInitFinalizeInParallel.
    Absent means the required level was not statically determinable;
    the analysis conservatively assumes Single.
  }];

let arguments = (ins
    Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
    Optional<I32>                        : $num_threads,
    Variadic<AnyType>                    : $private_vars,
    Variadic<AnyType>                    : $reduction_vars,
    OptionalAttr<ArrayAttr>              : $reduction_syms,
    OptionalAttr<SPMD_ThreadLevelAttr>   : $threadLevel,
    BoolAttr                             : $isProcessLevel,
    I32Attr                              : $usedModel);
  let results = (outs Optional<I32>:$exit_code);

  // AnyRegion: 1 block (simple main) or N blocks (cf.switch/cf.br main).
  let regions = (region AnyRegion : $region);

  let hasVerifier = 1;   // enforced in C++, see Part 2
}

// Mirrors omp.wsloop
// Probably noWait attribute should be called isBlocking to be inline 
// with the other collectives, or use explicit barriers
def SPMD_WsloopOp : SPMD_Op<"wsloop", [
    Collective,
    NoTerminator,
    SingleBlock,
    RecursiveMemoryEffects,
    GlobalConcurrency,
    AttrSizedOperandSegments
]> {
  let summary = "Worksharing-loop wrapper distributing iterations across ranks";
  let description = [{
    Mirrors `omp.wsloop`. Carries the execution context and barrier semantics.
    Must contain exactly one `spmd.loop_nest` in its region.
    This op is followed by an explicit barrier unless `$nowait` was set at
    omp origin.

    $reduction_vars and $reduction_syms mirror the same fields on spmd.parallel.
    The reduction combine is implicit (kept as metadata) for now — see the
    future-work note on spmd.declare_reduction for the planned explicit
    spmd.allreduce emission.
  }];

  let arguments = (ins
    Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
    OptionalAttr<SPMD_ScheduleKindAttr>  : $schedule_kind,
    Optional<AnyInteger>                 : $schedule_chunk,
    Variadic<AnyType>                    : $reduction_vars,
    OptionalAttr<ArrayAttr>              : $reduction_syms,
    I32Attr                              : $usedModel);

  let regions = (region SizedRegion<1> : $region);
}

// Mirrors omp.loop_nest
def SPMD_LoopNestOp : SPMD_Op<"loop_nest", [
    SingleBlock,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"YieldOp">,
    SameVariadicOperandSize
]> {
  let summary = "Concrete iteration space for a worksharing or parallel loop";
  let description = [{
    Mirrors `omp.loop_nest`. Each dimension contributes one lower bound,
    upper bound, and step operand. The region entry block carries one
    induction-variable block argument per dimension.

    The range is half-open unless `$loop_inclusive` is set, in which case the
    upper bound is part of it, as on `omp.loop_nest`.  C loops written
    `i <= n` reach the dialect that way, and dropping the attribute loses
    their last iteration.
  }];

  let arguments = (ins
    Variadic<AnyInteger> : $loop_lower_bounds,
    Variadic<AnyInteger> : $loop_upper_bounds,
    Variadic<AnyInteger> : $loop_steps,
    UnitAttr             : $loop_inclusive);

  let regions = (region SizedRegion<1> : $region);
}


// Mirrors omp.single
def SPMD_SingleOp : SPMD_Op<"single", [
    AutomaticAllocationScope,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TerminatorOp">,
    Collective
    // RuntimeReadT<>, RuntimeWriteT<> (cant be specified, when RecursiveMemoryEffects is that is more important here)
]> {
  let summary = "Structured block executed by exactly one rank in the team";
  let description = [{
    Mirrors `omp.single`, but with a trailing
    explicit barrier unless `$nowait` was set at omp origin.
  }];

  let arguments = (ins
    Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
    I32Attr                              : $usedModel);

  let regions = (region SizedRegion<1> : $region);
  // let results = (outs SPMD_ErrorType : $error);
}

// Mirrors omp.critical
def SPMD_CriticalOp : SPMD_Op<"critical", [
    AutomaticAllocationScope,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TerminatorOp">,
    // RuntimeReadT<>, RuntimeWriteT<> (cant be specified, when RecursiveMemoryEffects is that is more important here)
]> {
  let summary = "Mutually exclusive structured block";
  let description = [{
    Mirrors `omp.critical`. At most one rank executes the body at a time.
    The optional `$name` identifies the critical section; unnamed
    sections behave as though they share the same unnamed identifier.
  }];

  let arguments = (ins
    OptionalAttr<StrAttr> : $name,
    I32Attr               : $usedModel);

  let regions = (region SizedRegion<1> : $region);
  let results = (outs SPMD_ErrorType : $error);
}

// spmd.sections — worksharing construct distributing N bodies across the team.
// Followed by an explicit spmd.barrier unless $nowait was set at the OMP origin.
def SPMD_SectionsOp : SPMD_Op<"sections", [
    AutomaticAllocationScope,
    RecursiveMemoryEffects,
    SingleBlock,                              // one region, children are section ops
    NoTerminator,
    AttrSizedOperandSegments,
    GlobalConcurrency,
    Collective
]> {
  let summary = "Non-iterative worksharing: each section runs on exactly one rank";
  let description = [{
    Each `spmd.section` child is executed once by one rank in the team.
    Any two sections may execute simultaneously on different ranks —
    this concurrent-execution semantics is what distinguishes sections
    from spmd.single and is the key invariant for race analysis.

    A trailing explicit `spmd.barrier` is emitted by the conversion pass
    unless the original `omp.sections` carried `nowait`.

    $private_vars and $reduction_vars / $reduction_syms mirror the same
    fields on spmd.parallel.
  }];

  let arguments = (ins
    Arg<SPMD_CommType, "", [CommRead<>]> : $comm,
    Variadic<AnyType>                    : $private_vars,
    Variadic<AnyType>                    : $reduction_vars,
    OptionalAttr<ArrayAttr>              : $reduction_syms,
    I32Attr                              : $usedModel);

  let regions = (region SizedRegion<1> : $region);  // contains spmd.section ops
  let results = (outs SPMD_ErrorType : $error);
}

// spmd.section — one branch of a sections construct.
// Exactly one rank executes this body; concurrent with sibling sections.
def SPMD_SectionOp : SPMD_Op<"section", [
    HasParent<"SectionsOp">,
    AutomaticAllocationScope,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TerminatorOp">
]> {
  let summary = "One body within an spmd.sections construct";
  let description = [{
    Executed by exactly one rank. Sibling `spmd.section` ops within the
    same `spmd.sections` may execute simultaneously on distinct ranks.
    This is the fundamental concurrency edge for race detection.
  }];
  let regions = (region SizedRegion<1> : $region);
}

def SPMD_TaskOp : SPMD_Op<"task", [
    AutomaticAllocationScope,
    RecursiveMemoryEffects,
    AttrSizedOperandSegments,
    SingleBlockImplicitTerminator<"TerminatorOp">,
    GlobalConcurrency
]> {
  let summary = "Deferred async task body, executed by an arbitrary thread";
  let description = [{
    Models omp.task. The body is scheduled for deferred execution on an
    arbitrary thread from the enclosing parallel team. Temporal ordering
    relative to sibling tasks is determined solely by:
      - intervening spmd.taskwait ops (epoch boundary), or
      - $depend_vars edges forming a task DAG.

    If $if_expr evaluates to false, the task is "undeferred": the encountering
    thread executes the body immediately (semantically equivalent to removing
    the task construct). The analysis must treat this case as sequential.

    $untied: if true, the task may migrate to a different thread after a
    scheduling point. Conservative analysis ignores untied — always assumes
    arbitrary thread.
  }];

  let arguments = (ins
    // Dependency edges. Empty = unordered w.r.t. siblings in same epoch.
    // Preserved for future DAG-based precision; conservatively ignored now.
    Variadic<AnyType>             : $depend_vars,
    // if false → undeferred: treat body as immediately-executing sequential code
    Optional<I1>                  : $if_expr,
    // final: all tasks generated inside are also final (undeferred)
    Optional<I1>                  : $final_expr,
    BoolAttr                      : $untied,
    BoolAttr                      : $mergeable,
    I32Attr                       : $usedModel);

  let regions = (region SizedRegion<1> : $region);
}

def SPMD_TaskwaitOp : SPMD_Op<"taskwait"> {
  let summary = "Epoch boundary: all child tasks created before this point complete";
  let description = [{
    Models omp.taskwait. Acts as a hard epoch boundary for the task scheduling
    analysis: tasks created before this op belong to a different (earlier)
    epoch than tasks created after it and therefore cannot execute concurrently
    with them.

    $depend_vars: if present, this is a selective taskwait (OMP 5.0 depend
    clause) — only tasks that produced these outputs are awaited. The epoch
    analysis conservatively treats any taskwait as a full boundary regardless,
    which is sound but may produce false positives for selective taskwaits.
    Precise depend-graph analysis is deferred.

    NOT YET IMPLEMENTED: the epoch semantics described above are the intended
    contract, not current behaviour. No analysis under lib/spmd/Analysis/ reads
    this op, so today it imposes no ordering and spmd.task bodies are modelled
    as if they ran inline at their creation site. See the "Modelling
    limitations" block at the top of lib/spmd/passes/OMPToSPMD.cpp.
  }];

  let arguments = (ins
    Variadic<AnyType>  : $depend_vars,
    I32Attr            : $usedModel);

  let results = (outs SPMD_ErrorType : $error);
}


def SPMD_TaskgroupOp : SPMD_Op<"taskgroup", [
    AutomaticAllocationScope,
    RecursiveMemoryEffects,
    SingleBlockImplicitTerminator<"TerminatorOp">,
]> {
  let summary = "Scope with implicit taskwait at region end";
  let description = [{
    Models omp.taskgroup. All child tasks and their descendants generated
    within the region are part of the taskgroup set. There is an implicit
    task scheduling point at the end of the region: the enclosing task
    suspends until all tasks in the set complete.

    For epoch analysis: the region end is treated as a full spmd.taskwait —
    tasks created inside the region belong to epochs that are fully resolved
    before control leaves the taskgroup.

    $task_reduction_vars / $task_reduction_syms mirror omp.taskgroup's
    task_reduction clause; preserved for future reduction analysis.
  }];

  let arguments = (ins
    Variadic<AnyType>           : $task_reduction_vars,
    OptionalAttr<ArrayAttr>     : $task_reduction_syms,
    I32Attr                     : $usedModel);

  let regions = (region SizedRegion<1> : $region);
  let results = (outs SPMD_ErrorType : $error);
}

def SPMD_TaskloopOp : SPMD_Op<"taskloop", [
    AutomaticAllocationScope,
    RecursiveMemoryEffects,
    SingleBlock,
    NoTerminator,
    AttrSizedOperandSegments,
    GlobalConcurrency
]> {
  let summary = "Loop whose iterations execute as individual tasks";
  let description = [{
    Models omp.taskloop. Each iteration of the contained spmd.loop_nest
    becomes a separate task, dynamically scheduled. Multiple iterations
    may execute concurrently on different threads.

    For epoch/SERIALIZED analysis: treat each iteration as a task in the
    same epoch. If any iteration body contains a candidate communication op,
    and $grainsize / $num_tasks allow > 1 concurrent iteration (the default),
    that is a SERIALIZED violation.

    Unless $nogroup is true, there is an implicit spmd.taskgroup wrapping
    the entire taskloop — i.e., an epoch boundary at the loop end. When
    $nogroup is true the tasks remain live past the loop and must be awaited
    by an outer taskwait or taskgroup.

    Must contain exactly one spmd.loop_nest in its region (same constraint
    as spmd.wsloop).
  }];

  let arguments = (ins
    // Scheduling granularity: how many iterations per task.
    // Absent = runtime decides. Both absent = 1 iteration per task.
    Optional<I32>               : $grainsize,
    Optional<I32>               : $num_tasks,
    // Suppress implicit taskgroup at loop end.
    BoolAttr                    : $nogroup,
    BoolAttr                    : $untied,
    I32Attr                     : $usedModel);

  let regions = (region SizedRegion<1> : $region);
}

def SPMD_DeclareReductionOp : SPMD_Op<"declare_reduction", [
    IsolatedFromAbove,
    Symbol
]> {
  let summary = "Declares a reduction kind for use in SPMD parallel constructs";
  let description = [{
    Declares an SPMD reduction. Mirrors `omp.declare_reduction` structurally
    but adds an optional `$kind` attribute for well-known operators so analyses
    can fast-path without inspecting region bodies.

    Regions:
      1. `$allocRegion`           — optional; allocates thread-local storage.
      2. `$initializerRegion`     — initialises the thread-local copy with the
                                   neutral element (0 for add, 1 for mul, …).
      3. `$reductionRegion`       — combines two private values into one.
      4. `$atomicReductionRegion` — optional atomic variant of the combine step.
      5. `$cleanupRegion`         — optional; frees alloc/init-allocated memory.

    The `$kind` attribute is a *derived annotation*: set by ConvertDeclareReductionOp
    when the combiner matches a known single-op pattern (arith.addi → add, …).
    When absent the regions are the sole authoritative source.
    The verifier checks consistency between `$kind` and the combiner when both
    are present.

    `$byref_element_type` mirrors omp.declare_reduction: for by-ref reductions
    on allocatable/pointer variables the boxed element type is preserved here.

    Future work — explicit allreduce emission:
    Currently the reduction is kept implicit exactly as omp.parallel does:
    operator and variables are carried as $reduction_syms / $reduction_vars on
    the enclosing construct and the runtime performs the combine at the join.
    Once the IR is lowered to runtime calls the plan is:
      - Emit explicit `spmd.allreduce` before `spmd.parallel_yield`, referencing
        this symbol via a new optional $reduction_sym on spmd.allreduce.
      - Inline $initializerRegion into the parallel body for neutral-element init.
      - Map $atomicReductionRegion to the matching SPMD_OpType on spmd.allreduce,
        or inline it for thread-level regions where a true collective is not needed.
      - Use $reductionRegion as the merge kernel for models requiring explicit
        pairwise reduction (e.g. MPI tree-reduce with user-defined operators).
  }];

  let arguments = (ins
    SymbolNameAttr                        : $sym_name,
    TypeAttr                              : $type,
    OptionalAttr<TypeAttr>                : $byref_element_type,
    OptionalAttr<SPMD_ReductionKindAttr>  : $kind
  );

  let regions = (region
    MaxSizedRegion<1> : $allocRegion,
    AnyRegion         : $initializerRegion,
    AnyRegion         : $reductionRegion,
    AnyRegion         : $atomicReductionRegion,
    AnyRegion         : $cleanupRegion
  );

  let assemblyFormat = [{
    $sym_name `:` $type attr-dict-with-keyword
    ( `alloc`    $allocRegion^ )?
    `init`       $initializerRegion
    `combiner`   $reductionRegion
    ( `atomic`   $atomicReductionRegion^ )?
    ( `cleanup`  $cleanupRegion^ )?
  }];

  let hasVerifier = 1;
}

#endif // SPMD_OPS
