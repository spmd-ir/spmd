// RUN: spmd-opt %s -split-input-file -lower-spmd-to-mpi | FileCheck %s
//
// Which of a call's two forms the lowering picks follows from the operation's
// results, not from $isBlocking alone: the non-blocking form is the one that
// hands out a request, and it can only be used by an operation that has a
// result to put it in.
//
// The two disagree for the one-sided transfers.  -convert-apiCalls-to-spmd
// reads $isBlocking off USIS's execution.blocking, which is false for MPI_Put
// as well as for MPI_Rput -- MPI calls both non-blocking, and only MPI_Rput
// hands out a request -- so a raised MPI_Put carries isBlocking = false and no
// request result.  Reading $isBlocking on its own sent it to MPI_Rput, whose
// request there was nowhere to put, and the operation then lowered to nothing
// at all.

// CHECK-LABEL: func.func @put_without_request
// CHECK:         llvm.call @MPI_Put({{.*}}) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32) -> i32
func.func @put_without_request(%origin: !llvm.ptr, %rank: i32, %disp: i64, %win: !spmd.win) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %count = arith.constant 1 : i64
  %one = arith.constant 1 : i64
  %e = "spmd.put"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}

// -----

// The same operation with a request result is MPI_Rput, as before.
// CHECK-LABEL: func.func @put_with_request
// CHECK:         llvm.call @MPI_Rput({{.*}}) : (!llvm.ptr, i32, i32, i32, i64, i32, i32, i32, !llvm.ptr) -> i32
func.func @put_with_request(%origin: !llvm.ptr, %rank: i32, %disp: i64, %win: !spmd.win) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %count = arith.constant 1 : i64
  %one = arith.constant 1 : i64
  %req, %e = "spmd.put"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = false, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> (!spmd.req, !spmd.error)
  return
}

// -----

// Where $isBlocking and the results agree, it still decides: a blocking send
// that carries a request result stays MPI_Send, and the request it hands back
// is MPI_REQUEST_NULL -- the non-blocking form is not picked just because
// there is a result for its request.
// CHECK-LABEL: func.func @blocking_send_with_request
// CHECK:         llvm.call @MPI_Send(
// CHECK:         llvm.mlir.constant(738197504 : i32)
func.func @blocking_send_with_request(%buf: !llvm.ptr) {
  %count = arith.constant 1 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %req, %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  return
}
