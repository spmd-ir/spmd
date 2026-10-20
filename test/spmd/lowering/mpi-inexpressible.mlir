// RUN: spmd-opt %s -split-input-file -lower-spmd-to-mpi -verify-diagnostics
//
// Operands the dialect has and MPI has no parameter for.  The lowering used to
// drop each of them, so the MPI call did something other than the op said.
// Every one is reported before anything is rewritten, each with its reason.

func.func @strides(%out: !llvm.ptr, %in: !llvm.ptr, %stride: i64, %win: !spmd.win) {
  %one = arith.constant 1 : i64
  %two = arith.constant 2 : i64
  %count = arith.constant 4 : i64
  %rank = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %disp = arith.constant 0 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.sendrecv' op has a receive stride other than 1; MPI transfers contiguous elements only}}
  %e0 = "spmd.sendrecv"(%comm, %out, %count, %dt, %one, %rank, %tag, %in, %count, %dt, %two, %rank, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  // A stride that is not a constant is not known to be 1 either.
  // expected-error @+1 {{'spmd.put' op has an origin stride other than 1}}
  %e1 = "spmd.put"(%out, %count, %dt, %stride, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}

// -----

func.func @stream(%buf: !llvm.ptr, %stream: !spmd.stream) {
  %count = arith.constant 4 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.send' op carries a stream, which MPI has no parameter for}}
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag, %stream) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  return
}

// -----

func.func @comm_config(%color: i32, %key: i32, %config: !spmd.commConfig) {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.commSplit' op carries a communicator config, which MPI_Comm_split has no parameter for}}
  %split, %e = "spmd.commSplit"(%comm, %color, %key, %config) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32, !spmd.commConfig) -> (!spmd.comm, !spmd.error)
  return
}

// -----

func.func @alignment(%size: i64, %align: i64) {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.malloc' op asks for an alignment, which MPI_Alloc_mem has no parameter for}}
  %mem, %e = "spmd.malloc"(%comm, %size, %align) <{usedModel = 0 : i32}> : (!spmd.comm, i64, i64) -> (!llvm.ptr, !spmd.error)
  return
}

// -----

func.func @bcast_buffers(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr) {
  %count = arith.constant 4 : i64
  %root = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.bcast' op has distinct send and receive buffers; MPI_Bcast has one}}
  %e = "spmd.bcast"(%comm, %sbuf, %rbuf, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  return
}

// -----

func.func @windows(%base: !llvm.ptr, %unit: i32, %assert: i32) {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{'spmd.winCreate' op has no size; MPI_Win_create needs one}}
  %win, %e0 = "spmd.winCreate"(%comm, %base, %unit) <{usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  // expected-error @+1 {{'spmd.fence' op has no window; MPI needs one}}
  %e1 = "spmd.fence"(%assert) <{usedModel = 0 : i32}> : (i32) -> !spmd.error
  // expected-error @+1 {{'spmd.flushAll' op has no window; MPI needs one}}
  %e2 = "spmd.flushAll"() <{isLocal = false, usedModel = 0 : i32}> : () -> !spmd.error
  return
}

// -----

// The same operands on an op of another model are that model's business.
func.func @other_model(%buf: !llvm.ptr, %stream: !spmd.stream) {
  %count = arith.constant 4 : i64
  %dest = arith.constant 1 : i32
  %tag = arith.constant 0 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 3 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 3 : i32}> : () -> !spmd.comm
  %e = "spmd.send"(%comm, %buf, %count, %dt, %dest, %tag, %stream) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 3 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  return
}
