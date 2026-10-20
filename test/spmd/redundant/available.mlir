// RUN: spmd-opt %s -split-input-file -eliminate-redundant-comm | FileCheck %s
//
// What makes a collective's result still available at a later point, and
// what takes it away.  Killing is unconditional; the uniformity side of the
// GEN rule lives in uniformity.mlir.
//
// Each case is a program of its own, with its body in main's process-level
// spmd.parallel: a fact is only established by an op every rank executes, and
// that is only known there.

// @identical: nothing touches either buffer in between.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @send_buffer_written: a store to the send buffer changes what the reduction
// would produce.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    memref.store %c42, %send[%c0] : memref<1xi32>
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @recv_buffer_written: a store to the destination overwrites the result that
// made the second one unnecessary.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    memref.store %c42, %recv[%c0] : memref<1xi32>
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @disjoint_buffer_written: a store to an unrelated allocation does not.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %other = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    memref.store %c42, %other[%c0] : memref<1xi32>
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @maybe_aliasing_write: a store through a pointer of unknown provenance may
// land in either buffer.  Both pointers are loaded from the same slot.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c1 = arith.constant 1 : i32
    %c42 = arith.constant 42 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
    %unknown = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %alias = llvm.load %slot : !llvm.ptr -> !llvm.ptr
    %e0 = "spmd.allreduce"(%world, %send, %unknown, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    llvm.store %c42, %alias : i32, !llvm.ptr
    %e1 = "spmd.allreduce"(%world, %send, %unknown, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = llvm.load %unknown : !llvm.ptr -> i32
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @opaque_call_between: an opaque call may do anything to anything.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func private @opaque()
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    func.call @opaque() : () -> ()
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @different_root: different roots are different collectives.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.bcast
// CHECK-NOT:     spmd.bcast
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %root0 = arith.constant 0 : i32
    %root1 = arith.constant 1 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.bcast"(%world, %send, %recv, %count, %dt, %root0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
    %e1 = "spmd.bcast"(%world, %send, %recv, %count, %dt, %root1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @different_reduce_op: so are different reduction operators, even over the
// same buffers.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %max = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %max) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @observed_error: the second collective's error code is inspected, so it is
// observable and stays regardless of availability.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %code = "spmd.castSPMDErrorToI32"(%e1) : (!spmd.error) -> i32
    "spmd.parallel_yield"(%code) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @non_blocking_untouched: a non-blocking collective's write lands at its
// completion, not at the operation, so it neither publishes a fact nor
// consumes one.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.bcast
// CHECK-NOT:     spmd.bcast
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %root = arith.constant 0 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %r0, %e0 = "spmd.bcast"(%world, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
    %w0 = "spmd.wait"(%r0) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    %r1, %e1 = "spmd.bcast"(%world, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
    %w1 = "spmd.wait"(%r1) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @p2p_untouched: P2P is out of scope entirely: nothing about a peer's buffer
// is visible here.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.recv
// CHECK-NOT:     spmd.recv
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %src = arith.constant 0 : i32
    %tag = arith.constant 0 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.recv"(%world, %recv, %count, %dt, %src, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
    %e1 = "spmd.recv"(%world, %recv, %count, %dt, %src, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @cascade: three in a row: the cascade is picked up in one pass run.
// CHECK-LABEL: func.func @main
// CHECK:         spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e2 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @pending_receive: a receive into the destination is still pending at the
// first allreduce and completes before the second, so the second allreduce
// overwrites what the receive delivered.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %src = arith.constant 0 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %req, %er = "spmd.recv"(%world, %recv, %count, %dt, %src, %src) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %w = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @window_exposes_send_buffer: another rank may put into the send buffer
// between the two allreduces without any op here showing it.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %disp = arith.constant 4 : i32
    %size = arith.constant 4 : i64
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %win, %ew = "spmd.winCreate"(%world, %send, %size, %disp) <{usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, i32) -> (!spmd.win, !spmd.error)
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    %ef = "spmd.winFree"(%win) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @thread_team_writes: a thread team writes the destination between the two
// allreduces. The dataflow framework does not flow a spmd.parallel's region
// out to the point after it, so that write is only seen by forgetting
// everything across the region.
// CHECK-LABEL: func.func @main
// CHECK-COUNT-2: spmd.allreduce
// CHECK-NOT:     spmd.allreduce
func.func @main() -> i32 {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %r = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %c0 = arith.constant 0 : index
    %c42 = arith.constant 42 : i32
    %count = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
    %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
    %send = memref.alloca() : memref<1xi32>
    %recv = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    "spmd.parallel"(%world) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
      memref.store %c42, %recv[%c0] : memref<1xi32>
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm) -> ()
    %e1 = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}
