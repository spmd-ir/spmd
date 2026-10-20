// RUN: spmd-opt %s -split-input-file -eliminate-redundant-comm | FileCheck %s
//
// The trap redundant-collective elimination has to survive.
//
// A collective is idempotent only because it reads one buffer and writes
// another.  When the two are the same allocation - the MPI_IN_PLACE shape -
// repeating it reduces the already-reduced value.  At two ranks that can even
// look plausible; at scale it is silently wrong.  That is why
// `sendAlloc == recvAlloc` never enters the available set.
//
// Each case is a program of its own, with its body in main's process-level
// spmd.parallel: a fact is only established by an op every rank executes, and
// that is only known there.

// @inplace_allreduce
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
    %buf = memref.alloca() : memref<1xi32>
    %e0 = "spmd.allreduce"(%world, %buf, %buf, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %buf, %buf, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %buf[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @inplace_through_cast: same, reached through a cast rather than named
// directly: the aliasing is what matters, not the SSA value.
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
    %buf = memref.alloca() : memref<1xi32>
    %view = memref.cast %buf : memref<1xi32> to memref<?xi32>
    %e0 = "spmd.allreduce"(%world, %view, %buf, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<?xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %e1 = "spmd.allreduce"(%world, %view, %buf, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<?xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
    %v = memref.load %buf[%c0] : memref<1xi32>
    "spmd.parallel_yield"(%v) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  return %r : i32
}

// -----

// @not_inplace_control: a reduction that is *not* in place stays idempotent,
// and the second one goes.  This is the control: without it the two cases
// above would also pass if the pass simply did nothing.
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
