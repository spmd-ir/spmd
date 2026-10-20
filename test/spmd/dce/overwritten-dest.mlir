// RUN: spmd-opt %s -eliminate-dead-comm | FileCheck %s
//
// overwritten-dest: a data-moving collective whose destination is completely
// rewritten before anything reads it.  Every function here reads the buffer at
// the end, so the dead-dest rule does not fire and only the overwrite reasoning
// is exercised.

// Two identical broadcasts back to back: the first one is unobservable.
// CHECK-LABEL: func.func @overwritten_bcast
// CHECK-COUNT-1: spmd.bcast
// CHECK-NOT:     spmd.bcast
func.func @overwritten_bcast() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// A load between the two broadcasts observes the first one.
// CHECK-LABEL: func.func @read_between
// CHECK-COUNT-2: spmd.bcast
func.func @read_between() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %mid = memref.load %recv[%c0] : memref<1xi32>
  memref.store %mid, %send[%c0] : memref<1xi32>
  %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// The intervening read hides inside a nested region.
// CHECK-LABEL: func.func @read_between_nested
// CHECK-COUNT-2: spmd.bcast
func.func @read_between_nested(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  scf.if %cond {
    %mid = memref.load %recv[%c0] : memref<1xi32>
    memref.store %mid, %send[%c0] : memref<1xi32>
  }
  %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// An allreduce writes `count` elements on every rank, which covers the ranks
// the bcast writes on, so it kills the bcast even though the op kinds differ.
// CHECK-LABEL: func.func @bcast_then_allreduce
// CHECK-NOT:     spmd.bcast
// CHECK:         spmd.allreduce
func.func @bcast_then_allreduce() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// spmd.reduce only writes the destination on the root rank, so it cannot be
// used to argue that the bcast's data is gone everywhere.
// CHECK-LABEL: func.func @bcast_then_reduce
// CHECK:         spmd.bcast
// CHECK:         spmd.reduce
func.func @bcast_then_reduce() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.reduce"(%comm, %send, %recv, %count, %dt, %sum, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// Different element counts: the second broadcast may cover less memory.
// CHECK-LABEL: func.func @different_count
// CHECK-COUNT-2: spmd.bcast
func.func @different_count() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count4 = arith.constant 4 : i64
  %count1 = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<4xi32>
  %recv = memref.alloca() : memref<4xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count4, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, memref<4xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.bcast"(%comm, %send, %recv, %count1, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, memref<4xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<4xi32>
  return %v : i32
}

// The overwrite sits in a nested region, so it is not unconditional.
// CHECK-LABEL: func.func @overwrite_conditional
// CHECK-COUNT-2: spmd.bcast
func.func @overwrite_conditional(%cond: i1) -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  scf.if %cond {
    %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  }
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// A sub-range of the destination is not the destination.
// CHECK-LABEL: func.func @overwrite_subview
// CHECK-COUNT-2: spmd.bcast
func.func @overwrite_subview() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<4xi32>
  %recv = memref.alloca() : memref<4xi32>
  %tail = memref.subview %recv[1] [1] [1] : memref<4xi32> to memref<1xi32, strided<[1], offset: 1>>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, memref<4xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.bcast"(%comm, %send, %tail, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, memref<1xi32, strided<[1], offset: 1>>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<4xi32>
  return %v : i32
}

// A root decides which ranks write. reduce(root=1) does not write rank 0's
// buffer, so rank 0 still reads the first reduce's result.
// CHECK-LABEL: func.func @reduce_different_roots
// CHECK-COUNT-2: spmd.reduce
func.func @reduce_different_roots() -> i32 {
  %c0 = arith.constant 0 : index
  %root0 = arith.constant 0 : i32
  %root1 = arith.constant 1 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.reduce"(%comm, %send, %recv, %count, %dt, %sum, %root0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %e1 = "spmd.reduce"(%comm, %send, %recv, %count, %dt, %sum, %root1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// The same for gather.
// CHECK-LABEL: func.func @gather_different_roots
// CHECK-COUNT-2: spmd.gather
func.func @gather_different_roots() -> i32 {
  %c0 = arith.constant 0 : index
  %root0 = arith.constant 0 : i32
  %root1 = arith.constant 1 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<4xi32>
  %e0 = "spmd.gather"(%comm, %send, %count, %dt, %recv, %count, %dt, %root0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, memref<4xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.gather"(%comm, %send, %count, %dt, %recv, %count, %dt, %root1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, memref<4xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<4xi32>
  return %v : i32
}

// The same root picks the same ranks, so the first reduce goes.
// CHECK-LABEL: func.func @reduce_same_root
// CHECK-COUNT-1: spmd.reduce
// CHECK-NOT:     spmd.reduce
func.func @reduce_same_root() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.reduce"(%comm, %send, %recv, %count, %dt, %sum, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %e1 = "spmd.reduce"(%comm, %send, %recv, %count, %dt, %sum, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// An allreduce writes on every rank, the root of the reduce included.
// CHECK-LABEL: func.func @reduce_then_allreduce
// CHECK-NOT:     spmd.reduce
// CHECK:         spmd.allreduce
func.func @reduce_then_allreduce() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.reduce"(%comm, %send, %recv, %count, %dt, %sum, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %e1 = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// A root is a rank of one communicator; the second bcast may reach other
// ranks, or other processes, than the first.
// CHECK-LABEL: func.func @different_comms
// CHECK-COUNT-2: spmd.bcast
func.func @different_comms(%other: !spmd.comm) -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.bcast"(%other, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// Nor are two programming models' broadcasts the same operation.
// CHECK-LABEL: func.func @different_models
// CHECK-COUNT-2: spmd.bcast
func.func @different_models() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// allgather's count is per rank: it writes count * size elements, more than
// an allreduce with the same count.
// CHECK-LABEL: func.func @allgather_then_allreduce
// CHECK:         spmd.allgather
// CHECK:         spmd.allreduce
func.func @allgather_then_allreduce() -> i32 {
  %c0 = arith.constant 0 : index
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<4xi32>
  %e0 = "spmd.allgather"(%comm, %send, %count, %dt, %recv, %count, %dt) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, memref<4xi32>, i64, !spmd.datatype) -> !spmd.error
  %e1 = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<4xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<4xi32>
  return %v : i32
}

// A non-blocking send of the destination may read it at any point up to its
// wait, which here is after both broadcasts: its peer may receive the first
// one's result.
// CHECK-LABEL: func.func @pending_send
// CHECK-COUNT-2: spmd.bcast
func.func @pending_send() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %es = "spmd.send"(%comm, %recv, %count, %dt, %root, %root) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %w = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// A window exposes the destination: another rank may get it between the two
// broadcasts without any op here showing it. The put keeps the window itself
// alive; a window no data moves through is removed as a dead resource first.
// CHECK-LABEL: func.func @window_exposes_destination
// CHECK-COUNT-2: spmd.bcast
func.func @window_exposes_destination() -> i32 {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %disp = arith.constant 4 : i32
  %size = arith.constant 4 : i64
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %win, %ew = "spmd.winCreate"(%comm, %recv, %size, %disp) <{usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, i32) -> (!spmd.win, !spmd.error)
  %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %src = memref.alloca() : memref<1xi32>
  %stride = arith.constant 1 : i64
  %zero = arith.constant 0 : i64
  %ep = "spmd.put"(%src, %count, %dt, %stride, %root, %win, %zero, %count, %dt, %stride) <{isBlocking = true, isAtomic = false, usedModel = 0 : i32}> : (memref<1xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  %ef = "spmd.winFree"(%win) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  return %v : i32
}

// Inside a thread team another thread runs the same region concurrently, and
// may load the buffer while this one is between the two broadcasts.
// CHECK-LABEL: func.func @thread_team
// CHECK-COUNT-2: spmd.bcast
func.func @thread_team() {
  %c0 = arith.constant 0 : index
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  "spmd.parallel"(%comm) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    %e0 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
    %e1 = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
    %v = memref.load %recv[%c0] : memref<1xi32>
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  return
}
