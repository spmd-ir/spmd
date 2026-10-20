// RUN: spmd-opt %s -eliminate-dead-comm | FileCheck %s
//
// dead-resource: RMA windows and symmetric allocations that no data movement
// ever touches, removed together with their whole lifecycle.

// Nothing is ever put into or fetched from the window, so the window, both
// fences and the free are all dead.
// CHECK-LABEL: func.func @dead_window
// CHECK-NOT:     spmd.winCreate
// CHECK-NOT:     spmd.fence
// CHECK-NOT:     spmd.winFree
func.func @dead_window() {
  %assert = arith.constant 0 : i32
  %disp = arith.constant 4 : i32
  %size = arith.constant 16 : i64
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<4xi32>
  %win, %e0 = "spmd.winCreate"(%comm, %buf, %size, %disp) <{usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, i64, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.fence"(%win, %assert) <{usedModel = 0 : i32}> : (!spmd.win, i32) -> !spmd.error
  %e2 = "spmd.fence"(%win, %assert) <{usedModel = 0 : i32}> : (!spmd.win, i32) -> !spmd.error
  %e3 = "spmd.winFree"(%win) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// A put targets the window: everything stays.
// CHECK-LABEL: func.func @live_window
// CHECK:         spmd.winCreate
// CHECK-COUNT-2: spmd.fence
// CHECK:         spmd.winFree
func.func @live_window() {
  %assert = arith.constant 0 : i32
  %disp = arith.constant 4 : i32
  %size = arith.constant 16 : i64
  %rank = arith.constant 1 : i32
  %count = arith.constant 4 : i64
  %stride = arith.constant 1 : i64
  %zero = arith.constant 0 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<4xi32>
  %src = memref.alloca() : memref<4xi32>
  %win, %e0 = "spmd.winCreate"(%comm, %buf, %size, %disp) <{usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, i64, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.fence"(%win, %assert) <{usedModel = 0 : i32}> : (!spmd.win, i32) -> !spmd.error
  %e2 = "spmd.put"(%src, %count, %dt, %stride, %rank, %win, %zero, %count, %dt, %stride) <{isBlocking = true, isAtomic = false, usedModel = 0 : i32}> : (memref<4xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e3 = "spmd.fence"(%win, %assert) <{usedModel = 0 : i32}> : (!spmd.win, i32) -> !spmd.error
  %e4 = "spmd.winFree"(%win) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// A symmetric allocation that only ever reaches spmd.free.
// CHECK-LABEL: func.func @dead_malloc
// CHECK-NOT:     spmd.malloc
// CHECK-NOT:     spmd.free
func.func @dead_malloc() {
  %size = arith.constant 16 : i64
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %addr, %e0 = "spmd.malloc"(%comm, %size) <{usedModel = 0 : i32}> : (!spmd.comm, i64) -> (memref<?xi8>, !spmd.error)
  %e1 = "spmd.free"(%comm, %addr) <{usedModel = 0 : i32}> : (!spmd.comm, memref<?xi8>) -> !spmd.error
  return
}

// The symmetric buffer is written locally, so a remote rank may read it.
// CHECK-LABEL: func.func @live_malloc
// CHECK:         spmd.malloc
// CHECK:         spmd.free
func.func @live_malloc() {
  %c0 = arith.constant 0 : index
  %byte = arith.constant 7 : i8
  %size = arith.constant 16 : i64
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %addr, %e0 = "spmd.malloc"(%comm, %size) <{usedModel = 0 : i32}> : (!spmd.comm, i64) -> (memref<?xi8>, !spmd.error)
  memref.store %byte, %addr[%c0] : memref<?xi8>
  %e1 = "spmd.free"(%comm, %addr) <{usedModel = 0 : i32}> : (!spmd.comm, memref<?xi8>) -> !spmd.error
  return
}

// Removing the dead collective leaves the window unused, which the next
// iteration of the pass then collects: the cascade the max-iterations loop is
// there for.
// CHECK-LABEL: func.func @cascade
// CHECK-NOT:     spmd.allreduce
// CHECK-NOT:     spmd.winCreate
// CHECK-NOT:     spmd.winFree
func.func @cascade() {
  %assert = arith.constant 0 : i32
  %disp = arith.constant 4 : i32
  %size = arith.constant 16 : i64
  %count = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<4xi32>
  %recv = memref.alloca() : memref<4xi32>
  %win, %e0 = "spmd.winCreate"(%comm, %recv, %size, %disp) <{usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, i64, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, memref<4xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e2 = "spmd.winFree"(%win) <{usedModel = 0 : i32}> : (!spmd.win) -> !spmd.error
  return
}
