// RUN: spmd-opt %s -eliminate-dead-comm | FileCheck %s
//
// dead-dest: a data-moving collective whose destination buffer nothing ever
// reads.

// The reduced value is never observed -> the allreduce goes away.
// CHECK-LABEL: func.func @dead_allreduce
// CHECK-NOT:     spmd.allreduce
func.func @dead_allreduce() {
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %err = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// The reduced value is loaded afterwards -> the allreduce stays.
// CHECK-LABEL: func.func @live_allreduce
// CHECK:         spmd.allreduce
func.func @live_allreduce() -> i32 {
  %c0 = arith.constant 0 : index
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %err = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %v = memref.load %recv[%c0] : memref<1xi32>
  return %v : i32
}

// The load sits in a nested region, which the reader search must still see.
// CHECK-LABEL: func.func @live_allreduce_nested_read
// CHECK:         spmd.allreduce
func.func @live_allreduce_nested_read(%cond: i1) {
  %c0 = arith.constant 0 : index
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %err = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  scf.if %cond {
    %v = memref.load %recv[%c0] : memref<1xi32>
    memref.store %v, %send[%c0] : memref<1xi32>
  }
  return
}

// The error code is inspected, so the op has an observed result.
// CHECK-LABEL: func.func @observed_error
// CHECK:         spmd.allreduce
func.func @observed_error() -> i32 {
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %err = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%err) : (!spmd.error) -> i32
  return %code : i32
}

// The destination escapes into an opaque call -> assume it is read.
// CHECK-LABEL: func.func @escaping_dest
// CHECK:         spmd.allreduce
func.func private @sink(memref<1xi32>)
func.func @escaping_dest() {
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %err = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  func.call @sink(%recv) : (memref<1xi32>) -> ()
  return
}

// The destination is not a private allocation: another function may read it.
// CHECK-LABEL: func.func @dest_is_argument
// CHECK:         spmd.allreduce
func.func @dest_is_argument(%recv: memref<1xi32>) {
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %err = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// An in-place allreduce only reads the buffer through the op being removed.
// CHECK-LABEL: func.func @dead_inplace_allreduce
// CHECK-NOT:     spmd.allreduce
func.func @dead_inplace_allreduce() {
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<1xi32>
  %err = "spmd.allreduce"(%comm, %buf, %buf, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// A non-blocking collective is removed together with its unique completion.
// CHECK-LABEL: func.func @dead_ibcast
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.wait
func.func @dead_ibcast() {
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  %werr = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// The request is stashed in an array for a later spmd.waitAll.  That array
// holds nothing else, so the whole matched set - initiation, store, waitAll -
// is unique and goes together.  test/spmd/dce/comm-handles.mlir covers the
// cases where it is not.
// CHECK-LABEL: func.func @dead_ibcast_waitall
// CHECK-NOT:     spmd.bcast
// CHECK-NOT:     spmd.waitAll
// CHECK-NOT:     memref.store
func.func @dead_ibcast_waitall() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  %reqs = memref.alloca() : memref<1x!spmd.req>
  %req, %err = "spmd.bcast"(%comm, %send, %recv, %count, %dt, %root) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  memref.store %req, %reqs[%c0] : memref<1x!spmd.req>
  %werr = "spmd.waitAll"(%c1, %reqs) <{usedModel = 0 : i32}> : (i32, memref<1x!spmd.req>) -> !spmd.error
  return
}

// The shape the real pipeline produces: an llvm.alloca handed to the collective
// as a bare !llvm.ptr, bracketed by clang's lifetime markers.  Those markers
// annotate the allocation without observing it, so they must not count as uses.
// CHECK-LABEL: func.func @dead_allreduce_llvm_ptr
// CHECK-NOT:     spmd.allreduce
func.func @dead_allreduce_llvm_ptr() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : i32
  %c42 = arith.constant 42 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %send = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %recv = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start %send : !llvm.ptr
  llvm.intr.lifetime.start %recv : !llvm.ptr
  %view = "enzymexla.pointer2memref"(%send) : (!llvm.ptr) -> memref<?xi32>
  memref.store %c42, %view[%c0] : memref<?xi32>
  %err = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  llvm.intr.lifetime.end %recv : !llvm.ptr
  llvm.intr.lifetime.end %send : !llvm.ptr
  return
}

// A dead collective under rank-dependent control is still removed: the deletion
// is textual, so every rank loses it and no mismatch can appear.  Note this can
// remove exactly the collective that -check-collectives is meant to flag, which
// is why the check passes belong on the un-optimised IR.
// CHECK-LABEL: func.func @rank_conditional_dead_bcast
// CHECK-NOT:     spmd.bcast
func.func @rank_conditional_dead_bcast() {
  %c0 = arith.constant 0 : index
  %c42 = arith.constant 42 : i32
  %root = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %rank, %rerr = "spmd.getRankInComm"(%comm) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %data = memref.alloca() : memref<1xi32>
  %isRoot = arith.cmpi eq, %rank, %root : i32
  scf.if %isRoot {
    memref.store %c42, %data[%c0] : memref<1xi32>
    %err = "spmd.bcast"(%comm, %data, %data, %count, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  }
  return
}

// P2P is out of scope: the peer's matching operation would be left dangling.
// CHECK-LABEL: func.func @p2p_untouched
// CHECK:         spmd.recv
func.func @p2p_untouched() {
  %src = arith.constant 0 : i32
  %tag = arith.constant 0 : i32
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %recv = memref.alloca() : memref<1xi32>
  %err = "spmd.recv"(%comm, %recv, %count, %dt, %src, %tag) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
