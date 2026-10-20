// RUN: spmd-opt %s -coalesce-comm=allow-staging=false | FileCheck %s
//
// Wildcards and unreadable tags poison a whole communicator.
//
// The reasoning is partition-based: the pass has to know which envelopes exist
// before it may claim two of them can be fused.  A receive that will match
// *whatever arrives* denies it that knowledge - it could match either half of a
// merged message, or the merged message itself - so nothing on that
// communicator may be merged, not just the tag in question.
//
// MPI_ANY_TAG is -1 and MPI_ANY_SOURCE is -2 in the shim headers
// (includesProgramModels/mpi/include/mpi.h:503-505), so a negative constant is
// a complete wildcard test here.
//
// Each function carries its own communicator so the scenarios do not mask one
// another, and each pairs its wildcard with a pair that would otherwise merge.

// A wildcard tag anywhere on the communicator stops all merging on it - note
// the sends below are perfectly mergeable on their own.
// CHECK-LABEL: func.func @wildcard_tag
// CHECK:         spmd.recv
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @wildcard_tag(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %any = arith.constant -1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %in = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %r = "spmd.recv"(%comm, %in, %n, %dt, %peer, %any) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A wildcard source is the same problem seen from the other axis.
// CHECK-LABEL: func.func @wildcard_source
// CHECK:         spmd.recv
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @wildcard_source(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %anysrc = arith.constant -2 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %t2 = arith.constant 12 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %comm, %derr = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %in = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %r = "spmd.recv"(%comm, %in, %n, %dt, %anysrc, %t0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t2) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A tag that is not a compile-time constant cannot be placed in any block, so
// the partition over this communicator is unknown.
// CHECK-LABEL: func.func @non_constant_tag
// CHECK-COUNT-3: spmd.send
// CHECK-NOT:     spmd.send
func.func @non_constant_tag(%peer : i32, %dyntag : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 20 : i32
  %t1 = arith.constant 21 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %comm, %derr = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %other = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %ed = "spmd.send"(%comm, %other, %n, %dt, %peer, %dyntag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The control, and the point of the whole file: poisoning is scoped to one
// communicator.  The same shape on a communicator nobody used a wildcard on
// still merges, so the negatives above are not just "the pass did nothing".
// CHECK-LABEL: func.func @control_on_node_comm
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @control_on_node_comm(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 30 : i32
  %t1 = arith.constant 31 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commNode"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
