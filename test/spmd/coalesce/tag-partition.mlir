// RUN: spmd-opt %s -coalesce-comm=allow-staging=false | FileCheck %s
//
// A merge decision is a partition of the tag universe, and it is only legal
// when every operation in the module using one of a block's tags belongs to a
// group realising exactly that block.
//
// This is the file where functions are *supposed* to interfere with each other,
// so unlike the others it reuses tags across functions on purpose.  Each pair of
// functions is one scenario.

//===----------------------------------------------------------------------===//
// A tag that escapes the block
//===----------------------------------------------------------------------===//

// @escape_group would merge on its own.  It must not, because tag 1 is also
// used by a message elsewhere that is not part of any {0,1} group: that message
// expects 4 elements and would be handed 8.
// CHECK-LABEL: func.func @escape_group
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @escape_group(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @escape_lone_user
// CHECK:         spmd.send
func.func @escape_lone_user(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

//===----------------------------------------------------------------------===//
// The maximality counterexample
//===----------------------------------------------------------------------===//

// Both sides here are *maximal* for their own program text, and they still
// disagree: the sender can group {10,11}, the receiver cannot because a store to
// its first buffer sits between the two receives.  Per-side maximality is
// therefore not enough, and only the module-wide check catches it.  Merging the
// sender alone would truncate at the receiver.
// CHECK-LABEL: func.func @maximal_sender
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @maximal_sender(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @blocked_receiver
// CHECK-COUNT-2: spmd.recv
// CHECK-NOT:     spmd.recv
func.func @blocked_receiver(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %c42 = arith.constant 42 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.recv"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  llvm.store %c42, %buf : i32, !llvm.ptr
  %e1 = "spmd.recv"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

//===----------------------------------------------------------------------===//
// Overlapping but unequal blocks
//===----------------------------------------------------------------------===//

// {20,21} against {21,22}: both groups are internally fine, they share tag 21,
// and they do not agree on what the merged message contains.  Neither survives.
// CHECK-LABEL: func.func @overlap_low
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @overlap_low(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 20 : i32
  %t1 = arith.constant 21 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @overlap_high
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @overlap_high(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t1 = arith.constant 21 : i32
  %t2 = arith.constant 22 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e1 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e2 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t2) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

//===----------------------------------------------------------------------===//
// A tag held by an operation that can never be grouped
//===----------------------------------------------------------------------===//

// spmd.sendrecv is two envelopes in one operation and can never be a member, but
// its tags are still spoken for.  Tag 30 belongs to it, so the {30,31} block is
// not realisable.
// CHECK-LABEL: func.func @sendrecv_holds_a_tag
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @sendrecv_holds_a_tag(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 30 : i32
  %t1 = arith.constant 31 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @the_sendrecv
// CHECK:         spmd.sendrecv
func.func @the_sendrecv(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i64
  %t0 = arith.constant 30 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %out = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %in = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e = "spmd.sendrecv"(%comm, %out, %n, %dt, %c0, %peer, %t0, %in, %n, %dt, %c0, %peer, %t0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  return
}

//===----------------------------------------------------------------------===//
// Agreement, and the control
//===----------------------------------------------------------------------===//

// Both endpoints realise {40,41}, so both merge.  This is the shape a real halo
// exchange has, and it is what makes the negatives above meaningful.
// CHECK-LABEL: func.func @agreeing_sender
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @agreeing_sender(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 40 : i32
  %t1 = arith.constant 41 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @agreeing_receiver
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.recv"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.recv
func.func @agreeing_receiver(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 40 : i32
  %t1 = arith.constant 41 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.recv"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.recv"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
