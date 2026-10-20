// RUN: spmd-opt %s -coalesce-comm | FileCheck %s
//
// The envelope: everything two messages must share before their payloads are
// even looked at.  Each is compared by SSA identity, and each function here
// differs in exactly one field, so a regression that loosens one comparison
// shows up as exactly one failure.
//
// Every function uses its own tag range, because the legality check is
// module-wide - see the header of contiguous.mlir.  No function here uses a
// wildcard or a non-constant tag; those poison a whole communicator and live in
// wildcards.mlir.

// Two neighbours are two envelopes.  This is the miniWeather shape - a halo
// exchange posts to the left and to the right - and it is the honest negative:
// the most common multi-message pattern in real code is *not* coalescable,
// because the messages go to different places.
// CHECK-LABEL: func.func @different_peers
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @different_peers(%left : i32, %right : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %left, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %right, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A duplicated communicator is a different matching context.
// CHECK-LABEL: func.func @different_comms
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @different_comms(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dup, %derr = "spmd.commDup"(%comm) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%dup, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Two messages with one tag: MPI's non-overtaking guarantee makes their order
// observable to the receiver, so the tag cannot be the shared ordering key.
// CHECK-LABEL: func.func @duplicate_tags
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @duplicate_tags(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t = arith.constant 20 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Two spmd.constDatatype operations that happen to denote the same type are
// still two values, and the comparison is by SSA identity.  This is why the
// pass wants -cse to have run first: after CSE these become
// one value and the pair merges.
// CHECK-LABEL: func.func @unmerged_datatype_constants
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @unmerged_datatype_constants(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 30 : i32
  %t1 = arith.constant 31 : i32
  %n = arith.constant 4 : i64
  %dtA = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %dtB = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dtA, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dtB, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Genuinely different element types cannot share one count.  Merging them would
// need a byte-typed message, which the MPI lowering has no way to spell.
// CHECK-LABEL: func.func @different_datatypes
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @different_datatypes(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 40 : i32
  %t1 = arith.constant 41 : i32
  %n = arith.constant 4 : i64
  %i32t = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %f64t = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<16 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %i32t, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %f64t, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Buffered and standard mode are different calls on the wire.
// CHECK-LABEL: func.func @different_protocol
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @different_protocol(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 50 : i32
  %t1 = arith.constant 51 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = true, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Opposite directions never share a group; a send may not become part of a
// receive.  Both survive on their own.
// CHECK-LABEL: func.func @mixed_send_recv
// CHECK:         spmd.send
// CHECK:         spmd.recv
func.func @mixed_send_recv(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 60 : i32
  %t1 = arith.constant 61 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.recv"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Stream-ordered messages complete relative to a stream rather than to program
// order, so the local instruction order says nothing about when the buffer is
// read.  Same bail-out the other communication passes take.
// CHECK-LABEL: func.func @stream_operand
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @stream_operand(%peer : i32, %stream : !spmd.stream) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 70 : i32
  %t1 = arith.constant 71 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0, %stream) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1, %stream) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32, !spmd.stream) -> !spmd.error
  return
}

// Tag matching is MPI's.  The other models carry a tag operand without those
// semantics, so their messages are left alone.
// CHECK-LABEL: func.func @non_mpi_model
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @non_mpi_model(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 80 : i32
  %t1 = arith.constant 81 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Non-blocking initiations need the request array rebuilt and the waitAll count
// rewritten; that is the next increment, not this one.
// CHECK-LABEL: func.func @nonblocking
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @nonblocking(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 90 : i32
  %t1 = arith.constant 91 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %r0, %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %r1, %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = false, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %w0 = "spmd.wait"(%r0) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %w1 = "spmd.wait"(%r1) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// An observed error code is an observation of the individual message, so the
// message cannot be folded into another one.
// CHECK-LABEL: func.func @observed_error
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @observed_error(%peer : i32) -> i32 {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 100 : i32
  %t1 = arith.constant 101 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %code = "spmd.castSPMDErrorToI32"(%e0) : (!spmd.error) -> i32
  return %code : i32
}

// The control.  Same shape as the negatives above with nothing wrong with it,
// so the file cannot pass merely because the pass did nothing.
// CHECK-LABEL: func.func @control_merges
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @control_merges(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 110 : i32
  %t1 = arith.constant 111 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
