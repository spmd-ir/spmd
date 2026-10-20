// RUN: spmd-opt %s -coalesce-comm=allow-staging=false | FileCheck %s
//
// What may sit between two members, and what the intra-group order is.
//
// Merging moves the earlier members to the position of the last one.  That is
// safe only because no communication operation may lie between them, so the
// program's order of communication is unchanged; and because nothing between
// them touches a member's buffer, so the bytes are the same ones.
//
// Every function uses its own tag range - see the header of contiguous.mlir.

// The case that decides the intra-group order.  These sends appear textually as
// tag 1 then tag 0, which is the miniWeather shape: a halo exchange posts its
// receives in one tag order and its sends in the other.  Keying the layout on
// textual position would have one endpoint pack [t1|t0] while the other unpacks
// [t0|t1] - a published, real-world program miscompiled by the obvious rule.
// Ascending tag is a key both endpoints compute without talking.
// CHECK-LABEL: func.func @reversed_tag_order
// CHECK:         %[[T:.*]] = arith.constant 0 : i32
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}}, %[[T]])
// CHECK-NOT:     spmd.send
func.func @reversed_tag_order(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Uniform execution is not required: the group is inside a branch on the
// rank, so not every rank executes it.
// It merges anyway, because a group is a run inside one block - whichever ranks
// enter the block execute all of it - and because the grouping keys on the SSA
// identity of the peer expression rather than on its value.  A uniformity gate
// here would decline every real halo exchange.
// CHECK-LABEL: func.func @inside_rank_dependent_branch
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @inside_rank_dependent_branch() {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %rank, %rerr = "spmd.getRankInComm"(%comm) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %peer = arith.addi %rank, %c1 : i32
  %cond = arith.cmpi sgt, %rank, %c0 : i32
  scf.if %cond {
    %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
    %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  }
  return
}

// A write to a member's buffer between the members: merging would send the
// post-write bytes for the earlier piece too.
// CHECK-LABEL: func.func @store_to_group_buffer_between
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @store_to_group_buffer_between(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %c42 = arith.constant 42 : i32
  %t0 = arith.constant 20 : i32
  %t1 = arith.constant 21 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  llvm.store %c42, %buf : i32, !llvm.ptr
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A write to something else is transparent.  This is the positive that makes
// the previous function meaningful rather than a blanket "any store blocks".
// CHECK-LABEL: func.func @store_to_unrelated_between
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @store_to_unrelated_between(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %c42 = arith.constant 42 : i32
  %t0 = arith.constant 30 : i32
  %t1 = arith.constant 31 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %other = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  llvm.store %c42, %other : i32, !llvm.ptr
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The deadlock case.  Moving the first send past this receive reorders
// communication: the peer may be blocked in the matching send, waiting for a
// receive that no longer runs first.
// CHECK-LABEL: func.func @intervening_recv
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @intervening_recv(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 40 : i32
  %t1 = arith.constant 41 : i32
  %t2 = arith.constant 42 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %in = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %r = "spmd.recv"(%comm, %in, %n, %dt, %peer, %t2) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A barrier is communication too: crossing it changes which ranks have arrived
// by the time the message is posted.
// CHECK-LABEL: func.func @intervening_barrier
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @intervening_barrier(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 50 : i32
  %t1 = arith.constant 51 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %b = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A call whose effects are not described could do anything, including
// communicate or write the buffers.
// CHECK-LABEL: func.func @opaque_call_between
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func private @external()
func.func @opaque_call_between(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 60 : i32
  %t1 = arith.constant 61 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  func.call @external() : () -> ()
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Members in different blocks are not a run: whether both execute depends on
// the branch, so collapsing them would change per-rank message counts.
// CHECK-LABEL: func.func @different_blocks
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @different_blocks(%peer : i32, %cond : i1) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 70 : i32
  %t1 = arith.constant 71 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  scf.if %cond {
    %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  }
  return
}

// Inside a thread team other threads run the same block concurrently, so the
// ops between two members are not all that happens between them.
// CHECK-LABEL: func.func @inside_thread_team
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @inside_thread_team(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 80 : i32
  %t1 = arith.constant 81 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  "spmd.parallel"(%comm) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
    %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  return
}

// A receive into the group's buffer is pending across the group: it writes the
// buffer at some point up to its wait, not where it sits.
// CHECK-LABEL: func.func @pending_recv_into_group_buffer
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @pending_recv_into_group_buffer(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 90 : i32
  %t1 = arith.constant 91 : i32
  %t2 = arith.constant 92 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %req, %er = "spmd.recv"(%comm, %buf, %n, %dt, %peer, %t2) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %w = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// A call nested in a region is as opaque as one that is not: this helper
// receives, and merging would move the first send past that receive.
// CHECK-LABEL: func.func @call_inside_region_between
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func private @receiving_helper(%comm : !spmd.comm, %peer : i32) {
  %c1 = arith.constant 1 : i32
  %t = arith.constant 102 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %in = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %r = "spmd.recv"(%comm, %in, %n, %dt, %peer, %t) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
func.func @call_inside_region_between(%peer : i32, %cond : i1) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 100 : i32
  %t1 = arith.constant 101 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  scf.if %cond {
    func.call @receiving_helper(%comm, %peer) : (!spmd.comm, i32) -> ()
  }
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A wait carries no Communication trait, but it blocks until another rank has
// acted, which orders it against the sends around it just the same.
// CHECK-LABEL: func.func @wait_between
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @wait_between(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 110 : i32
  %t1 = arith.constant 111 : i32
  %t2 = arith.constant 112 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %in = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %req, %er = "spmd.recv"(%comm, %in, %n, %dt, %peer, %t2) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %w = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
