// RUN: spmd-opt %s -coalesce-comm | FileCheck %s
// RUN: spmd-opt %s -coalesce-comm=allow-staging=false | FileCheck %s --check-prefix=NOSTAGE
//
// The staging tier: pieces that are not already adjacent are packed through a
// scratch buffer.
//
// This tier is strictly more demanding than the zero-copy one, and for a reason
// worth stating.  A contiguous merge ships a flat total, so only the *sum* has
// to agree between the two endpoints - and it does, because each original pair
// matched.  Staging has the compiler choose where each piece sits in the
// message, so *every* piece must agree, which is what forces compile-time
// constant counts here and nowhere else.  Getting that wrong is silent: the
// data arrives, at the wrong offsets.
//
// Layout is a plain concatenation in ascending-tag order with no padding, so
// both endpoints derive it without having exchanged anything.

// Pack, send once, and the scratch buffer lives in the entry block.
// CHECK-LABEL: func.func @two_disjoint_sends
// CHECK:         llvm.alloca {{.*}} x i8
// CHECK:         llvm.intr.memcpy
// CHECK:         llvm.intr.memcpy
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
//
// NOSTAGE-LABEL: func.func @two_disjoint_sends
// NOSTAGE-COUNT-2: spmd.send
// NOSTAGE-NOT:     spmd.send
// NOSTAGE-NOT:     llvm.intr.memcpy
func.func @two_disjoint_sends(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Receiving is the mirror image: one receive, then the copies out.  The unpack
// has to follow the receive, which is why the non-blocking form - where the
// data lands at the completion rather than at the call - is a separate problem
// and out of scope here.
// NOSTAGE-LABEL: func.func @two_disjoint_recvs
// NOSTAGE-COUNT-2: spmd.recv
// NOSTAGE-NOT:     spmd.recv
//
// CHECK-LABEL: func.func @two_disjoint_recvs
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.recv"({{.*}}, %[[N]], {{.*}})
// CHECK:         llvm.intr.memcpy
// CHECK:         llvm.intr.memcpy
// CHECK-NOT:     spmd.recv
func.func @two_disjoint_recvs(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e0 = "spmd.recv"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.recv"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The group is inside a loop, so the scratch buffer must not be: an allocation
// at the merge point would grow the stack on every iteration.  Exactly one
// alloca, in the entry block, with the copies inside.
// CHECK-LABEL: func.func @inside_loop
// CHECK:         llvm.alloca {{.*}} x i8
// CHECK:         scf.for
// CHECK-NOT:     llvm.alloca {{.*}} x i8
// CHECK:         llvm.intr.memcpy
// CHECK:         llvm.intr.memcpy
// CHECK:         spmd.send
// CHECK-NOT:     spmd.send
func.func @inside_loop(%peer : i32, %ub : index) {
  %c1 = arith.constant 1 : i32
  %c0i = arith.constant 0 : index
  %c1i = arith.constant 1 : index
  %t0 = arith.constant 20 : i32
  %t1 = arith.constant 21 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  scf.for %i = %c0i to %ub step %c1i {
    %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
    %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  }
  return
}

// One constant count per tag.  A count nobody can read at compile time may
// differ per rank, and then the two endpoints disagree about where the second
// piece begins - the data arrives, silently misplaced.  Declined.
// CHECK-LABEL: func.func @non_constant_count
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
// CHECK-NOT:     llvm.intr.memcpy
func.func @non_constant_count(%peer : i32, %n : i64) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 30 : i32
  %t1 = arith.constant 31 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<64 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<64 x i32> : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Past the staging threshold the copies cost more than the per-message overhead
// they save.
// CHECK-LABEL: func.func @over_staging_threshold
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
// CHECK-NOT:     llvm.intr.memcpy
func.func @over_staging_threshold(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 40 : i32
  %t1 = arith.constant 41 : i32
  %n = arith.constant 2048 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<2048 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<2048 x i32> : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The eager-limit guard, on the zero-copy tier where no copies are involved at
// all.  These two pieces are adjacent and merging them is free, but the result
// would be a 32 KiB message where there were two 16 KiB ones - and two sends
// that each fit under the transport's eager threshold can merge into one that
// does not, turning a program that never blocked into one that waits for its
// match.
// CHECK-LABEL: func.func @over_eager_threshold
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @over_eager_threshold(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 50 : i32
  %t1 = arith.constant 51 : i32
  %n = arith.constant 4096 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8192 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4096] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The eager limit applies to a count nobody can read at compile time too.  These
// pieces are adjacent, so the zero-copy tier would take them, but nothing shows
// that 2n elements stay under the limit: n may be 4096 here just as well.
// CHECK-LABEL: func.func @symbolic_count_eager_threshold
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @symbolic_count_eager_threshold(%peer : i32, %n : i64) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 70 : i32
  %t1 = arith.constant 71 : i32
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8192 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[%n] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Receive buffers may overlap.  This program receives tag 1 into buf[2..6)
// and then tag 0 into buf[0..4), so buf[2..4) ends up holding tag 0's data.
// The layout is still in tag order, but the unpack copies follow the program:
// tag 1's first, tag 0's last.
// NOSTAGE-LABEL: func.func @overlapping_recvs_against_tag_order
//
// CHECK-LABEL: func.func @overlapping_recvs_against_tag_order
// CHECK:         %[[BUF:.*]] = llvm.alloca {{.*}} x !llvm.array<8 x i32>
// CHECK:         %[[MID:.*]] = llvm.getelementptr %[[BUF]][2]
// CHECK:         spmd.recv
// CHECK:         "llvm.intr.memcpy"(%[[MID]],
// CHECK:         "llvm.intr.memcpy"(%[[BUF]],
// CHECK-NOT:     spmd.recv
func.func @overlapping_recvs_against_tag_order(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 80 : i32
  %t1 = arith.constant 81 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %mid = llvm.getelementptr %buf[2] : (!llvm.ptr) -> !llvm.ptr, i32
  %e1 = "spmd.recv"(%comm, %mid, %n, %dt, %peer, %t1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e0 = "spmd.recv"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The sender the receives above match, so that their group is legal.
// CHECK-LABEL: func.func @sender_for_overlapping_recvs
// CHECK:         spmd.send
// CHECK-NOT:     spmd.send
func.func @sender_for_overlapping_recvs(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 80 : i32
  %t1 = arith.constant 81 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The raiser hands buffers over as memref views of host memory.  They are
// packed at the address enzymexla.memref2pointer gives them.
// CHECK-LABEL: func.func @memref_buffers_staged
// CHECK:         %[[STAGING:.*]] = llvm.alloca %{{.*}} x i8
// CHECK:         %[[A:.*]] = "enzymexla.memref2pointer"(%{{.*}}) : (memref<4xi32>) -> !llvm.ptr
// CHECK:         llvm.intr.memcpy{{.*}}(%[[STAGING]], %[[A]],
// CHECK:         %[[B:.*]] = "enzymexla.memref2pointer"(%{{.*}}) : (memref<4xi32>) -> !llvm.ptr
// CHECK:         llvm.intr.memcpy{{.*}}(%{{.*}}, %[[B]],
// CHECK-COUNT-1: "spmd.send"(%{{[^,]+}}, %[[STAGING]],
// CHECK-NOT:     spmd.send
func.func @memref_buffers_staged(%peer : i32) {
  %t0 = arith.constant 60 : i32
  %t1 = arith.constant 61 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = memref.alloca() : memref<4xi32>
  %b = memref.alloca() : memref<4xi32>
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A memref with a layout does not start where its elements do, so it is not
// staged.
// CHECK-LABEL: func.func @strided_memref_not_staged
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     llvm.intr.memcpy
func.func @strided_memref_not_staged(%peer : i32) {
  %t0 = arith.constant 62 : i32
  %t1 = arith.constant 63 : i32
  %n = arith.constant 2 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = memref.alloca() : memref<8xi32>
  %b = memref.alloca() : memref<8xi32>
  %a2 = memref.reinterpret_cast %a to offset: [2], sizes: [2], strides: [1] : memref<8xi32> to memref<2xi32, strided<[1], offset: 2>>
  %b2 = memref.reinterpret_cast %b to offset: [2], sizes: [2], strides: [1] : memref<8xi32> to memref<2xi32, strided<[1], offset: 2>>
  %e0 = "spmd.send"(%comm, %a2, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<2xi32, strided<[1], offset: 2>>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b2, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<2xi32, strided<[1], offset: 2>>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
