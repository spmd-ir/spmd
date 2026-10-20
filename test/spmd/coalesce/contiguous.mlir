// RUN: spmd-opt %s -coalesce-comm=allow-staging=false | FileCheck %s
//
// The zero-copy tier: pieces that are already adjacent inside one allocation
// merge by rewriting the count alone.  Staging is off here, so "merged" means
// exactly "the adjacency test succeeded".
//
// Every function uses its own tag range.  That is not cosmetic: the legality
// check is module-wide, so two functions sharing tag 0 with different groupings
// would invalidate each other, and the tests would be measuring interference
// rather than the property under test.  tag-partition.mlir exercises that
// interference deliberately.

// Two pieces of one allocation, adjacent in ascending-tag order: one send of
// the sum, carrying the least tag, from the lowest address.  No copies.
// CHECK-LABEL: func.func @two_adjacent_sends
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
// CHECK-NOT:     llvm.intr.memcpy
func.func @two_adjacent_sends(%peer : i32) {
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

// Receives merge the same way.
// CHECK-LABEL: func.func @two_adjacent_recvs
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.recv"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.recv
func.func @two_adjacent_recvs(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.recv"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.recv"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Three pieces chain the same way.
// CHECK-LABEL: func.func @three_adjacent
// CHECK:         %[[N:.*]] = arith.constant 12 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @three_adjacent(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 20 : i32
  %t1 = arith.constant 21 : i32
  %t2 = arith.constant 22 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<12 x i32> : (i32) -> !llvm.ptr
  %mid = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %top = llvm.getelementptr %buf[8] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %mid, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e2 = "spmd.send"(%comm, %top, %n, %dt, %peer, %t2) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A gap between the pieces: the merged message would carry bytes nobody sent.
// CHECK-LABEL: func.func @gap_between_pieces
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @gap_between_pieces(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 30 : i32
  %t1 = arith.constant 31 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<16 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[5] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Overlapping pieces would send the shared bytes once instead of twice.
// CHECK-LABEL: func.func @overlapping_pieces
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @overlapping_pieces(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 40 : i32
  %t1 = arith.constant 41 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<16 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[2] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Ascending tag order is the layout, so the pieces must ascend in *address*
// order too or the merged message would put the far piece first.  Here the
// higher tag names the lower address.  Silent corruption if merged.
// CHECK-LABEL: func.func @tag_order_not_address_order
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @tag_order_not_address_order(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 50 : i32
  %t1 = arith.constant 51 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Separate allocations are never adjacent, whatever the addresses turn out to
// be at run time.  Only the staging tier can take this one.
// CHECK-LABEL: func.func @different_allocations
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @different_allocations(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 60 : i32
  %t1 = arith.constant 61 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The case the whole symbolic-offset machinery exists for: neither offset is a
// constant, but `&u[i]` of 4 elements is adjacent to `&u[i+4]` because the `i`
// terms cancel.  Without this every loop-carried halo would be declined.
// CHECK-LABEL: func.func @symbolic_offsets_cancel
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @symbolic_offsets_cancel(%peer : i32, %i : i64) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 70 : i32
  %t1 = arith.constant 71 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<256 x i32> : (i32) -> !llvm.ptr
  %lo = llvm.getelementptr %buf[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %hi = llvm.getelementptr %lo[%n] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %lo, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Two different symbolic strides do not cancel, and nothing is known about
// their difference.  "Unknown distance" is not "adjacent".
// CHECK-LABEL: func.func @symbolic_offsets_differ
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @symbolic_offsets_differ(%peer : i32, %i : i64, %j : i64) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 80 : i32
  %t1 = arith.constant 81 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<256 x i32> : (i32) -> !llvm.ptr
  %lo = llvm.getelementptr %buf[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %hi = llvm.getelementptr %buf[%j] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %lo, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// The base does not have to be an allocation.  Adjacency is a fact about
// pointer arithmetic, and two pieces of a buffer that arrived as a function
// argument are as adjacent as two pieces of an alloca - which matters, because
// raised code passes buffers around.  What the argument *does* cost is the
// precise escape analysis, so the intervening-operation test falls back to its
// coarser form; nothing pure sits between these two, so the group still forms.
// CHECK-LABEL: func.func @argument_base_still_adjacent
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @argument_base_still_adjacent(%peer : i32, %opaque : !llvm.ptr) {
  %t0 = arith.constant 90 : i32
  %t1 = arith.constant 91 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %hi = llvm.getelementptr %opaque[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %opaque, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A multi-index GEP into an aggregate resolves like a flat one: the second
// piece is at byte 16, right after the first.
// CHECK-LABEL: func.func @multi_index_gep
// CHECK:         %[[N:.*]] = arith.constant 8 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @multi_index_gep(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 100 : i32
  %t1 = arith.constant 101 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<8 x i32> : (i32) -> !llvm.ptr
  %hi = llvm.getelementptr %buf[0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A displacement the analysis declines to model - here memref.reinterpret_cast
// - is reported as unknown, and unknown is not adjacent.  The piece really is
// at element 4, so this is the pass being conservative rather than
// correct-by-luck: it must not guess.
// CHECK-LABEL: func.func @unknown_displacement
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
func.func @unknown_displacement(%peer : i32) {
  %t0 = arith.constant 110 : i32
  %t1 = arith.constant 111 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<8xi32>
  %hi = memref.reinterpret_cast %buf to offset: [4], sizes: [4], strides: [1] : memref<8xi32> to memref<4xi32, strided<[1], offset: 4>>
  %e0 = "spmd.send"(%comm, %buf, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32, strided<[1], offset: 4>>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}
