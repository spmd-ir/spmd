// RUN: spmd-opt %s -test-spmd-buffer-slices | FileCheck %s
//
// The buffer queries a communication optimization builds on, one buffer per
// case: resolveBufferSlice (allocation + byte displacement), getDatatypeByteSize
// and computeBufferFacts, and what CommBufferOpInterface says about each op.
//
// A slice reads "<base> + <displacement>". The displacement is a constant plus
// "<scale>*<value>" terms, or "?" when a step of the address computation has no
// recoverable displacement. Getting one wrong would make a coalescing pass
// merge the wrong pieces, which corrupts data instead of failing, so the
// arithmetic is pinned here on its own.

// A constant GEP index scales by the element type: 4 * sizeof(i32).
// CHECK-LABEL: func.func @gep_constant
// CHECK:         %[[BUF:.*]] = llvm.alloca
// CHECK:         spmd.send
// CHECK-SAME:    test.send = "%[[BUF]] + 16, 4 bytes, private, readers=1, writers=0"
func.func @gep_constant() {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<16 x i32> : (i32) -> !llvm.ptr
  %gep = llvm.getelementptr %buf[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e = "spmd.send"(%comm, %gep, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A dynamic index becomes a term instead of giving up. It is what every halo
// loop produces; an analysis that only knew constants would decline them all.
// CHECK-LABEL: func.func @gep_dynamic
// CHECK:         test.send = "%{{.*}} + 0 + 8*%arg0, 8 bytes, private, readers=1, writers=0"
func.func @gep_dynamic(%i : i64) {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<16 x f64> : (i32) -> !llvm.ptr
  %gep = llvm.getelementptr %buf[%i] : (!llvm.ptr, i64) -> !llvm.ptr, f64
  %e = "spmd.send"(%comm, %gep, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Two GEPs off one base with the same dynamic term are a constant 16 bytes
// apart although neither offset is constant. Both sends read the allocation.
// CHECK-LABEL: func.func @gep_dynamic_pair
// CHECK:         test.send = "%{{.*}} + 0 + 4*%arg0, 4 bytes, private, readers=2, writers=0"
// CHECK:         test.send = "%{{.*}} + 16 + 4*%arg0, 4 bytes, private, readers=2, writers=0"
func.func @gep_dynamic_pair(%i : i64) {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<64 x i32> : (i32) -> !llvm.ptr
  %lo = llvm.getelementptr %buf[%i] : (!llvm.ptr, i64) -> !llvm.ptr, i32
  %hi = llvm.getelementptr %lo[4] : (!llvm.ptr) -> !llvm.ptr, i32
  %e0 = "spmd.send"(%comm, %lo, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %hi, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A multi-index GEP descends into the aggregate: field 2 of { i8, i32, f64 }
// starts at 8 after padding, and a dynamic index into an array field is a term
// scaled by that array's element.
// CHECK-LABEL: func.func @gep_struct
// CHECK:         test.send = "%{{.*}} + 8, 8 bytes, private, readers=2, writers=0"
// CHECK:         test.send = "%{{.*}} + 8 + 8*%arg0, 8 bytes, escapes"
func.func @gep_struct(%i : i64) {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %s = llvm.alloca %c1 x !llvm.struct<(i8, i32, f64)> : (i32) -> !llvm.ptr
  %field = llvm.getelementptr %s[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(i8, i32, f64)>
  %e0 = "spmd.send"(%comm, %field, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %v = llvm.load %s : !llvm.ptr -> i8
  %t = llvm.alloca %c1 x !llvm.struct<(i32, array<4 x f64>)> : (i32) -> !llvm.ptr
  %elem = llvm.getelementptr %t[0, 1, %i] : (!llvm.ptr, i64) -> !llvm.ptr, !llvm.struct<(i32, array<4 x f64>)>
  %e1 = "spmd.send"(%comm, %elem, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  llvm.call @keep(%t) : (!llvm.ptr) -> ()
  return
}
llvm.func @keep(!llvm.ptr)

// The second element of a request array, spelled the way clang spells
// &reqs[1]: the element sits 4 bytes in. The irecv writes the buffer, the
// store writes it too.
// CHECK-LABEL: func.func @array_element
// CHECK:         test.recv = "%{{.*}} + 4, 4 bytes, private, readers=0, writers=2"
func.func @array_element() {
  %c1 = arith.constant 1 : i32
  %c0_64 = arith.constant 0 : i64
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %arr = llvm.alloca %c1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %elem = llvm.getelementptr inbounds %arr[0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<2 x i32>
  llvm.store %c1, %elem : i32, !llvm.ptr
  %e = "spmd.recv"(%comm, %elem, %c4, %dt, %c1, %c1) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// memref.subview: offset 4 * stride 1 * sizeof(i32).
// CHECK-LABEL: func.func @subview_static
// CHECK:         test.send = "%alloca + 16, 4 bytes, private, readers=1, writers=0"
func.func @subview_static() {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<16xi32>
  %sv = memref.subview %buf[4] [4] [1] : memref<16xi32> to memref<4xi32, strided<[1], offset: 4>>
  %e = "spmd.send"(%comm, %sv, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32, strided<[1], offset: 4>>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// enzymexla.subindex peels the leading dimension: index * 8 elements * 4 bytes.
// CHECK-LABEL: func.func @subindex
// CHECK:         test.send = "%alloca + 0 + 32*%arg0, 4 bytes, private, readers=1, writers=0"
func.func @subindex(%i : index) {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<4x8xi32>
  %sub = "enzymexla.subindex"(%buf, %i) : (memref<4x8xi32>, index) -> memref<8xi32>
  %e = "spmd.send"(%comm, %sub, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// memref.cast and the memref<->pointer bridges only re-type the address.
// CHECK-LABEL: func.func @cast_and_bridge
// CHECK:         test.send = "%alloca + 0, 4 bytes, private, readers=1, writers=0"
func.func @cast_and_bridge() {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<16xi32>
  %cast = memref.cast %buf : memref<16xi32> to memref<?xi32>
  %ptr = "enzymexla.memref2pointer"(%cast) : (memref<?xi32>) -> !llvm.ptr
  %e = "spmd.send"(%comm, %ptr, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// memref.reinterpret_cast points into its source, but where is not modelled:
// the base is still found, the displacement is "?" rather than a guessed 0.
// CHECK-LABEL: func.func @reinterpret_cast
// CHECK:         test.send = "%alloca + ?, 4 bytes, private, readers=1, writers=0"
func.func @reinterpret_cast() {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<16xi32>
  %rc = memref.reinterpret_cast %buf to offset: [4], sizes: [4], strides: [1] : memref<16xi32> to memref<4xi32, strided<[1], offset: 4>>
  %e = "spmd.send"(%comm, %rc, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<4xi32, strided<[1], offset: 4>>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Element sizes come from spmd.constDatatype's type, whatever its signedness;
// a datatype that is not a constant has none.
// CHECK-LABEL: func.func @elem_sizes
// CHECK:         test.send = "%alloca + 0, 1 bytes
// CHECK:         test.send = "%alloca + 0, 4 bytes
// CHECK:         test.send = "%alloca + 0, 4 bytes
// CHECK:         test.send = "%alloca + 0, 8 bytes
// CHECK:         test.send = "%alloca + 0, ? bytes
func.func @elem_sizes(%opaque : !spmd.datatype) {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = memref.alloca() : memref<16xi32>
  %i8t = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}> : () -> !spmd.datatype
  %si32t = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %ui32t = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %f64t = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %e0 = "spmd.send"(%comm, %buf, %c4, %i8t, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e3 = "spmd.send"(%comm, %buf, %c4, %si32t, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e4 = "spmd.send"(%comm, %buf, %c4, %ui32t, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %buf, %c4, %f64t, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e2 = "spmd.send"(%comm, %buf, %c4, %opaque, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Memory that other code can reach without a local use showing is not
// private: an address passed to a call, an address stored into memory, and
// symmetric memory from spmd.malloc.
// CHECK-LABEL: func.func @escapes
// CHECK:         test.send = "%{{.*}} + 0, 4 bytes, escapes"
// CHECK:         test.send = "%{{.*}} + 0, 4 bytes, escapes"
// CHECK:         test.send = "%{{.*}} + 0, 4 bytes, escapes"
func.func @escapes() {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %c16 = arith.constant 16 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  llvm.call @keep(%a) : (!llvm.ptr) -> ()
  %e0 = "spmd.send"(%comm, %a, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %b = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %slot = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  llvm.store %b, %slot : !llvm.ptr, !llvm.ptr
  %e1 = "spmd.send"(%comm, %b, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %sym, %em = "spmd.malloc"(%comm, %c16) <{usedModel = 0 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  %e2 = "spmd.send"(%comm, %sym, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Lifetime markers neither read nor capture the buffer.
// CHECK-LABEL: func.func @lifetime_markers
// CHECK:         test.send = "%{{.*}} + 0, 4 bytes, private, readers=1, writers=0"
func.func @lifetime_markers() {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %buf = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  llvm.intr.lifetime.start %buf : !llvm.ptr
  %e = "spmd.send"(%comm, %buf, %c4, %dt, %c1, %c1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  llvm.intr.lifetime.end %buf : !llvm.ptr
  return
}

// What CommBufferOpInterface states per op, beyond the operands.
// CHECK-LABEL: func.func @interface_roles
// CHECK:         spmd.allreduce
// CHECK-NOT:     test.recv_not_on_all_ranks
// CHECK-SAME:    test.recv = "%[[R:.*]] + 0, 8 bytes, private, readers=0, writers={{[0-9]+}}"
// CHECK-SAME:    test.send = "%[[S:.*]] + 0, 8 bytes, private, readers={{[0-9]+}}, writers=0"
// CHECK:         spmd.reduce
// CHECK-SAME:    test.recv_not_on_all_ranks
// CHECK-SAME:    test.root = "%c0_i32"
// CHECK:         spmd.bcast
// CHECK-SAME:    test.recv_not_on_all_ranks
// CHECK-SAME:    test.root = "%c0_i32"
// CHECK:         spmd.reduceScatter
// CHECK-SAME:    test.send_count_per_rank
// CHECK:         spmd.exscan
// CHECK-SAME:    test.recv_not_on_all_ranks
// CHECK:         spmd.scatter
// CHECK-SAME:    test.root = "%c0_i32"
// CHECK-SAME:    test.send_count_per_rank
// CHECK:         spmd.gather
// CHECK-SAME:    test.recv_count_per_rank
// CHECK-SAME:    test.recv_not_on_all_ranks
// CHECK-SAME:    test.root = "%c0_i32"
// CHECK:         spmd.allgather
// CHECK-SAME:    test.recv_count_per_rank
// CHECK-NOT:     test.send_count_per_rank
// CHECK:         spmd.alltoall
// CHECK-SAME:    test.recv_count_per_rank
// CHECK-SAME:    test.send_count_per_rank
// CHECK:         spmd.sendrecv
// CHECK-SAME:    test.recv = "%[[R]] + 0, 8 bytes
// CHECK-SAME:    test.send = "%[[S]] + 0, 8 bytes
func.func @interface_roles() {
  %c0_i32 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c1_64 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %send = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
  %recv = llvm.alloca %c1 x !llvm.array<8 x f64> : (i32) -> !llvm.ptr
  %e0 = "spmd.allreduce"(%comm, %send, %recv, %c8, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e1 = "spmd.reduce"(%comm, %send, %recv, %c8, %dt, %sum, %c0_i32) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %e2 = "spmd.bcast"(%comm, %send, %recv, %c8, %dt, %c0_i32) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  %e3 = "spmd.reduceScatter"(%comm, %send, %recv, %c8, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e4 = "spmd.exscan"(%comm, %send, %recv, %c8, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %e5 = "spmd.scatter"(%comm, %send, %c8, %dt, %recv, %c8, %dt, %c0_i32) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  %e6 = "spmd.gather"(%comm, %send, %c8, %dt, %recv, %c8, %dt, %c0_i32) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype, i32) -> !spmd.error
  %e7 = "spmd.allgather"(%comm, %send, %c8, %dt, %recv, %c8, %dt) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> !spmd.error
  %e8 = "spmd.alltoall"(%comm, %send, %c8, %dt, %recv, %c8, %dt) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, !llvm.ptr, i64, !spmd.datatype) -> !spmd.error
  %e9 = "spmd.sendrecv"(%comm, %send, %c8, %dt, %c1_64, %c1, %c0_i32, %recv, %c8, %dt, %c1_64, %c1, %c0_i32) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32, !llvm.ptr, i64, !spmd.datatype, i64, i32, i32) -> !spmd.error
  return
}
