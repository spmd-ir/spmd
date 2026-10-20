// RUN: spmd-opt %s --allow-unregistered-dialect -coalesce-comm="max-staged-bytes=0 max-merged-bytes=0" | FileCheck %s
//
// The staged tier with both size limits off, so that nothing but the shape of
// the input decides: sizes that do not fit the arithmetic, and a group with no
// function to hold its scratch buffer.
//
// Each scenario uses its own tags; the legality check is module-wide, see the
// header of contiguous.mlir.

// Two pieces of 2^30 i32 each.  The second starts at byte 2^32 of the scratch
// buffer, which does not fit a 32-bit GEP index; truncated, it would be byte
// 0, and the second copy would overwrite the first.
// CHECK-LABEL: func.func @offset_beyond_int32
// CHECK:         %[[STAGE:.*]] = llvm.alloca {{.*}} x i8
// CHECK:         "llvm.intr.memcpy"(%[[STAGE]],
// CHECK:         %[[OFF:.*]] = llvm.mlir.constant(4294967296 : i64)
// CHECK-NEXT:    %[[SLOT:.*]] = llvm.getelementptr %[[STAGE]][%[[OFF]]]
// CHECK:         "llvm.intr.memcpy"(%[[SLOT]],
// CHECK:         %[[N:.*]] = arith.constant 2147483648 : i64
// CHECK:         "spmd.send"({{.*}}, %[[N]], {{.*}})
// CHECK-NOT:     spmd.send
func.func @offset_beyond_int32(%peer : i32) {
  %t0 = arith.constant 0 : i32
  %t1 = arith.constant 1 : i32
  %n = arith.constant 1073741824 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %n x i32 : (i64) -> !llvm.ptr
  %b = llvm.alloca %n x i32 : (i64) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// A piece whose size in bytes does not fit in 64 bits: 2^62 i32.
// CHECK-LABEL: func.func @extent_overflows
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
// CHECK-NOT:     llvm.intr.memcpy
func.func @extent_overflows(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 10 : i32
  %t1 = arith.constant 11 : i32
  %n = arith.constant 4611686018427387904 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// Two pieces that fit on their own, 2^62 bytes each, and whose sum does not.
// CHECK-LABEL: func.func @total_overflows
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
// CHECK-NOT:     llvm.intr.memcpy
func.func @total_overflows(%peer : i32) {
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 20 : i32
  %t1 = arith.constant 21 : i32
  %n = arith.constant 4611686018427387904 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x i8 : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x i8 : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// No function encloses this group, so there is no entry block to hold the
// scratch buffer.  The group stays as it is.
// CHECK:         "test.container"
// CHECK-COUNT-2: spmd.send
// CHECK-NOT:     spmd.send
// CHECK-NOT:     llvm.intr.memcpy
"test.container"() ({
  %peer = arith.constant 1 : i32
  %c1 = arith.constant 1 : i32
  %t0 = arith.constant 30 : i32
  %t1 = arith.constant 31 : i32
  %n = arith.constant 4 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %a = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1 x !llvm.array<4 x i32> : (i32) -> !llvm.ptr
  %e0 = "spmd.send"(%comm, %a, %n, %dt, %peer, %t0) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  %e1 = "spmd.send"(%comm, %b, %n, %dt, %peer, %t1) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  "test.terminator"() : () -> ()
}) : () -> ()
