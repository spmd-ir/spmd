// RUN: spmd-opt %s -lower-spmd-to-shmem | FileCheck %s --implicit-check-not='"spmd.'
//
// One-sided transfers.  A window is the symmetric address it is over, so a
// transfer's target address is the window itself.

// A window allocated by spmd.winAlloc is shmem_malloc's symmetric address, and
// the transfers go to it: roundtrip passed the origin buffer as both source
// and target.  A stride other than 1 is shmem_<T>_iput, which takes the target
// stride before the origin's; an i128 transfer has no type name, only
// shmem_put128.  The non-blocking forms are _nbi and hand out no request, so
// the wait and the free of the window each complete everything outstanding.
// CHECK-LABEL: func.func @put_get
// CHECK:         %[[C1:.*]] = arith.constant 1 : i64
// CHECK:         %[[BASE:.*]] = llvm.call @shmem_malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:         llvm.call @shmem_int_put(%[[BASE]], %arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_int_get_nbi(%arg0, %[[BASE]], %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_int_iput(%[[BASE]], %arg0, %[[C1]], %arg3, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32) -> ()
// CHECK:         llvm.call @shmem_put128(%[[BASE]], %arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_int_put_nbi(%[[BASE]], %arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_quiet() : () -> ()
// CHECK:         llvm.call @shmem_quiet() : () -> ()
// CHECK-NEXT:    llvm.call @shmem_free(%[[BASE]]) : (!llvm.ptr) -> ()
func.func @put_get(%src: !llvm.ptr, %n: i64, %pe: i32, %stride: i64) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64
  %unit = arith.constant 1 : i32
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %i128 = "spmd.constDatatype"() <{typeAttr = i128, usedModel = 1 : i32}> : () -> !spmd.datatype
  %base, %win, %e0 = "spmd.winAlloc"(%world, %c8, %unit) <{usedModel = 1 : i32}> : (!spmd.comm, i64, i32) -> (!llvm.ptr, !spmd.win, !spmd.error)
  %e1 = "spmd.put"(%src, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e2 = "spmd.get"(%src, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = false, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e3 = "spmd.put"(%src, %n, %i32, %stride, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e4 = "spmd.put"(%src, %n, %i128, %c1, %pe, %win, %c0, %n, %i128, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %req, %e5 = "spmd.put"(%src, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = false, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> (!spmd.req, !spmd.error)
  %e6 = "spmd.wait"(%req) <{usedModel = 1 : i32}> : (!spmd.req) -> !spmd.error
  %e7 = "spmd.winFree"(%win) <{usedModel = 1 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// A transfer of ui32 is shmem_uint_put, which a raised shmem_uint_put lowers
// back to, and one of i32, whose signedness is unknown, is the sized
// shmem_put32.  char's i8 is shmem_char_put rather than shmem_put8.
// CHECK-LABEL: func.func @put_signedness
// CHECK:         llvm.call @shmem_uint_put(%arg3, %arg0, %arg1, %arg2)
// CHECK:         llvm.call @shmem_put32(%arg3, %arg0, %arg1, %arg2)
// CHECK:         llvm.call @shmem_char_put(%arg3, %arg0, %arg1, %arg2)
func.func @put_signedness(%src: !llvm.ptr, %n: i64, %pe: i32, %sym: !llvm.ptr) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %ui32 = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %i32 = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %i8 = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 1 : i32}> : () -> !spmd.datatype
  %win, %e0 = "spmd.winCreate"(%world, %sym, %c0_i32) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.put"(%src, %n, %ui32, %c1, %pe, %win, %c0, %n, %ui32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e2 = "spmd.put"(%src, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e3 = "spmd.put"(%src, %n, %i8, %c1, %pe, %win, %c0, %n, %i8, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}

// A window created over an address is that address, and freeing it completes
// what is outstanding but leaves the memory to the program, which made it: a
// shmem_free here would free it a second time.  The atomic fetch returns the
// element, which the lowering stores where the operation reads it from.
// CHECK-LABEL: func.func @window_over
// CHECK:         %[[VALUE:.*]] = llvm.call @shmem_double_atomic_fetch(%arg0, %arg2) : (!llvm.ptr, i32) -> f64
// CHECK-NEXT:    llvm.store %[[VALUE]], %arg1 : f64, !llvm.ptr
// CHECK:         llvm.call @shmem_quiet() : () -> ()
// CHECK-NOT:     @shmem_free
// CHECK:         return
func.func @window_over(%sym: !llvm.ptr, %dst: !llvm.ptr, %pe: i32) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %f64 = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 1 : i32}> : () -> !spmd.datatype
  %win, %e0 = "spmd.winCreate"(%world, %sym, %c0_i32) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.get"(%dst, %c1, %f64, %c1, %pe, %win, %c0, %c1, %f64, %c1) <{isAtomic = true, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e2 = "spmd.winFree"(%win) <{usedModel = 1 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// A global variable is symmetric in OpenSHMEM, so a window over one is a
// target like any other: shmem_int_put(&counter, ...) raises to exactly this.
llvm.mlir.global internal @counter(0 : i32) : i32

// CHECK-LABEL: func.func @window_over_global
// CHECK:         %[[COUNTER:.*]] = llvm.mlir.addressof @counter : !llvm.ptr
// CHECK:         llvm.call @shmem_int_put(%[[COUNTER]], %arg0, %{{.*}}, %arg1)
func.func @window_over_global(%src: !llvm.ptr, %pe: i32) {
  %world = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %counter = llvm.mlir.addressof @counter : !llvm.ptr
  %win, %e0 = "spmd.winCreate"(%world, %counter, %c0_i32) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 1 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.put"(%src, %c1, %i32, %c1, %pe, %win, %c0, %c1, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}

// One count passed as both the origin and the target count, widened by the
// raiser once for each: the call takes it once, and the two widenings are the
// same value.
// CHECK-LABEL: func.func @count_widened_twice
// CHECK:         %[[ORIGIN:.*]] = arith.extui %arg1 : i32 to i64
// CHECK:         llvm.call @shmem_int_put(%{{.*}}, %arg0, %[[ORIGIN]], %arg2)
func.func @count_widened_twice(%src: !llvm.ptr, %n: i32, %pe: i32, %win: !spmd.win) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %originCount = arith.extui %n : i32 to i64
  %targetCount = arith.extui %n : i32 to i64
  %e = "spmd.put"(%src, %originCount, %i32, %c1, %pe, %win, %c0, %targetCount, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}

// A get raised from MPI_Get of several elements, retargeted: not atomic, so the
// buffer transfer rather than the one-element atomic fetch.  Before MPI_Get
// raised non-atomic it matched only the atomic fetch, and was rejected.
// CHECK-LABEL: func.func @translated_get
// CHECK:         %[[WIN:.*]] = builtin.unrealized_conversion_cast %arg3 : !spmd.win to !llvm.ptr
// CHECK:         llvm.call @shmem_int_get_nbi(%arg0, %[[WIN]], %arg1, %arg2)
func.func @translated_get(%dst: !llvm.ptr, %n: i64, %pe: i32, %win: !spmd.win) {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}> : () -> !spmd.datatype
  %e = "spmd.get"(%dst, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = false, usedModel = 1 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}
