// RUN: spmd-opt %s -lower-spmd-to-nvshmem | FileCheck %s --implicit-check-not='"spmd.'
//
// One-sided transfers, as shmem-rma.mlir has them for OpenSHMEM.  A window is
// the symmetric address it is over, so a transfer's target address is the
// window itself; in NVSHMEM that address is on the GPU.

// A window allocated by spmd.winAlloc is nvshmem_malloc's symmetric address,
// and the transfers go to it.  A stride other than 1 is nvshmem_<T>_iput,
// which takes the target stride before the origin's.  The non-blocking forms
// are _nbi and hand out no request, so the wait and the free of the window
// each complete everything outstanding with nvshmem_quiet.
// CHECK-LABEL: func.func @put_get
// CHECK:         %[[C1:.*]] = arith.constant 1 : i64
// CHECK:         %[[BASE:.*]] = llvm.call @nvshmem_malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:         llvm.call @nvshmem_int_put(%[[BASE]], %arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_int_get_nbi(%arg0, %[[BASE]], %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_int_iput(%[[BASE]], %arg0, %[[C1]], %arg3, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_int_put_nbi(%[[BASE]], %arg0, %arg1, %arg2) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_quiet() : () -> ()
// CHECK:         llvm.call @nvshmem_quiet() : () -> ()
// CHECK-NEXT:    llvm.call @nvshmem_free(%[[BASE]]) : (!llvm.ptr) -> ()
func.func @put_get(%src: !llvm.ptr, %n: i64, %pe: i32, %stride: i64) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c8 = arith.constant 8 : i64
  %unit = arith.constant 1 : i32
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %base, %win, %e0 = "spmd.winAlloc"(%world, %c8, %unit) <{usedModel = 2 : i32}> : (!spmd.comm, i64, i32) -> (!llvm.ptr, !spmd.win, !spmd.error)
  %e1 = "spmd.put"(%src, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e2 = "spmd.get"(%src, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = false, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e3 = "spmd.put"(%src, %n, %i32, %stride, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %req, %e5 = "spmd.put"(%src, %n, %i32, %c1, %pe, %win, %c0, %n, %i32, %c1) <{isAtomic = false, isBlocking = false, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> (!spmd.req, !spmd.error)
  %e6 = "spmd.wait"(%req) <{usedModel = 2 : i32}> : (!spmd.req) -> !spmd.error
  %e7 = "spmd.winFree"(%win) <{usedModel = 2 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// The element type comes from the datatype: NVSHMEM's half and bfloat16,
// which OpenSHMEM does not have, uint for ui32, char for an i8 of unknown
// signedness, and long, the first spelling of si64.  A window spmd.winCreate
// made over the program's own symmetric memory is not freed with it.
// CHECK-LABEL: func.func @element_types
// CHECK:         llvm.call @nvshmem_half_put(%arg3, %arg0, %arg1, %arg2)
// CHECK:         llvm.call @nvshmem_bfloat16_put(%arg3, %arg0, %arg1, %arg2)
// CHECK:         llvm.call @nvshmem_uint_get(%arg0, %arg3, %arg1, %arg2)
// CHECK:         llvm.call @nvshmem_char_put(%arg3, %arg0, %arg1, %arg2)
// CHECK:         llvm.call @nvshmem_long_put(%arg3, %arg0, %arg1, %arg2)
// CHECK-NOT:     llvm.call @nvshmem_free
// CHECK:         llvm.call @nvshmem_quiet() : () -> ()
// CHECK-NOT:     llvm.call @nvshmem_free
// CHECK:         return
func.func @element_types(%src: !llvm.ptr, %n: i64, %pe: i32, %sym: !llvm.ptr) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %f16 = "spmd.constDatatype"() <{typeAttr = f16, usedModel = 2 : i32}> : () -> !spmd.datatype
  %bf16 = "spmd.constDatatype"() <{typeAttr = bf16, usedModel = 2 : i32}> : () -> !spmd.datatype
  %ui32 = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %i8 = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 2 : i32}> : () -> !spmd.datatype
  %si64 = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 2 : i32}> : () -> !spmd.datatype
  %win, %e0 = "spmd.winCreate"(%world, %sym, %c0_i32) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.put"(%src, %n, %f16, %c1, %pe, %win, %c0, %n, %f16, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e2 = "spmd.put"(%src, %n, %bf16, %c1, %pe, %win, %c0, %n, %bf16, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e3 = "spmd.get"(%src, %n, %ui32, %c1, %pe, %win, %c0, %n, %ui32, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e4 = "spmd.put"(%src, %n, %i8, %c1, %pe, %win, %c0, %n, %i8, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e5 = "spmd.put"(%src, %n, %si64, %c1, %pe, %win, %c0, %n, %si64, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e6 = "spmd.winFree"(%win) <{usedModel = 2 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// The fetching atomics return the element, which the lowering stores into the
// buffer the raiser gave the operation for it.
// CHECK-LABEL: func.func @atomics
// CHECK:         %[[FETCHED:.*]] = "enzymexla.memref2pointer"(%arg2) : (memref<1xi32>) -> !llvm.ptr
// CHECK:         %[[F:.*]] = llvm.call @nvshmem_int_atomic_fetch(%arg0, %arg1) : (!llvm.ptr, i32) -> i32
// CHECK:         llvm.store %[[F]], %[[FETCHED]] : i32, !llvm.ptr
// CHECK:         %[[FI:.*]] = llvm.call @nvshmem_int_atomic_fetch_inc(%arg0, %arg1) : (!llvm.ptr, i32) -> i32
// CHECK:         llvm.store %[[FI]], %[[FETCHED]] : i32, !llvm.ptr
func.func @atomics(%sym: !llvm.ptr, %pe: i32, %fetched: memref<1xi32>) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %i32 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %inc = "spmd.constOpType"() <{stringAttr = "inc", usedModel = 2 : i32}> : () -> !spmd.op
  %win, %e0 = "spmd.winCreate"(%world, %sym, %c0_i32) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  %e1 = "spmd.get"(%fetched, %c1, %i32, %c1, %pe, %win, %c0, %c1, %i32, %c1) <{isAtomic = true, isBlocking = true, usedModel = 2 : i32}> : (memref<1xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  %e2 = "spmd.getAccumulate"(%null, %c1, %i32, %c1, %fetched, %c1, %i32, %c1, %pe, %win, %c0, %c1, %i32, %c1, %inc) <{isAtomic = true, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, memref<1xi32>, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64, !spmd.op) -> !spmd.error
  %e3 = "spmd.winFree"(%win) <{usedModel = 2 : i32}> : (!spmd.win) -> !spmd.error
  return
}
