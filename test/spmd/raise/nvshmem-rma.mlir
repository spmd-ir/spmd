// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @nvshmem_"
//
// NVSHMEM's one-sided transfers from the host, tagged usedModel = 2. They
// raise as OpenSHMEM's do (shmem-rma.mlir): each transfer gets a window of its
// own over the remote address, the element type comes from the name, nelems
// is both counts.
//
// The remote address is `dest` (8 ints) for a put and `source` (16 ints) for a
// get; nelems is 5 and the PE 3. Not here: nvshmem_<T>_p and
// _atomic_compare_swap (they do not raise to valid operations, finding 6 in
// MR 16's description) and iget's strides (finding 7).

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK-DAG:     %[[NOORIGIN:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK-DAG:     %[[DEST:.*]] = llvm.alloca %c8_i32 x i32
// CHECK-DAG:     %[[SRC:.*]] = llvm.alloca %c16_i32 x i32
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %c2_i64 = arith.constant 2 : i64
  %c5_i64 = arith.constant 5 : i64
  %c6_i64 = arith.constant 6 : i64
  llvm.call @nvshmem_init() : () -> ()
  %dest = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
  %src = llvm.alloca %c16 x i32 : (i32) -> !llvm.ptr

  // CHECK: %[[O0:.*]] = "enzymexla.pointer2memref"(%[[SRC]]) : (!llvm.ptr) -> memref<16xi32>
  // CHECK: %[[T0:.*]] = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}>
  // CHECK: %[[D0:.*]] = "enzymexla.pointer2memref"(%[[DEST]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: %[[C0:.*]] = "spmd.commWorld"() <{usedModel = 2 : i32}>
  // CHECK: %[[W0:.*]], %{{.*}} = "spmd.winCreate"(%[[C0]], %[[D0]], %c0_i32) <{usedModel = 2 : i32}>
  // CHECK: "spmd.put"(%[[O0]], %c5_i64, %[[T0]], %c1_i64, %c3_i32, %[[W0]], %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}>
  llvm.call @nvshmem_int_put(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // CHECK: %[[O1:.*]] = "enzymexla.pointer2memref"(%[[DEST]]) : (!llvm.ptr) -> memref<8xi32>
  // CHECK: %[[D1:.*]] = "enzymexla.pointer2memref"(%[[SRC]]) : (!llvm.ptr) -> memref<16xi32>
  // CHECK: %[[W1:.*]], %{{.*}} = "spmd.winCreate"(%{{.*}}, %[[D1]], %c0_i32)
  // CHECK: "spmd.get"(%[[O1]], %c5_i64, %{{.*}}, %c1_i64, %c3_i32, %[[W1]], %c0_i64, %c5_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}>
  llvm.call @nvshmem_int_get(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // iput(dest, source, dst = 2, sst = 6, ...): the origin steps by sst.
  // CHECK: "spmd.put"(%{{.*}}, %c5_i64, %{{.*}}, %c6_i64, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %c2_i64) <{isAtomic = false, isBlocking = true
  // CHECK-SAME: : (memref<16xi32>,
  llvm.call @nvshmem_int_iput(%dest, %src, %c2_i64, %c6_i64, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32) -> ()
  // CHECK: "spmd.get"(%{{.*}}, %c5_i64, %{{.*}}, %{{.*}}, %c3_i32, %{{.*}}, %c0_i64, %c5_i64, %{{.*}}, %{{.*}}) <{isAtomic = false, isBlocking = true
  // CHECK-SAME: : (memref<8xi32>,
  llvm.call @nvshmem_int_iget(%dest, %src, %c2_i64, %c6_i64, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i64, i64, i32) -> ()

  // CHECK: "spmd.put"({{.*}}) <{isAtomic = false, isBlocking = false, usedModel = 2 : i32}>
  llvm.call @nvshmem_int_put_nbi(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // CHECK: "spmd.get"({{.*}}) <{isAtomic = false, isBlocking = false, usedModel = 2 : i32}>
  llvm.call @nvshmem_int_get_nbi(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  // CHECK: %[[TMP6:.*]] = memref.alloca() : memref<1xi32>
  // CHECK: "spmd.get"(%[[TMP6]], %c1_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64) <{isAtomic = false, isBlocking = true
  // CHECK: %[[G:.*]] = memref.load %[[TMP6]][%c0] : memref<1xi32>
  %g = llvm.call @nvshmem_int_g(%src, %c3) : (!llvm.ptr, i32) -> i32
  // CHECK: %[[TMP7:.*]] = memref.alloca() : memref<1xi32>
  // CHECK: "spmd.get"(%[[TMP7]], %c1_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64) <{isAtomic = true, isBlocking = true
  // CHECK: %[[F:.*]] = memref.load %[[TMP7]][%c0] : memref<1xi32>
  %f = llvm.call @nvshmem_int_atomic_fetch(%src, %c3) : (!llvm.ptr, i32) -> i32
  // CHECK: %[[TMP8:.*]] = memref.alloca() : memref<1xi32>
  // CHECK: %[[INC:.*]] = "spmd.constOpType"() <{stringAttr = "inc", usedModel = 2 : i32}>
  // CHECK: "spmd.getAccumulate"(%[[NOORIGIN]], %c1_i64, %{{.*}}, %c1_i64, %[[TMP8]], %c1_i64, %{{.*}}, %c1_i64, %c3_i32, %{{.*}}, %c0_i64, %c1_i64, %{{.*}}, %c1_i64, %[[INC]]) <{isAtomic = true, isBlocking = true
  // CHECK: %[[FI:.*]] = memref.load %[[TMP8]][%c0] : memref<1xi32>
  %fi = llvm.call @nvshmem_int_atomic_fetch_inc(%dest, %c3) : (!llvm.ptr, i32) -> i32

  // A type NVSHMEM has and OpenSHMEM does not: half is f16.
  // CHECK: %[[HALF:.*]] = "spmd.constDatatype"() <{typeAttr = f16, usedModel = 2 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c5_i64, %[[HALF]],
  llvm.call @nvshmem_half_put(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // And bfloat16, which is bf16.
  // CHECK: %[[BF16:.*]] = "spmd.constDatatype"() <{typeAttr = bf16, usedModel = 2 : i32}>
  // CHECK: "spmd.put"(%{{.*}}, %c5_i64, %[[BF16]],
  llvm.call @nvshmem_bfloat16_put(%dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()

  llvm.call @nvshmem_finalize() : () -> ()
  // CHECK: %[[S:.*]] = arith.addi %[[G]], %[[F]] : i32
  // CHECK: arith.addi %[[S]], %[[FI]] : i32
  %s = arith.addi %g, %f : i32
  %t = arith.addi %s, %fi : i32
  llvm.return %t : i32
}
llvm.func @nvshmem_init()
llvm.func @nvshmem_finalize()
llvm.func @nvshmem_int_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_int_get(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_int_iput(!llvm.ptr, !llvm.ptr, i64, i64, i64, i32)
llvm.func @nvshmem_int_iget(!llvm.ptr, !llvm.ptr, i64, i64, i64, i32)
llvm.func @nvshmem_int_put_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_int_get_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_int_g(!llvm.ptr, i32) -> i32
llvm.func @nvshmem_int_atomic_fetch(!llvm.ptr, i32) -> i32
llvm.func @nvshmem_int_atomic_fetch_inc(!llvm.ptr, i32) -> i32
llvm.func @nvshmem_half_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_bfloat16_put(!llvm.ptr, !llvm.ptr, i64, i32)
