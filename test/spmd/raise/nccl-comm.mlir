// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @nccl"
//
// NCCL communicator queries and splitting, and NCCL's memory allocator. The
// queries are answered at once, so they block; what they answer is written
// back through the pointer the program passed. ncclMemAlloc names no
// communicator; the operation allocates on the world of NCCL's model group.
// Initialisation and ncclCommCount are in nccl-init-order.mlir.

// CHECK-LABEL: llvm.func @comm(
// CHECK-SAME:    %[[COMM:.*]]: !llvm.ptr)
// CHECK-DAG:     %[[RANK:.*]] = llvm.alloca %c1_i32 x i32
// CHECK-DAG:     %[[DEV:.*]] = llvm.alloca %c2_i32 x i32
// CHECK-DAG:     %[[NEW:.*]] = llvm.alloca %c1_i32 x !llvm.ptr
// CHECK-DAG:     %[[MEM:.*]] = llvm.alloca %c2_i32 x !llvm.ptr
llvm.func @comm(%comm: !llvm.ptr) {
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c9 = arith.constant 9 : i32
  %c7 = arith.constant 7 : i32
  %c64 = arith.constant 64 : i64
  %null = llvm.mlir.zero : !llvm.ptr
  %rankp = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %devp = llvm.alloca %c2 x i32 : (i32) -> !llvm.ptr
  %newcomm = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  %memp = llvm.alloca %c2 x !llvm.ptr : (i32) -> !llvm.ptr

  // CHECK: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[R:.*]], %{{.*}} = "spmd.getRankInComm"(%[[C0]]) <{usedModel = 3 : i32}>
  // CHECK: %[[RS:.*]] = "enzymexla.pointer2memref"(%[[RANK]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: memref.store %[[R]], %[[RS]][%c0] : memref<1xi32>
  %0 = llvm.call @ncclCommUserRank(%comm, %rankp) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[D:.*]], %{{.*}} = "spmd.getDeviceInComm"(%[[C1]]) <{usedModel = 3 : i32}>
  // CHECK-SAME: : (!spmd.comm) -> (i32, !spmd.error)
  // CHECK: %[[DS:.*]] = "enzymexla.pointer2memref"(%[[DEV]]) : (!llvm.ptr) -> memref<2xi32>
  // CHECK: memref.store %[[D]], %[[DS]][%c0] : memref<2xi32>
  %1 = llvm.call @ncclCommCuDevice(%comm, %devp) : (!llvm.ptr, !llvm.ptr) -> i32

  // Color 9, key 7; the configuration is not modelled.
  // CHECK: %[[C2:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[SPLIT:.*]], %{{.*}} = "spmd.commSplit"(%[[C2]], %c9_i32, %c7_i32) <{usedModel = 3 : i32}>
  // CHECK-SAME: : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  // CHECK: %[[NS:.*]] = "enzymexla.pointer2memref"(%[[NEW]]) : (!llvm.ptr) -> memref<1x!llvm.ptr>
  // CHECK: %[[NP:.*]] = builtin.unrealized_conversion_cast %[[SPLIT]] : !spmd.comm to !llvm.ptr
  // CHECK: memref.store %[[NP]], %[[NS]][%c0] : memref<1x!llvm.ptr>
  %2 = llvm.call @ncclCommSplit(%comm, %c9, %c7, %newcomm, %null) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[AW:.*]] = "spmd.commWorld"() <{usedModel = 3 : i32}>
  // CHECK: %[[ADDR:.*]], %{{.*}} = "spmd.malloc"(%[[AW]], %c64_i64) <{usedModel = 3 : i32}>
  // CHECK: %[[MS:.*]] = "enzymexla.pointer2memref"(%[[MEM]]) : (!llvm.ptr) -> memref<2x!llvm.ptr>
  // CHECK: memref.store %[[ADDR]], %[[MS]][%c0] : memref<2x!llvm.ptr>
  %3 = llvm.call @ncclMemAlloc(%memp, %c64) : (!llvm.ptr, i64) -> i32
  // CHECK: %[[P:.*]] = llvm.load %[[MEM]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[FW:.*]] = "spmd.commWorld"() <{usedModel = 3 : i32}>
  // CHECK: "spmd.free"(%[[FW]], %[[P]]) <{usedModel = 3 : i32}>
  %p = llvm.load %memp : !llvm.ptr -> !llvm.ptr
  %4 = llvm.call @ncclMemFree(%p) : (!llvm.ptr) -> i32
  llvm.return
}
llvm.func @ncclCommUserRank(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclCommCuDevice(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclCommSplit(!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclMemAlloc(!llvm.ptr, i64) -> i32
llvm.func @ncclMemFree(!llvm.ptr) -> i32
