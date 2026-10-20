// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @shmem_"
//
// OpenSHMEM team collectives. The team is the communicator (here
// SHMEM_TEAM_WORLD, which is the world), the element type comes from the
// name, and nelems is the count on both sides. OpenSHMEM passes the
// destination before the source; the operation takes send before receive.
//
// The destination holds 8 ints and the source 16, nelems is 5 and the root 3.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
llvm.mlir.global external @SHMEM_TEAM_WORLD() : !llvm.ptr
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c3 = arith.constant 3 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  %c5_i64 = arith.constant 5 : i64
  llvm.call @shmem_init() : () -> ()
  %dest = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
  %src = llvm.alloca %c16 x i32 : (i32) -> !llvm.ptr
  %wa = llvm.mlir.addressof @SHMEM_TEAM_WORLD : !llvm.ptr
  %team = llvm.load %wa : !llvm.ptr -> !llvm.ptr

  // shmem_int_broadcast(team, dest, source, nelems, root)
  // CHECK: "spmd.bcast"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %c3_i32) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %shmem_int_broadcast = llvm.call @shmem_int_broadcast(%team, %dest, %src, %c5_i64, %c3) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> i32
  // collect gathers varying counts, fcollect fixed ones; both raise to an
  // allgather of nelems.
  // CHECK: "spmd.allgather"(%{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, memref<8xi32>, i64, !spmd.datatype) -> !spmd.error
  %shmem_int_collect = llvm.call @shmem_int_collect(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: "spmd.allgather"(%{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, memref<8xi32>, i64, !spmd.datatype) -> !spmd.error
  %shmem_int_fcollect = llvm.call @shmem_int_fcollect(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: "spmd.alltoall"(%{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, i64, !spmd.datatype, memref<8xi32>, i64, !spmd.datatype) -> !spmd.error
  %shmem_int_alltoall = llvm.call @shmem_int_alltoall(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32

  // Reductions: every operation OpenSHMEM names, on a type that has all of
  // them. The operation keeps OpenSHMEM's spelling.
  // CHECK: %[[OP1:.*]] = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 1 : i32}>
  // CHECK: "spmd.allreduce"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[OP1]]) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %sum = llvm.call @shmem_uint_sum_reduce(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[OP2:.*]] = "spmd.constOpType"() <{stringAttr = "prod", usedModel = 1 : i32}>
  // CHECK: "spmd.allreduce"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[OP2]]) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %prod = llvm.call @shmem_uint_prod_reduce(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[OP3:.*]] = "spmd.constOpType"() <{stringAttr = "min", usedModel = 1 : i32}>
  // CHECK: "spmd.allreduce"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[OP3]]) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %min = llvm.call @shmem_uint_min_reduce(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[OP4:.*]] = "spmd.constOpType"() <{stringAttr = "max", usedModel = 1 : i32}>
  // CHECK: "spmd.allreduce"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[OP4]]) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %max = llvm.call @shmem_uint_max_reduce(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[OP5:.*]] = "spmd.constOpType"() <{stringAttr = "and", usedModel = 1 : i32}>
  // CHECK: "spmd.allreduce"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[OP5]]) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %and = llvm.call @shmem_uint_and_reduce(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[OP6:.*]] = "spmd.constOpType"() <{stringAttr = "or", usedModel = 1 : i32}>
  // CHECK: "spmd.allreduce"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[OP6]]) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %or = llvm.call @shmem_uint_or_reduce(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  // CHECK: %[[OP7:.*]] = "spmd.constOpType"() <{stringAttr = "xor", usedModel = 1 : i32}>
  // CHECK: "spmd.allreduce"(%{{.*}}, %{{.*}}, %{{.*}}, %c5_i64, %{{.*}}, %[[OP7]]) <{isBlocking = true, usedModel = 1 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<16xi32>, memref<8xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %xor = llvm.call @shmem_uint_xor_reduce(%team, %dest, %src, %c5_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32

  llvm.call @shmem_finalize() : () -> ()
  llvm.return %c0 : i32
}
llvm.func @shmem_init()
llvm.func @shmem_finalize()
llvm.func @shmem_int_broadcast(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i32) -> i32
llvm.func @shmem_int_collect(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_int_fcollect(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_int_alltoall(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_sum_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_prod_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_min_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_max_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_and_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_or_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_xor_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
