// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// Persistent and partitioned requests. An *_init call only creates the
// request: it raises to spmd.sendInit or spmd.recvInit, whose request is
// written back like any other, and MPI_Start (MPI_Pready for one partition)
// takes that request by value.
//
// The send modes are recorded as for MPI_Send (mpi-p2p.mlir): buffered and
// ready are attributes, synchronous is not. A partitioned init has the
// partition count (4) before the per-partition count (5); the peer is 3 and
// the tag 7.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK:         %[[W:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c4 = arith.constant 4 : i32
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %c5_i64 = arith.constant 5 : i64
  // MPI_INT, MPI_COMM_WORLD and MPI_INFO_NULL in the bundled MPICH ABI.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %infonull = arith.constant 469762048 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %buf = llvm.alloca %c5 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %init = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[R0:.*]], %{{.*}} = "spmd.sendInit"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBuffered = false, isReadyMode = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<5xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q0:.*]] = builtin.unrealized_conversion_cast %[[R0]] : !spmd.req to i32
  // CHECK: memref.store %[[Q0]]
  %0 = llvm.call @MPI_Send_init(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.sendInit"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBuffered = true, isReadyMode = false
  %1 = llvm.call @MPI_Bsend_init(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.sendInit"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBuffered = false, isReadyMode = false
  %2 = llvm.call @MPI_Ssend_init(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.sendInit"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBuffered = false, isReadyMode = true
  %3 = llvm.call @MPI_Rsend_init(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R4:.*]], %{{.*}} = "spmd.recvInit"(%[[W]], %{{.*}}, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<5xi32>, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q4:.*]] = builtin.unrealized_conversion_cast %[[R4]] : !spmd.req to i32
  // CHECK: memref.store %[[Q4]]
  %4 = llvm.call @MPI_Recv_init(%buf, %c5, %int, %c3, %c7, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  // MPI_Start reads the request out of the variable it is given.
  // CHECK: %[[L5:.*]] = memref.load %{{.*}}[%c0] : memref<1xi32>
  // CHECK: %[[H5:.*]] = builtin.unrealized_conversion_cast %[[L5]] : i32 to !spmd.req
  // CHECK: "spmd.start"(%[[H5]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.req) -> !spmd.error
  %5 = llvm.call @MPI_Start(%req) : (!llvm.ptr) -> i32

  // CHECK: %[[R6:.*]], %{{.*}} = "spmd.sendInit"(%[[W]], %{{.*}}, %c4_i32, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{isBuffered = false, isReadyMode = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<5xi32>, i32, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q6:.*]] = builtin.unrealized_conversion_cast %[[R6]] : !spmd.req to i32
  // CHECK: memref.store %[[Q6]]
  %6 = llvm.call @MPI_Psend_init(%buf, %c4, %c5_i64, %int, %c3, %c7, %world, %infonull, %req) : (!llvm.ptr, i32, i64, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[R7:.*]], %{{.*}} = "spmd.recvInit"(%[[W]], %{{.*}}, %c4_i32, %c5_i64, %{{.*}}, %c3_i32, %c7_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<5xi32>, i32, i64, !spmd.datatype, i32, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q7:.*]] = builtin.unrealized_conversion_cast %[[R7]] : !spmd.req to i32
  // CHECK: memref.store %[[Q7]]
  %7 = llvm.call @MPI_Precv_init(%buf, %c4, %c5_i64, %int, %c3, %c7, %world, %infonull, %req) : (!llvm.ptr, i32, i64, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  // MPI_Pready takes the request by value: partition 1 of it.
  // CHECK: %[[L8:.*]] = llvm.load
  // CHECK: %[[H8:.*]] = builtin.unrealized_conversion_cast %[[L8]] : i32 to !spmd.req
  // CHECK: "spmd.pReady"(%c1_i32, %[[H8]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (i32, !spmd.req) -> !spmd.error
  %handle = llvm.load %req : !llvm.ptr -> i32
  %8 = llvm.call @MPI_Pready(%c1, %handle) : (i32, i32) -> i32

  %fin = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Send_init(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Bsend_init(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Ssend_init(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Rsend_init(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Recv_init(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Start(!llvm.ptr) -> i32
llvm.func @MPI_Psend_init(!llvm.ptr, i32, i64, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Precv_init(!llvm.ptr, i32, i64, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Pready(i32, i32) -> i32
llvm.func @MPI_Finalize() -> i32
