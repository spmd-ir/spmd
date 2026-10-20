// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// Every blocking MPI collective and its nonblocking twin. They raise to the same
// operation, told apart by isBlocking and by the request the nonblocking one
// returns, which is written back into the MPI_Request the program passed.
//
// The send buffer holds 8 ints and the receive buffer 16, the send count is 2,
// the receive count 4 and the root 6: the operand types and constants show
// which argument went where. Datatype and reduction decoding have their own
// test, mpi-constants.mlir.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK:         %[[W:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK:         "spmd.init"
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c4 = arith.constant 4 : i32
  %c6 = arith.constant 6 : i32
  %c8 = arith.constant 8 : i32
  %c16 = arith.constant 16 : i32
  // MPI_INT, MPI_COMM_WORLD and MPI_SUM in the bundled MPICH ABI.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %sum = arith.constant 1476395011 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %sbuf = llvm.alloca %c8 x i32 : (i32) -> !llvm.ptr
  %rbuf = llvm.alloca %c16 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %init = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  // CHECK: "spmd.barrier"(%[[W]]) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm) -> !spmd.error
  %r0 = llvm.call @MPI_Barrier(%world) : (i32) -> i32
  // CHECK: %[[R1:.*]], %{{.*}} = "spmd.barrier"(%[[W]]) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q1:.*]] = builtin.unrealized_conversion_cast %[[R1]] : !spmd.req to i32
  // CHECK: memref.store %[[Q1]], %{{.*}}[%c0] : memref<1xi32>
  %r1 = llvm.call @MPI_Ibarrier(%world, %req) : (i32, !llvm.ptr) -> i32

  // MPI_Bcast has one buffer, which is both what the root sends and where
  // the others receive; the op takes it twice.
  // CHECK: "spmd.bcast"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %c6_i32) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<8xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %r2 = llvm.call @MPI_Bcast(%sbuf, %c2, %int, %c6, %world) : (!llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[R3:.*]], %{{.*}} = "spmd.bcast"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %c6_i32) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<8xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q3:.*]] = builtin.unrealized_conversion_cast %[[R3]] : !spmd.req to i32
  // CHECK: memref.store %[[Q3]], %{{.*}}[%c0] : memref<1xi32>
  %r3 = llvm.call @MPI_Ibcast(%sbuf, %c2, %int, %c6, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32

  // Reductions.
  // CHECK: "spmd.reduce"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c6_i32) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  %r4 = llvm.call @MPI_Reduce(%sbuf, %rbuf, %c2, %int, %sum, %c6, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> i32
  // CHECK: %[[R5:.*]], %{{.*}} = "spmd.reduce"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c6_i32) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q5:.*]] = builtin.unrealized_conversion_cast %[[R5]] : !spmd.req to i32
  // CHECK: memref.store %[[Q5]], %{{.*}}[%c0] : memref<1xi32>
  %r5 = llvm.call @MPI_Ireduce(%sbuf, %rbuf, %c2, %int, %sum, %c6, %world, %req) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.allreduce"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %r6 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %c2, %int, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[R7:.*]], %{{.*}} = "spmd.allreduce"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q7:.*]] = builtin.unrealized_conversion_cast %[[R7]] : !spmd.req to i32
  // CHECK: memref.store %[[Q7]], %{{.*}}[%c0] : memref<1xi32>
  %r7 = llvm.call @MPI_Iallreduce(%sbuf, %rbuf, %c2, %int, %sum, %world, %req) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.scan"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %r8 = llvm.call @MPI_Scan(%sbuf, %rbuf, %c2, %int, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[R9:.*]], %{{.*}} = "spmd.scan"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q9:.*]] = builtin.unrealized_conversion_cast %[[R9]] : !spmd.req to i32
  // CHECK: memref.store %[[Q9]], %{{.*}}[%c0] : memref<1xi32>
  %r9 = llvm.call @MPI_Iscan(%sbuf, %rbuf, %c2, %int, %sum, %world, %req) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.exscan"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %r10 = llvm.call @MPI_Exscan(%sbuf, %rbuf, %c2, %int, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[R11:.*]], %{{.*}} = "spmd.exscan"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q11:.*]] = builtin.unrealized_conversion_cast %[[R11]] : !spmd.req to i32
  // CHECK: memref.store %[[Q11]], %{{.*}}[%c0] : memref<1xi32>
  %r11 = llvm.call @MPI_Iexscan(%sbuf, %rbuf, %c2, %int, %sum, %world, %req) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32

  // MPI_Reduce_scatter_block's count is the size of each rank's block.
  // CHECK: "spmd.reduceScatter"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  %r12 = llvm.call @MPI_Reduce_scatter_block(%sbuf, %rbuf, %c2, %int, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[R13:.*]], %{{.*}} = "spmd.reduceScatter"(%[[W]], %{{[0-9]+}}, %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, memref<16xi32>, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q13:.*]] = builtin.unrealized_conversion_cast %[[R13]] : !spmd.req to i32
  // CHECK: memref.store %[[Q13]], %{{.*}}[%c0] : memref<1xi32>
  %r13 = llvm.call @MPI_Ireduce_scatter_block(%sbuf, %rbuf, %c2, %int, %sum, %world, %req) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32

  // Data movement: send and receive side each keep their own count and
  // datatype.
  // CHECK: "spmd.gather"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}, %c6_i32) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %r14 = llvm.call @MPI_Gather(%sbuf, %c2, %int, %rbuf, %c4, %int, %c6, %world) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[R15:.*]], %{{.*}} = "spmd.gather"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}, %c6_i32) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q15:.*]] = builtin.unrealized_conversion_cast %[[R15]] : !spmd.req to i32
  // CHECK: memref.store %[[Q15]], %{{.*}}[%c0] : memref<1xi32>
  %r15 = llvm.call @MPI_Igather(%sbuf, %c2, %int, %rbuf, %c4, %int, %c6, %world, %req) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.scatter"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}, %c6_i32) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype, i32) -> !spmd.error
  %r16 = llvm.call @MPI_Scatter(%sbuf, %c2, %int, %rbuf, %c4, %int, %c6, %world) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[R17:.*]], %{{.*}} = "spmd.scatter"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}, %c6_i32) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype, i32) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q17:.*]] = builtin.unrealized_conversion_cast %[[R17]] : !spmd.req to i32
  // CHECK: memref.store %[[Q17]], %{{.*}}[%c0] : memref<1xi32>
  %r17 = llvm.call @MPI_Iscatter(%sbuf, %c2, %int, %rbuf, %c4, %int, %c6, %world, %req) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.allgather"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype) -> !spmd.error
  %r18 = llvm.call @MPI_Allgather(%sbuf, %c2, %int, %rbuf, %c4, %int, %world) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
  // CHECK: %[[R19:.*]], %{{.*}} = "spmd.allgather"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q19:.*]] = builtin.unrealized_conversion_cast %[[R19]] : !spmd.req to i32
  // CHECK: memref.store %[[Q19]], %{{.*}}[%c0] : memref<1xi32>
  %r19 = llvm.call @MPI_Iallgather(%sbuf, %c2, %int, %rbuf, %c4, %int, %world, %req) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
  // CHECK: "spmd.alltoall"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype) -> !spmd.error
  %r20 = llvm.call @MPI_Alltoall(%sbuf, %c2, %int, %rbuf, %c4, %int, %world) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
  // CHECK: %[[R21:.*]], %{{.*}} = "spmd.alltoall"(%[[W]], %{{[0-9]+}}, %c2_i64, %{{[0-9]+}}, %{{[0-9]+}}, %c4_i64, %{{[0-9]+}}) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, memref<8xi32>, i64, !spmd.datatype, memref<16xi32>, i64, !spmd.datatype) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q21:.*]] = builtin.unrealized_conversion_cast %[[R21]] : !spmd.req to i32
  // CHECK: memref.store %[[Q21]], %{{.*}}[%c0] : memref<1xi32>
  %r21 = llvm.call @MPI_Ialltoall(%sbuf, %c2, %int, %rbuf, %c4, %int, %world, %req) : (!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
  
  %fin = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Barrier(i32) -> i32
llvm.func @MPI_Ibarrier(i32, !llvm.ptr) -> i32
llvm.func @MPI_Bcast(!llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Ibcast(!llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Reduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32) -> i32
llvm.func @MPI_Ireduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Iallreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Scan(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Iscan(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Exscan(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Iexscan(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Reduce_scatter_block(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Ireduce_scatter_block(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Gather(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Igather(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Scatter(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Iscatter(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Allgather(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
llvm.func @MPI_Iallgather(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Alltoall(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32) -> i32
llvm.func @MPI_Ialltoall(!llvm.ptr, i32, i32, !llvm.ptr, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
