// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @nccl"
//
// Every NCCL collective. NCCL enqueues the work on a CUDA stream and returns:
// the operation does not block, and takes the stream (spmd.castStream of the
// program's cudaStream_t) as its last operand. The communicator is the
// program's ncclComm_t, cast. Buffers are device pointers and stay pointers.
//
// %sbuf and %rbuf are the function's third and fourth argument, the count is
// 5 and the root 3. ncclBcast has one buffer for both sides, like MPI_Bcast.

// CHECK-LABEL: llvm.func @collectives(
// CHECK-SAME:    %[[COMM:.*]]: !llvm.ptr, %[[STREAM:.*]]: !llvm.ptr, %[[S:.*]]: !llvm.ptr, %[[R:.*]]: !llvm.ptr)
llvm.func @collectives(%comm: !llvm.ptr, %stream: !llvm.ptr, %sbuf: !llvm.ptr, %rbuf: !llvm.ptr) {
  %c3 = arith.constant 3 : i32
  %n = arith.constant 5 : i64
  // ncclInt32 and ncclSum.
  %int = arith.constant 2 : i32
  %sum = arith.constant 0 : i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.allgather"(%[[C]], %[[S]], %c5_i64, %{{.*}}, %[[R]], %c5_i64, %{{.*}}, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r0 = llvm.call @ncclAllGather(%sbuf, %rbuf, %n, %int, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.alltoall"(%[[C]], %[[S]], %c5_i64, %{{.*}}, %[[R]], %c5_i64, %{{.*}}, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r1 = llvm.call @ncclAlltoAll(%sbuf, %rbuf, %n, %int, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.bcast"(%[[C]], %[[S]], %[[S]], %c5_i64, %{{.*}}, %c3_i32, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r2 = llvm.call @ncclBcast(%sbuf, %n, %int, %c3, %comm, %stream) : (!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.bcast"(%[[C]], %[[S]], %[[R]], %c5_i64, %{{.*}}, %c3_i32, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r3 = llvm.call @ncclBroadcast(%sbuf, %rbuf, %n, %int, %c3, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.gather"(%[[C]], %[[S]], %c5_i64, %{{.*}}, %[[R]], %c5_i64, %{{.*}}, %c3_i32, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r4 = llvm.call @ncclGather(%sbuf, %rbuf, %n, %int, %c3, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.scatter"(%[[C]], %[[S]], %c5_i64, %{{.*}}, %[[R]], %c5_i64, %{{.*}}, %c3_i32, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r5 = llvm.call @ncclScatter(%sbuf, %rbuf, %n, %int, %c3, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.reduce"(%[[C]], %[[S]], %[[R]], %c5_i64, %{{.*}}, %{{.*}}, %c3_i32, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r6 = llvm.call @ncclReduce(%sbuf, %rbuf, %n, %int, %sum, %c3, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // The count is that of each rank's block.
  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.reduceScatter"(%[[C]], %[[S]], %[[R]], %c5_i64, %{{.*}}, %{{.*}}, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r7 = llvm.call @ncclReduceScatter(%sbuf, %rbuf, %n, %int, %sum, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[C:.*]] = builtin.unrealized_conversion_cast %[[COMM]] : !llvm.ptr to !spmd.comm
  // CHECK: %[[ST:.*]] = "spmd.castStream"(%[[STREAM]]) <{usedModel = 3 : i32}>
  // CHECK: "spmd.allreduce"(%[[C]], %[[S]], %[[R]], %c5_i64, %{{.*}}, %{{.*}}, %[[ST]]) <{isBlocking = false, usedModel = 3 : i32}>
  // CHECK-SAME: -> !spmd.error
  %r8 = llvm.call @ncclAllReduce(%sbuf, %rbuf, %n, %int, %sum, %comm, %stream) : (!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  llvm.return
}
llvm.func @ncclAllGather(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclAlltoAll(!llvm.ptr, !llvm.ptr, i64, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclBcast(!llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclBroadcast(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclGather(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclScatter(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclReduce(!llvm.ptr, !llvm.ptr, i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclReduceScatter(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @ncclAllReduce(!llvm.ptr, !llvm.ptr, i64, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
