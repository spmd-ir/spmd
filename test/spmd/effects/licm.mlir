// RUN: spmd-opt %s -loop-invariant-code-motion | FileCheck %s
//
// -loop-invariant-code-motion hoists an op out of a loop when it is speculatable
// and memory-effect free. Hoisting a communication op out of a loop whose trip
// count is rank-dependent changes how many calls each rank makes: rank 0 below
// runs the body zero times, so hoisting hands it a call it never had. That is a
// deadlock, not a wrong answer, so nothing here may move.

// CHECK-LABEL: func.func @comm_op_in_rank_dependent_loop
// CHECK:         scf.for
// CHECK:           spmd.commDup
// CHECK:           spmd.commDestroy
func.func @comm_op_in_rank_dependent_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %ub = arith.index_cast %rank : i32 to index
  scf.for %i = %c0 to %ub step %c1 {
    %dup, %e0 = "spmd.commDup"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    %e1 = "spmd.commDestroy"(%dup) <{usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  }
  return
}

// CHECK-LABEL: func.func @collective_in_rank_dependent_loop
// CHECK:         scf.for
// CHECK:           spmd.allreduce
func.func @collective_in_rank_dependent_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %count = arith.constant 1 : i64
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %sum = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %ub = arith.index_cast %rank : i32 to index
  %send = memref.alloca() : memref<1xi32>
  %recv = memref.alloca() : memref<1xi32>
  scf.for %i = %c0 to %ub step %c1 {
    %e = "spmd.allreduce"(%world, %send, %recv, %count, %dt, %sum) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi32>, memref<1xi32>, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  }
  return
}

// A barrier moves no data, so no buffer effect holds it in place; the runtime
// and communicator effects must.
// CHECK-LABEL: func.func @barrier_in_rank_dependent_loop
// CHECK:         scf.for
// CHECK:           spmd.barrier
func.func @barrier_in_rank_dependent_loop() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %rank, %er = "spmd.getRankInComm"(%world) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %ub = arith.index_cast %rank : i32 to index
  scf.for %i = %c0 to %ub step %c1 {
    %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  }
  return
}
