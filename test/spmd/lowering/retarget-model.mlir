// RUN: spmd-opt %s -spmd-retarget-model="from=mpi to=shmem" | FileCheck %s --implicit-check-not="usedModel = 0"
//
// Re-tagging is what makes a translation possible: nothing else moves
// usedModel, so lowering an MPI-raised program to another model would convert
// nothing.  The operations keep their form; only the tag changes.  usedModel
// is a required attribute of every SPMD operation, so the handle producers
// move with the calls that use them without being closed over.

// CHECK-LABEL: func.func @collectives
// CHECK:         "spmd.commWorld"() <{usedModel = 1 : i32}>
// CHECK:         "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}>
// CHECK:         "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 1 : i32}>
// CHECK:         "spmd.barrier"(%{{.*}}) <{isBlocking = true, usedModel = 1 : i32}>
// CHECK:         "spmd.allreduce"({{.*}}) <{{{.*}}usedModel = 1 : i32}>
func.func @collectives(%send: !llvm.ptr, %recv: !llvm.ptr) {
  %count = arith.constant 4 : i64
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %e0 = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %e1 = "spmd.allreduce"(%comm, %send, %recv, %count, %dt, %op) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// A model the table does not name keeps its operations, tag and all.

// CHECK-LABEL: func.func @other_model
// CHECK:         "spmd.commWorld"() <{usedModel = 3 : i32}>
// CHECK:         "spmd.barrier"(%{{.*}}) <{isBlocking = true, usedModel = 3 : i32}>
func.func @other_model() {
  %comm = "spmd.commWorld"() <{usedModel = 3 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 3 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
