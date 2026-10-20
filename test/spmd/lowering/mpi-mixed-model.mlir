// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// Only MPI-tagged operations are this pass's.  An operation of another model
// keeps its spmd types, and so does the cast of its result:
// spmd.castSPMDErrorToI32 and spmd.castThreadLevelToI32 carry no usedModel of
// their own and go with the operation whose result they cast.

// CHECK-LABEL: func.func @mpi_and_shmem
// CHECK:         %[[MPIRC:.*]] = llvm.call @MPI_Barrier(
// CHECK:         %[[SHMEM:.*]] = "spmd.barrier"(%{{.*}}) <{isBlocking = true, usedModel = 1 : i32}>
// CHECK:         %[[SHMEMRC:.*]] = "spmd.castSPMDErrorToI32"(%[[SHMEM]])
// CHECK:         return %[[MPIRC]], %[[SHMEMRC]] : i32, i32
func.func @mpi_and_shmem() -> (i32, i32) {
  %mpiWorld = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e0 = "spmd.barrier"(%mpiWorld) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  %rc0 = "spmd.castSPMDErrorToI32"(%e0) : (!spmd.error) -> i32
  %shmemWorld = "spmd.commWorld"() <{usedModel = 1 : i32}> : () -> !spmd.comm
  %e1 = "spmd.barrier"(%shmemWorld) <{isBlocking = true, usedModel = 1 : i32}> : (!spmd.comm) -> !spmd.error
  %rc1 = "spmd.castSPMDErrorToI32"(%e1) : (!spmd.error) -> i32
  return %rc0, %rc1 : i32, i32
}

// CHECK-LABEL: func.func @shmem_thread_level
// CHECK:         %[[PROVIDED:.*]], %{{.*}} = "spmd.init"
// CHECK:         "spmd.castThreadLevelToI32"(%[[PROVIDED]])
func.func @shmem_thread_level() -> i32 {
  %multiple = "spmd.constThreadLevel"() <{level = "MULTIPLE", usedModel = 1 : i32}> : () -> !spmd.threadLevel
  %provided, %e = "spmd.init"(%multiple) <{usedModel = 1 : i32}> : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
  %level = "spmd.castThreadLevelToI32"(%provided) : (!spmd.threadLevel) -> i32
  return %level : i32
}
