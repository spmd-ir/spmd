// RUN: spmd-opt %s -lower-spmd="models=mpi=nvshmem" -verify-diagnostics
// RUN: spmd-opt %s -lower-spmd="models=nvshmem=shmem" -verify-diagnostics
// RUN: spmd-opt %s -lower-spmd="models=nvshmem=nccl" -verify-diagnostics
// RUN: spmd-opt %s -spmd-retarget-model="from=shmem to=nvshmem" -verify-diagnostics
//
// Every model has a lowering now, but NVSHMEM cannot be translated to or from:
// the symmetric heap and every buffer of an NVSHMEM program are GPU memory,
// where MPI's and OpenSHMEM's are host memory.  NCCL's are GPU memory too, but
// no lowering translates between the two (driver-table-nccl-translation.mlir
// refuses the other direction).  A translation is refused before anything is
// re-tagged, in both directions.

// expected-error @unknown {{'nvshmem' cannot be translated to or from: its buffers are GPU memory, where MPI's and OpenSHMEM's are host memory; NCCL's are GPU memory too, but no lowering translates between NVSHMEM and NCCL}}
func.func @unused() {
  return
}
