// RUN: spmd-opt %s -lower-spmd="models=mpi=nccl" -verify-diagnostics
// RUN: spmd-opt %s -lower-spmd="models=nccl=shmem" -verify-diagnostics
// RUN: spmd-opt %s -lower-spmd="models=nccl=nvshmem" -verify-diagnostics
// RUN: spmd-opt %s -spmd-retarget-model="from=mpi to=nccl" -verify-diagnostics
//
// NCCL cannot be translated to or from either: its buffers are GPU memory, and
// every communication call takes a CUDA stream that a program of another model
// does not pass.  The checks of each operation would report a missing or
// left-over stream, but not an spmd.malloc, which has none: translated from
// MPI, it would hand the host the device memory of ncclMemAlloc.  So the whole
// translation is refused, before anything is re-tagged, in both directions.

// expected-error @unknown {{'nccl' cannot be translated to or from: its buffers are GPU memory, where MPI's and OpenSHMEM's are host memory, and every communication call takes a CUDA stream, which no MPI, OpenSHMEM or NVSHMEM program passes}}
func.func @unused() {
  return
}
