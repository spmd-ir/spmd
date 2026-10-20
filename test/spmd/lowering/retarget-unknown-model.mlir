// RUN: spmd-opt %s -spmd-retarget-model="from=mpi to=sshmem" -verify-diagnostics

// expected-error @unknown {{'sshmem' in to= is not a programming model; known models are mpi, shmem, nvshmem, nccl, cuda, omp}}
func.func @unused() {
  return
}
