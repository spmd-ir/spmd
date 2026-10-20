// RUN: spmd-opt %s -lower-spmd="models=mpo=mpi" -verify-diagnostics
//
// What the table cannot mean.  Each spelling gets its own run, because the
// pass reports the table before it looks at the module, and a run stops at the
// first thing it cannot read.

// expected-error @unknown {{'mpo' in models= is not a programming model; known models are mpi, shmem, nvshmem, nccl, cuda, omp}}
func.func @unused() {
  return
}
