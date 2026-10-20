// RUN: spmd-opt %s -lower-spmd="models=mpi.collectives=nccl" -verify-diagnostics
//
// Lowering one class of operations to another model needs a communicator of
// one model to bootstrap the other's, which no lowering does yet.  Reading the
// source as the whole model instead would lower more than was asked for.

// expected-error @unknown {{'mpi.collectives' in models= names a class of operations; only whole models can be retargeted so far, as in models="mpi=shmem"}}
func.func @unused() {
  return
}
