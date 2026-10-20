// RUN: spmd-opt %s -split-input-file -spmd-retarget-model="from=mpi to=shmem" -verify-diagnostics
//
// A handle stands for a runtime object of one model - an MPI communicator
// becomes an int, an OpenSHMEM team an i64 - so a value that ends up crossing
// two models after the retarget is reported rather than lowered into a call
// that reads the wrong thing.  A module that mixed two models before the
// retarget is left to the single-model passes, which lower one at a time; this
// is about the end the retarget moved.

func.func @handle_left_behind() {
  // expected-note @+1 {{handle produced here}}
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // The communicator moves to SHMEM with the operations of its own model; this
  // one is NCCL's and stays.
  // expected-error @+1 {{operand #0 is a 'shmem' handle, but the operation is tagged for 'nccl': the value crosses programming models}}
  %e = "spmd.barrier"(%comm) <{isBlocking = true, usedModel = 3 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

// The handle is followed back through the branch that passes it, so reaching
// its user as a block argument does not hide the crossing.

func.func @through_a_branch() {
  // expected-note @+1 {{handle produced here}}
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  cf.br ^bb1(%comm : !spmd.comm)
^bb1(%arg: !spmd.comm):
  // expected-error @+1 {{operand #0 is a 'shmem' handle, but the operation is tagged for 'nccl': the value crosses programming models}}
  %e = "spmd.barrier"(%arg) <{isBlocking = true, usedModel = 3 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
