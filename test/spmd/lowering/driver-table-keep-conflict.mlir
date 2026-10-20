// RUN: spmd-opt %s -lower-spmd="models=mpi=keep,shmem=mpi" -verify-diagnostics
//
// Retargeting one model onto another that is kept cannot mean anything: once
// the OpenSHMEM operations carry MPI's usedModel, nothing tells them from the
// MPI operations that were already there, and the table asks for one set to be
// lowered and the other to be left alone.  Without this the kept operations
// were lowered too, silently.

// expected-error @unknown {{models= retargets 'shmem' to 'mpi', but 'mpi' is itself kept; the operations would be indistinguishable once they carry its usedModel}}
func.func @unused() {
  return
}
