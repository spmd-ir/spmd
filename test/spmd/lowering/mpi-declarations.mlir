// RUN: spmd-opt %s -split-input-file -lower-spmd-to-mpi -verify-diagnostics | FileCheck %s
//
// A module may already declare the MPI functions the lowering calls.  Such a
// declaration is reused only when it has the signature the call needs: calling
// a function declared with other parameter types would pass the arguments
// under the wrong ABI, and a symbol that is not an LLVM function cannot be
// called by llvm.call at all.

// A matching declaration is reused, not duplicated.
// CHECK-LABEL: module
// CHECK:         llvm.func @MPI_Barrier(i32) -> i32
// CHECK-NOT:     llvm.func @MPI_Barrier
// CHECK:         llvm.call @MPI_Barrier(
llvm.func @MPI_Barrier(i32) -> i32
func.func @matching() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

llvm.func @MPI_Barrier(i64) -> i32
func.func @mismatching() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+2 {{'spmd.barrier' op needs @MPI_Barrier declared as '!llvm.func<i32 (i32)>', but the module declares it as '!llvm.func<i32 (i64)>'}}
  // expected-error @+1 {{failed to legalize operation 'spmd.barrier'}}
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}

// -----

func.func private @MPI_Barrier(i32) -> i32
func.func @not_llvm() {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  // expected-error @+2 {{'spmd.barrier' op needs @MPI_Barrier to be an LLVM function, but the module defines it as 'func.func'}}
  // expected-error @+1 {{failed to legalize operation 'spmd.barrier'}}
  %e = "spmd.barrier"(%world) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
