// RUN: spmd-opt %s -lower-spmd-to-cuda -verify-diagnostics
//
// What -lower-spmd-to-cuda reports for the whole module before it rewrites
// anything.  The scope is the raiser's map, so an operation outside it is
// named as such rather than as something CUDA cannot do -- cudaMalloc and
// cudaFree exist, they are simply not raised, and the message says which
// direction the fix goes in.

// Operations no call of the map raises to.  cudaMalloc and cudaFree do exist;
// they sit commented out in the raiser's map, so nothing produces spmd.malloc
// for CUDA today and the lowering has nothing to emit for it.  CUDA has no
// communicator at all.  Every one of them is reported, not just the first:
// this is the whole-module pass a driver runs before it rewrites anything.
func.func @allocation(%size: i64) {
  // expected-error @+1 {{has no CUDA counterpart; -convert-apiCalls-to-spmd raises cudaMemcpy and cudaMemset, and the lowering covers what it raises}}
  %world = "spmd.commWorld"() <{usedModel = 4 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{has no CUDA counterpart; -convert-apiCalls-to-spmd raises cudaMemcpy and cudaMemset, and the lowering covers what it raises}}
  %addr, %e = "spmd.malloc"(%world, %size) <{operandSegmentSizes = array<i32: 1, 1, 0>, usedModel = 4 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  return
}

// A direction the program fixed to a value enum cudaMemcpyKind has no
// enumerator for.  cudaMemcpy would take it and fail at run time; the lowering
// says so where the program can still be fixed.
func.func @bad_direction(%dst: !llvm.ptr, %src: !llvm.ptr, %n: i64) {
  %kind = arith.constant 9 : i32
  // expected-error @+1 {{copies with kind 9, which is no direction of enum cudaMemcpyKind; cudaMemcpy would return cudaErrorInvalidMemcpyDirection}}
  %e = "spmd.memcpy"(%dst, %src, %n, %kind) <{usedModel = 4 : i32}> : (!llvm.ptr, !llvm.ptr, i64, i32) -> !spmd.error
  return
}
