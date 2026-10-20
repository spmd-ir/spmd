// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s --implicit-check-not='"spmd.'
//
// MPI_Bcast has one buffer and spmd.bcast two, which the raiser reads from the
// same argument.  It makes each of them a view of its own, so an in-place
// broadcast raised from IR that no -cse has run on has two
// enzymexla.pointer2memref of one address.  They are the same buffer, and the
// broadcast lowers back to MPI_Bcast; it used to be reported as having distinct
// send and receive buffers.

// A view of a pointer passed back to MPI is that pointer.
// CHECK-LABEL: func.func @bcast_two_views
// CHECK:         llvm.call @MPI_Bcast(%arg0,
func.func @bcast_two_views(%buf: !llvm.ptr, %root: i32) {
  %n = arith.constant 128 : i64
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}> : () -> !spmd.datatype
  %send = "enzymexla.pointer2memref"(%buf) : (!llvm.ptr) -> memref<1xi8>
  %recv = "enzymexla.pointer2memref"(%buf) : (!llvm.ptr) -> memref<1xi8>
  %e = "spmd.bcast"(%world, %send, %recv, %n, %dt, %root) <{isBlocking = true, usedModel = 0 : i32}> : (!spmd.comm, memref<1xi8>, memref<1xi8>, i64, !spmd.datatype, i32) -> !spmd.error
  return
}
