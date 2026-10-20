// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// The completion calls that take an *array* of requests.  MPI writes their
// out-parameters (index, outcount, flag) through pointers, so each one gets an
// alloca that is loaded straight back; the i1 flag of the test family is the
// int MPI wrote compared against zero.
//
// The request/status/index arrays reach MPI as bare pointers.  Raised IR
// carries them as !llvm.ptr already; spelled as memrefs here, they reach the
// call through enzymexla.memref2pointer, which -convert-polygeist-to-llvm
// lowers, instead of a malformed llvm.call.

// CHECK-LABEL: func.func @wait_any
// CHECK:         %[[STATUS:.*]] = llvm.alloca %{{.*}} x !llvm.array<5 x i32>
// CHECK:         %[[REQS:.*]] = "enzymexla.memref2pointer"(%arg0) : (memref<2x!spmd.req>) -> !llvm.ptr
// CHECK:         llvm.call @MPI_Waitany(%{{.*}}, %[[REQS]], %[[IDX:[0-9]+]], %[[STATUS]])
// CHECK:         llvm.load %[[IDX]]
func.func @wait_any(%reqs: memref<2x!spmd.req>) {
  %n = arith.constant 2 : i32
  %i, %st, %e = "spmd.waitAny"(%n, %reqs) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> (i32, !spmd.status, !spmd.error)
  return
}

// Without a status result MPI_STATUS_IGNORE goes in instead of a status buffer.
// CHECK-LABEL: func.func @wait_any_no_status
// CHECK-NOT:     llvm.array<5 x i32>
// CHECK:         llvm.call @MPI_Waitany(
func.func @wait_any_no_status(%reqs: memref<2x!spmd.req>) {
  %n = arith.constant 2 : i32
  %i, %e = "spmd.waitAny"(%n, %reqs) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> (i32, !spmd.error)
  return
}

// CHECK-LABEL: func.func @wait_some
// CHECK:         llvm.call @MPI_Waitsome(%{{[^,]+}}, %{{[^,]+}}, %[[OUT:[0-9]+]], %{{[^,]+}}, %{{[^,)]+}})
// CHECK:         llvm.load %[[OUT]]
func.func @wait_some(%reqs: memref<2x!spmd.req>, %idx: memref<2xi32>, %st: memref<2x!spmd.status>) {
  %n = arith.constant 2 : i32
  %out, %e = "spmd.waitSome"(%n, %reqs, %idx, %st) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>, memref<2xi32>, memref<2x!spmd.status>) -> (i32, !spmd.error)
  return
}

// CHECK-LABEL: func.func @test_all
// CHECK:         llvm.call @MPI_Testall(%{{[^,]+}}, %{{[^,]+}}, %[[FLAG:[0-9]+]], %{{[^,)]+}})
// CHECK:         %[[F:.*]] = llvm.load %[[FLAG]]
// CHECK:         %[[ZERO:.*]] = llvm.mlir.constant(0 : i32)
// CHECK:         llvm.icmp "ne" %[[F]], %[[ZERO]]
func.func @test_all(%reqs: memref<2x!spmd.req>, %st: memref<2x!spmd.status>) {
  %n = arith.constant 2 : i32
  %flag, %e = "spmd.testAll"(%n, %reqs, %st) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>, memref<2x!spmd.status>) -> (i1, !spmd.error)
  return
}

// CHECK-LABEL: func.func @test_any
// CHECK:         %[[STATUS:.*]] = llvm.alloca %{{.*}} x !llvm.array<5 x i32>
// CHECK:         llvm.call @MPI_Testany(%{{[^,]+}}, %{{[^,]+}}, %[[IDX:[0-9]+]], %[[FLAG:[0-9]+]], %[[STATUS]])
// CHECK:         llvm.load %[[IDX]]
// CHECK:         llvm.load %[[FLAG]]
// CHECK:         llvm.icmp "ne"
func.func @test_any(%reqs: memref<2x!spmd.req>) {
  %n = arith.constant 2 : i32
  %i, %flag, %st, %e = "spmd.testAny"(%n, %reqs) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> (i32, i1, !spmd.status, !spmd.error)
  return
}

// CHECK-LABEL: func.func @test_some
// CHECK:         llvm.call @MPI_Testsome(%{{[^,]+}}, %{{[^,]+}}, %[[OUT:[0-9]+]], %{{[^,]+}}, %{{[^,)]+}})
// CHECK:         llvm.load %[[OUT]]
func.func @test_some(%reqs: memref<2x!spmd.req>, %idx: memref<2xi32>, %st: memref<2x!spmd.status>) {
  %n = arith.constant 2 : i32
  %out, %e = "spmd.testSome"(%n, %reqs, %idx, %st) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>, memref<2xi32>, memref<2x!spmd.status>) -> (i32, !spmd.error)
  return
}

// spmd.waitAll already lowered before this change; it is here so the whole
// completion family is covered by one file.  With no status array it passes
// MPI_STATUSES_IGNORE.
// CHECK-LABEL: func.func @wait_all
// CHECK:         llvm.call @MPI_Waitall(
func.func @wait_all(%reqs: memref<2x!spmd.req>) {
  %n = arith.constant 2 : i32
  %e = "spmd.waitAll"(%n, %reqs) <{usedModel = 0 : i32}> : (i32, memref<2x!spmd.req>) -> !spmd.error
  return
}
