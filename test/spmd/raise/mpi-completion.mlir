// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// The completion calls wait-test-results.mlir and request-array-element.mlir
// leave out: MPI_Wait on one request, MPI_Waitall and MPI_Testall over an
// array, and MPI_Request_free.
//
// One request is read out of the variable the program passes and handed to
// the op by value. An array of requests, or of statuses, is passed by address
// as a memref, and the op completes it in place.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK:         %[[REQ:.*]] = llvm.alloca %c1_i32 x i32
// CHECK-NEXT:    %[[REQS:.*]] = llvm.alloca %c1_i32 x !llvm.array<2 x i32>
// CHECK-NEXT:    %[[STATUSES:.*]] = llvm.alloca %c1_i32 x !llvm.array<10 x i32>
// CHECK-NEXT:    %[[FLAG:.*]] = llvm.alloca %c1_i32 x i32
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  // MPI_STATUS_IGNORE and MPI_STATUSES_IGNORE.
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %req = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %reqs = llvm.alloca %c1 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %statuses = llvm.alloca %c1 x !llvm.array<10 x i32> : (i32) -> !llvm.ptr
  %flag = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %init = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[S0:.*]] = "enzymexla.pointer2memref"(%[[REQ]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[L0:.*]] = memref.load %[[S0]][%c0] : memref<1xi32>
  // CHECK: %[[H0:.*]] = builtin.unrealized_conversion_cast %[[L0]] : i32 to !spmd.req
  // CHECK: "spmd.wait"(%[[H0]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.req) -> !spmd.error
  %0 = llvm.call @MPI_Wait(%req, %ignore) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[A1:.*]] = builtin.unrealized_conversion_cast %[[REQS]] : !llvm.ptr to memref<?x!spmd.req>
  // CHECK: %[[B1:.*]] = builtin.unrealized_conversion_cast %[[STATUSES]] : !llvm.ptr to memref<?x!spmd.status>
  // CHECK: "spmd.waitAll"(%c2_i32, %[[A1]], %[[B1]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (i32, memref<?x!spmd.req>, memref<?x!spmd.status>) -> !spmd.error
  %1 = llvm.call @MPI_Waitall(%c2, %reqs, %statuses) : (i32, !llvm.ptr, !llvm.ptr) -> i32

  // The flag is an i1 result, widened to the program's int.
  // CHECK: %[[A2:.*]] = builtin.unrealized_conversion_cast %[[REQS]] : !llvm.ptr to memref<?x!spmd.req>
  // CHECK: %[[F2:.*]], %{{.*}} = "spmd.testAll"(%c2_i32, %[[A2]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (i32, memref<?x!spmd.req>) -> (i1, !spmd.error)
  // CHECK: %[[W2:.*]] = arith.extui %[[F2]] : i1 to i32
  // CHECK: %[[S2:.*]] = "enzymexla.pointer2memref"(%[[FLAG]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: memref.store %[[W2]], %[[S2]][%c0] : memref<1xi32>
  %2 = llvm.call @MPI_Testall(%c2, %reqs, %flag, %ignore) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[S3:.*]] = "enzymexla.pointer2memref"(%[[REQ]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[L3:.*]] = memref.load %[[S3]][%c0] : memref<1xi32>
  // CHECK: %[[H3:.*]] = builtin.unrealized_conversion_cast %[[L3]] : i32 to !spmd.req
  // CHECK: "spmd.reqFree"(%[[H3]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.req) -> !spmd.error
  %3 = llvm.call @MPI_Request_free(%req) : (!llvm.ptr) -> i32

  %fin = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Testall(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Request_free(!llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
