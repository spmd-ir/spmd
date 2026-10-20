// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_" --implicit-check-not="memref<1x!llvm.array"
//
// A request is usually one element of an array: MPI_Isend(..., &reqs[0]) and
// MPI_Irecv(..., &reqs[1]) on MPI_Request reqs[2], which clang spells as the
// alloca itself and as a byte GEP into it. The pointer resolves to the whole
// !llvm.array<2 x i32> allocation, so
//  - the request results were never written back: the store gave up on an
//    aggregate allocation holding a handle, silently, and MPI_Waitall saw
//    requests nothing had produced;
//  - MPI_Wait(&reqs[1]) loaded the whole array and cast it to !spmd.req.
// Both now go through a one-element i32 view at the element's own address.

// CHECK-LABEL: llvm.func @main
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  // MPI_INT and MPI_COMM_WORLD in the bundled MPICH ABI; MPI_STATUS(ES)_IGNORE.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %a = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  // CHECK: %[[REQS:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i32>
  %reqs = llvm.alloca %c1_i32 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // &reqs[0]
  // CHECK: %[[R0:.*]], %{{.*}} = "spmd.send"
  // CHECK: %[[M0:.*]] = "enzymexla.pointer2memref"(%[[REQS]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[R0]] : !spmd.req to i32
  // CHECK: memref.store %[[C0]], %[[M0]]
  %1 = llvm.call @MPI_Isend(%a, %c1_i32, %int, %c1_i32, %c0_i32, %world, %reqs) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  // &reqs[1]
  // CHECK: %[[GEP:.*]] = llvm.getelementptr inbounds %[[REQS]][4]
  // CHECK: %[[R1:.*]], %{{.*}} = "spmd.recv"
  // CHECK: %[[M1:.*]] = "enzymexla.pointer2memref"(%[[GEP]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[C1:.*]] = builtin.unrealized_conversion_cast %[[R1]] : !spmd.req to i32
  // CHECK: memref.store %[[C1]], %[[M1]]
  %reqs1 = llvm.getelementptr inbounds %reqs[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %2 = llvm.call @MPI_Irecv(%b, %c1_i32, %int, %c0_i32, %c0_i32, %world, %reqs1) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32

  // CHECK: %[[M2:.*]] = "enzymexla.pointer2memref"(%[[GEP]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[L2:.*]] = memref.load %[[M2]]
  // CHECK: %[[Q2:.*]] = builtin.unrealized_conversion_cast %[[L2]] : i32 to !spmd.req
  // CHECK: "spmd.wait"(%[[Q2]])
  %3 = llvm.call @MPI_Wait(%reqs1, %ignore) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: "spmd.waitAll"
  %4 = llvm.call @MPI_Waitall(%c2_i32, %reqs, %ignore) : (i32, !llvm.ptr, !llvm.ptr) -> i32
  %5 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Irecv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
