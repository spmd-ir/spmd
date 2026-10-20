// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// The scalar out-parameters of the test/wait family become op results. Their
// spmd type comes from the parameter's kind in api.json, and a kind the switch
// did not know used to fall through to the C argument's type, !llvm.ptr, which
// no op declares:
//  - flag (kind BOOL) of MPI_Test/Testany: "result #0 must be 1-bit signless
//    integer, but got '!llvm.ptr'";
//  - index (kind INDEX_ARRAY, a single int despite the name) of
//    MPI_Testany/Waitany;
//  - outcount of MPI_Waitsome/Testsome, which has no kind at all, so the entry
//    names the type.
// array_of_indices has no kind either and reached spmd.waitSome as a raw
// pointer ("operand #2 must be memref of 32-bit signless integer values").

// CHECK-LABEL: llvm.func @main
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  // MPI_STATUS_IGNORE / MPI_STATUSES_IGNORE.
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %req = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %reqs = llvm.alloca %c1_i32 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %flag = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %index = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %outcount = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %indices = llvm.alloca %c1_i32 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[F0:.*]], %{{.*}} = "spmd.test"
  // CHECK-SAME: -> (i1, !spmd.error)
  // CHECK: %[[W0:.*]] = arith.extui %[[F0]] : i1 to i32
  // CHECK: memref.store %[[W0]]
  %1 = llvm.call @MPI_Test(%req, %flag, %ignore) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[I1:.*]], %[[F1:.*]], %{{.*}} = "spmd.testAny"
  // CHECK-SAME: -> (i32, i1, !spmd.error)
  // CHECK: memref.store %[[I1]]
  // CHECK: %[[W1:.*]] = arith.extui %[[F1]] : i1 to i32
  // CHECK: memref.store %[[W1]]
  %2 = llvm.call @MPI_Testany(%c2_i32, %reqs, %index, %flag, %ignore) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[I2:.*]], %{{.*}} = "spmd.waitAny"
  // CHECK-SAME: -> (i32, !spmd.error)
  // CHECK: memref.store %[[I2]]
  %3 = llvm.call @MPI_Waitany(%c2_i32, %reqs, %index, %ignore) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[IDX3:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.ptr to memref<?xi32>
  // CHECK: %[[O3:.*]], %{{.*}} = "spmd.waitSome"(%c2_i32, %{{.*}}, %[[IDX3]])
  // CHECK-SAME: : (i32, memref<?x!spmd.req>, memref<?xi32>) -> (i32, !spmd.error)
  // CHECK: memref.store %[[O3]]
  %4 = llvm.call @MPI_Waitsome(%c2_i32, %reqs, %outcount, %indices, %ignore) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[IDX4:.*]] = builtin.unrealized_conversion_cast %{{.*}} : !llvm.ptr to memref<?xi32>
  // CHECK: %[[O4:.*]], %{{.*}} = "spmd.testSome"(%c2_i32, %{{.*}}, %[[IDX4]])
  // CHECK-SAME: : (i32, memref<?x!spmd.req>, memref<?xi32>) -> (i32, !spmd.error)
  // CHECK: memref.store %[[O4]]
  %5 = llvm.call @MPI_Testsome(%c2_i32, %reqs, %outcount, %indices, %ignore) : (i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32

  %6 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Test(!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Testany(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Waitany(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Waitsome(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Testsome(i32, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
