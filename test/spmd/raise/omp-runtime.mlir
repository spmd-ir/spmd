// RUN: spmd-opt %s -omp-to-spmd -canonicalize -convert-apiCalls-to-spmd -canonicalize | FileCheck %s --implicit-check-not="llvm.call @omp_get_thread_num" --implicit-check-not="llvm.call @omp_get_num_threads"
//
// The OpenMP runtime calls the raiser maps, in the order spmd-verify runs the
// passes. Inside a parallel region a thread is a rank of the region's team:
// omp_get_thread_num is the rank and omp_get_num_threads the size of the
// communicator spmd.parallel runs over, tagged usedModel = 5.
//
// omp_get_max_threads is not mapped and stays a call.

// CHECK-LABEL: llvm.func @main
// CHECK:         %[[SLOT:.*]] = llvm.alloca
// CHECK:         %[[TEAM:.*]], %{{.*}} = "spmd.commDup"
// CHECK:         "spmd.parallel"(%[[TEAM]], %c4_i32) <{isProcessLevel = false,
// CHECK:           %[[RANK:.*]], %{{.*}} = "spmd.getRankInComm"(%[[TEAM]]) <{usedModel = 5 : i32}>
// CHECK:           %[[SIZE:.*]], %{{.*}} = "spmd.getSizeOfComm"(%[[TEAM]]) <{usedModel = 5 : i32}>
// CHECK:           %[[SUM:.*]] = arith.addi %[[RANK]], %[[SIZE]] : i32
// CHECK:           llvm.store %[[SUM]], %[[SLOT]]
// CHECK:         llvm.call @omp_get_max_threads()
llvm.func @main() -> i32 {
  %c1 = arith.constant 1 : i32
  %c4 = arith.constant 4 : i32
  %slot = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  omp.parallel num_threads(%c4 : i32) {
    %t = llvm.call @omp_get_thread_num() : () -> i32
    %n = llvm.call @omp_get_num_threads() : () -> i32
    %s = arith.addi %t, %n : i32
    llvm.store %s, %slot : i32, !llvm.ptr
    omp.terminator
  }
  %max = llvm.call @omp_get_max_threads() : () -> i32
  llvm.return %max : i32
}
llvm.func @omp_get_num_threads() -> i32
llvm.func @omp_get_thread_num() -> i32
llvm.func @omp_get_max_threads() -> i32
