// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @shmem_"
// RUN: env SPMD_CHECK_API_COVERAGE=1 spmd-opt %s -convert-apiCalls-to-spmd -o /dev/null 2>&1 | FileCheck %s --check-prefix=COV
//
// The sized RMA routines shmem_put<N>_nbi / shmem_get<N>_nbi are derived from
// their blocking forms by expanding a "SIZE" template. The put entry used to be
// expanded with the "TYPENAME" helper, which left the literal name
// shmem_putSIZE_nbi<N> in the table, and the get entry was missing, so both
// calls stayed unconverted llvm.calls the analyses never saw.
//
// Trimmed from `spmd-verify check-dataRace emitPRESPMDIR` on
//   static long src = 1, dst = 0;
//   shmem_put64_nbi(&dst, &src, 1, pe); shmem_get8_nbi(&dst, &src, 1, pe);

// The coverage check fails the pass on any mapped name with no API entry that
// is not listed as a known miss.
// COV: [SPMD] coverage shmem:
// COV-NOT: UNRESOLVED

llvm.mlir.global internal @src(1 : i64) : i64
llvm.mlir.global internal @dst(0 : i64) : i64

// CHECK-LABEL: llvm.func @main
llvm.func @main() -> i32 {
  %c1_i64 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %dst = llvm.mlir.addressof @dst : !llvm.ptr
  %src = llvm.mlir.addressof @src : !llvm.ptr
  llvm.call @shmem_init() : () -> ()
  %me = llvm.call @shmem_my_pe() : () -> i32
  // CHECK: "spmd.put"
  // CHECK-SAME: isBlocking = false
  llvm.call @shmem_put64_nbi(%dst, %src, %c1_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  // CHECK: "spmd.get"
  // CHECK-SAME: isBlocking = false
  llvm.call @shmem_get8_nbi(%dst, %src, %c1_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_finalize() : () -> ()
  llvm.return %c0_i32 : i32
}
llvm.func @shmem_init()
llvm.func @shmem_my_pe() -> i32
llvm.func @shmem_put64_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_get8_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_finalize()
