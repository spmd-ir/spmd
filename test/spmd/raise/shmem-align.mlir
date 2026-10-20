// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @shmem_align"
//
// void *shmem_align(size_t alignment, size_t size) takes the alignment first,
// and spmd.malloc has the size first.  The CACTS entry used to read both in
// the operation's order, so a 1024-byte allocation aligned to 64 raised to one
// of 64 bytes aligned to 1024.  Nothing read the alignment, so nothing showed
// it until the OpenSHMEM lowering, which inverts the same entry, put them back.

// CHECK-LABEL: llvm.func @main
// CHECK-DAG:     %[[ALIGN:.*]] = arith.constant 64 : i64
// CHECK-DAG:     %[[SIZE:.*]] = arith.constant 1024 : i64
// CHECK:         "spmd.malloc"(%{{.*}}, %[[SIZE]], %[[ALIGN]])
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c64 = arith.constant 64 : i64
  %c1024 = arith.constant 1024 : i64
  llvm.call @shmem_init() : () -> ()
  %p = llvm.call @shmem_align(%c64, %c1024) : (i64, i64) -> !llvm.ptr
  llvm.call @shmem_free(%p) : (!llvm.ptr) -> ()
  llvm.call @shmem_finalize() : () -> ()
  llvm.return %c0_i32 : i32
}
llvm.func @shmem_init()
llvm.func @shmem_align(i64, i64) -> !llvm.ptr
llvm.func @shmem_free(!llvm.ptr)
llvm.func @shmem_finalize()
