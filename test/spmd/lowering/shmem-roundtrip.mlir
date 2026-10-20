// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd-to-shmem | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not="@shmem_sync_all"
//
// Raising and lowering compose: an OpenSHMEM program raised by
// -convert-apiCalls-to-spmd lowers back to the calls it was raised from, with
// the same typed names and the same arguments in the same places, and the
// spmd.parallel the raiser wraps around main is gone together with its
// self-matching barrier.  The typed names keep their signedness:
// shmem_uint_max_reduce does not come back as shmem_int_max_reduce, and the
// sized shmem_put32 not as a typed transfer.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         llvm.call @shmem_init() : () -> ()
// CHECK:         %[[ME:.*]] = llvm.call @shmem_my_pe() : () -> i32
// CHECK:         %[[BUF:.*]] = llvm.call @shmem_malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:         %[[BUF2:.*]] = llvm.call @shmem_malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:         llvm.call @shmem_int_sum_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_double_max_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_uint_max_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_uchar_and_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @shmem_int_put(%[[BUF2]], %[[BUF]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_uint_put(%[[BUF2]], %[[BUF]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_put32(%[[BUF2]], %[[BUF]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_long_get_nbi(%[[BUF]], %[[BUF2]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @shmem_quiet() : () -> ()
// CHECK:         llvm.call @shmem_barrier_all() : () -> ()
// CHECK:         llvm.call @shmem_free(%[[BUF]]) : (!llvm.ptr) -> ()
// CHECK:         llvm.call @shmem_finalize() : () -> ()
// CHECK:         llvm.return
llvm.mlir.global external @SHMEM_TEAM_WORLD() : !llvm.ptr
llvm.func @main() -> i32 {
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %c16 = arith.constant 16 : i64
  llvm.call @shmem_init() : () -> ()
  %me = llvm.call @shmem_my_pe() : () -> i32
  %n = llvm.call @shmem_n_pes() : () -> i32
  %buf = llvm.call @shmem_malloc(%c16) : (i64) -> !llvm.ptr
  %buf2 = llvm.call @shmem_malloc(%c16) : (i64) -> !llvm.ptr
  %wa = llvm.mlir.addressof @SHMEM_TEAM_WORLD : !llvm.ptr
  %w = llvm.load %wa : !llvm.ptr -> !llvm.ptr
  %r = llvm.call @shmem_int_sum_reduce(%w, %buf2, %buf, %c4_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  %r2 = llvm.call @shmem_double_max_reduce(%w, %buf2, %buf, %c1_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  %r3 = llvm.call @shmem_uint_max_reduce(%w, %buf2, %buf, %c4_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  %r4 = llvm.call @shmem_uchar_and_reduce(%w, %buf2, %buf, %c4_i64) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.call @shmem_int_put(%buf2, %buf, %c4_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_uint_put(%buf2, %buf, %c4_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_put32(%buf2, %buf, %c4_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_long_get_nbi(%buf, %buf2, %c1_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_quiet() : () -> ()
  llvm.call @shmem_barrier_all() : () -> ()
  llvm.call @shmem_free(%buf) : (!llvm.ptr) -> ()
  llvm.call @shmem_finalize() : () -> ()
  llvm.return %c0_i32 : i32
}
llvm.func @shmem_init()
llvm.func @shmem_my_pe() -> i32
llvm.func @shmem_n_pes() -> i32
llvm.func @shmem_malloc(i64) -> !llvm.ptr
llvm.func @shmem_free(!llvm.ptr)
llvm.func @shmem_int_sum_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_double_max_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uint_max_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_uchar_and_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_int_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_uint_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_put32(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_long_get_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_quiet()
llvm.func @shmem_barrier_all()
llvm.func @shmem_finalize()
