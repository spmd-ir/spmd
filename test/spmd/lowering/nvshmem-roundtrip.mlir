// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd-to-nvshmem | FileCheck %s --implicit-check-not='"spmd.'
// RUN: spmd-opt %s -convert-apiCalls-to-spmd -lower-spmd | FileCheck %s --implicit-check-not='"spmd.'
//
// Raising and lowering compose: an NVSHMEM program raised by
// -convert-apiCalls-to-spmd lowers back to the calls it was raised from, with
// the same typed names and the same arguments in the same places, and the
// spmd.parallel the raiser wraps around main is gone together with its
// self-matching barrier, which would otherwise be a second nvshmem_sync_all.
// NVSHMEMX_TEAM_NODE comes back as the constant 2 it was.  The one name that
// changes is a spelling: nvshmem_uint32_and_reduce reduces the same type as
// nvshmem_uint_and_reduce, which the lowering prefers.  -lower-spmd's default
// table lowers NVSHMEM to itself too, to the same calls.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         %[[NODE:.*]] = llvm.mlir.constant(2 : i32) : i32
// CHECK:         llvm.call @nvshmem_init() : () -> ()
// CHECK:         llvm.call @nvshmem_team_my_pe(%[[NODE]]) : (i32) -> i32
// CHECK:         %[[ME:.*]] = llvm.call @nvshmem_my_pe() : () -> i32
// CHECK:         %[[BUF:.*]] = llvm.call @nvshmem_malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:         %[[BUF2:.*]] = llvm.call @nvshmem_malloc(%{{.*}}) : (i64) -> !llvm.ptr
// CHECK:         llvm.call @nvshmem_int_sum_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_double_max_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_uint_max_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_uint_and_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_half_sum_reduce(%{{.*}}, %[[BUF2]], %[[BUF]], %{{.*}}) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
// CHECK:         llvm.call @nvshmem_int_put(%[[BUF2]], %[[BUF]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_uint_put(%[[BUF2]], %[[BUF]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_bfloat16_put(%[[BUF2]], %[[BUF]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_long_get_nbi(%[[BUF]], %[[BUF2]], %{{.*}}, %[[ME]]) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
// CHECK:         llvm.call @nvshmem_quiet() : () -> ()
// CHECK:         llvm.call @nvshmem_barrier_all() : () -> ()
// CHECK:         llvm.call @nvshmem_sync_all() : () -> ()
// CHECK:         llvm.call @nvshmem_free(%[[BUF]]) : (!llvm.ptr) -> ()
// CHECK:         llvm.call @nvshmem_free(%[[BUF2]]) : (!llvm.ptr) -> ()
// CHECK:         llvm.call @nvshmem_finalize() : () -> ()
// CHECK-NOT:     llvm.call @nvshmem_sync_all
// CHECK:         llvm.return
llvm.func @main() -> i32 {
  %c1_i64 = arith.constant 1 : i64
  %c4_i64 = arith.constant 4 : i64
  %c0_i32 = arith.constant 0 : i32
  %c2_i32 = arith.constant 2 : i32
  %c16 = arith.constant 16 : i64
  llvm.call @nvshmem_init() : () -> ()
  %node = llvm.call @nvshmem_team_my_pe(%c2_i32) : (i32) -> i32
  %me = llvm.call @nvshmem_my_pe() : () -> i32
  %n = llvm.call @nvshmem_n_pes() : () -> i32
  %buf = llvm.call @nvshmem_malloc(%c16) : (i64) -> !llvm.ptr
  %buf2 = llvm.call @nvshmem_malloc(%c16) : (i64) -> !llvm.ptr
  %r = llvm.call @nvshmem_int_sum_reduce(%c0_i32, %buf2, %buf, %c4_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  %r2 = llvm.call @nvshmem_double_max_reduce(%c0_i32, %buf2, %buf, %c1_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  %r3 = llvm.call @nvshmem_uint_max_reduce(%c0_i32, %buf2, %buf, %c4_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  %r4 = llvm.call @nvshmem_uint32_and_reduce(%c0_i32, %buf2, %buf, %c4_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  %r5 = llvm.call @nvshmem_half_sum_reduce(%c0_i32, %buf2, %buf, %c4_i64) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.call @nvshmem_int_put(%buf2, %buf, %c4_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @nvshmem_uint_put(%buf2, %buf, %c4_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @nvshmem_bfloat16_put(%buf2, %buf, %c4_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @nvshmem_long_get_nbi(%buf, %buf2, %c1_i64, %me) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @nvshmem_quiet() : () -> ()
  llvm.call @nvshmem_barrier_all() : () -> ()
  llvm.call @nvshmem_sync_all() : () -> ()
  llvm.call @nvshmem_free(%buf) : (!llvm.ptr) -> ()
  llvm.call @nvshmem_free(%buf2) : (!llvm.ptr) -> ()
  llvm.call @nvshmem_finalize() : () -> ()
  %s = arith.addi %me, %node : i32
  llvm.return %s : i32
}
llvm.func @nvshmem_init()
llvm.func @nvshmem_finalize()
llvm.func @nvshmem_my_pe() -> i32
llvm.func @nvshmem_n_pes() -> i32
llvm.func @nvshmem_team_my_pe(i32) -> i32
llvm.func @nvshmem_malloc(i64) -> !llvm.ptr
llvm.func @nvshmem_free(!llvm.ptr)
llvm.func @nvshmem_int_sum_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_double_max_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_uint_max_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_uint32_and_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_half_sum_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_int_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_uint_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_bfloat16_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_long_get_nbi(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_quiet()
llvm.func @nvshmem_barrier_all()
llvm.func @nvshmem_sync_all()
