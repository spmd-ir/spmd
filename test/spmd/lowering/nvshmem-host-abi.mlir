// RUN: spmd-opt %s -split-input-file -nvshmem-to-host-abi -verify-diagnostics | FileCheck %s
//
// -nvshmem-to-host-abi maps a module written against the stand-in NVSHMEM
// headers to what NVIDIA's host library exports.  nvshmem_init,
// nvshmem_init_thread and nvshmem_finalize are inline functions of NVIDIA's
// header, so their declarations get a body over the host library's own
// initialisation; the calls stay as they are.  The stand-in's own inline stub
// bodies go, and a call of a routine the library lacks is reported.

// CHECK-LABEL: module
// CHECK-DAG:   llvm.func @nvshmemx_hostlib_init_attr(i32, !llvm.ptr) -> i32
// CHECK-DAG:   llvm.func @nvshmemx_hostlib_finalize()
// CHECK-DAG:   llvm.func @nvshmem_query_thread(!llvm.ptr)
// CHECK-DAG:   llvm.func @exit(i32)
//
// nvshmem_init: the host library with the bootstrap the environment selects,
// and exit(-1) when that fails, as NVIDIA's header does.
// CHECK:       llvm.func internal @nvshmem_init() {
// CHECK:         %[[FLAGS:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         %[[NOATTR:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         %[[STATUS:.*]] = llvm.call @nvshmemx_hostlib_init_attr(%[[FLAGS]], %[[NOATTR]]) : (i32, !llvm.ptr) -> i32
// CHECK:         %[[FAILED:.*]] = llvm.icmp "ne" %[[STATUS]], %{{.*}} : i32
// CHECK:         llvm.cond_br %[[FAILED]], ^[[FAIL:.*]], ^[[OK:.*]]
// CHECK:       ^[[FAIL]]:
// CHECK:         %[[MINUS1:.*]] = llvm.mlir.constant(-1 : i32) : i32
// CHECK:         llvm.call @exit(%[[MINUS1]]) : (i32) -> ()
// CHECK:         llvm.unreachable
// CHECK:       ^[[OK]]:
// CHECK:         llvm.return
//
// nvshmem_init_thread: the same, and the granted level from
// nvshmem_query_thread unless the program passed no place for it.  The
// requested level is not passed on: NVSHMEM provides SERIALIZED whatever it
// is asked for.
// CHECK:       llvm.func internal @nvshmem_init_thread(%[[REQUESTED:.*]]: i32, %[[PROVIDED:.*]]: !llvm.ptr) -> i32 {
// CHECK:         %[[STATUS:.*]] = llvm.call @nvshmemx_hostlib_init_attr(
// CHECK:         llvm.cond_br %{{.*}}, ^[[FAIL:.*]], ^[[OK:.*]]
// CHECK:       ^[[FAIL]]:
// CHECK:         llvm.call @exit(
// CHECK:         llvm.unreachable
// CHECK:       ^[[OK]]:
// CHECK:         %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         %[[WANTED:.*]] = llvm.icmp "ne" %[[PROVIDED]], %[[NULL]] : !llvm.ptr
// CHECK:         llvm.cond_br %[[WANTED]], ^[[QUERY:.*]], ^[[DONE:.*]]
// CHECK:       ^[[QUERY]]:
// CHECK:         llvm.call @nvshmem_query_thread(%[[PROVIDED]]) : (!llvm.ptr) -> ()
// CHECK:         llvm.br ^[[DONE]]
// CHECK:       ^[[DONE]]:
// CHECK:         llvm.return %[[STATUS]] : i32
//
// CHECK:       llvm.func internal @nvshmem_finalize() {
// CHECK:         llvm.call @nvshmemx_hostlib_finalize() : () -> ()
// CHECK:         llvm.return
//
// The calls are unchanged, and so is every routine the host library has.
// CHECK:       llvm.func @main
// CHECK:         llvm.call @nvshmem_init() : () -> ()
// CHECK:         llvm.call @nvshmem_init_thread(
// CHECK:         llvm.call @nvshmem_int_put(
// CHECK:         llvm.call @nvshmem_finalize() : () -> ()
// CHECK:       llvm.func @nvshmem_int_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @nvshmem_init()
llvm.func @nvshmem_init_thread(i32, !llvm.ptr) -> i32
llvm.func @nvshmem_finalize()
llvm.func @main(%p: !llvm.ptr, %n: i64, %pe: i32) -> i32 {
  %c2 = llvm.mlir.constant(2 : i32) : i32
  llvm.call @nvshmem_init() : () -> ()
  %s = llvm.call @nvshmem_init_thread(%c2, %p) : (i32, !llvm.ptr) -> i32
  llvm.call @nvshmem_int_put(%p, %p, %n, %pe) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @nvshmem_finalize() : () -> ()
  llvm.return %s : i32
}
llvm.func @nvshmem_int_put(!llvm.ptr, !llvm.ptr, i64, i32)

// -----

// A call of a routine NVIDIA's host library does not have, with the reason.
// A declaration nothing calls is left alone.
llvm.func @nvshmem_realloc(!llvm.ptr, i64) -> !llvm.ptr
llvm.func @nvshmem_int_wait_until(!llvm.ptr, i32, i32)
llvm.func @nvshmem_int_put_signal(!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32)
llvm.func @nvshmem_int_and_reduce(i32, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @nvshmem_int_test(!llvm.ptr, i32, i32) -> i32
llvm.func @hostless(%p: !llvm.ptr, %n: i64, %pe: i32) {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  // expected-error @+1 {{'nvshmem_realloc' is declared by the stand-in NVSHMEM headers, but NVIDIA's host library has no such routine: NVSHMEM has no realloc}}
  %q = llvm.call @nvshmem_realloc(%p, %n) : (!llvm.ptr, i64) -> !llvm.ptr
  // expected-error @+1 {{'nvshmem_int_wait_until' is declared by the stand-in NVSHMEM headers, but NVIDIA's host library has no such routine: it is a device routine in NVSHMEM, which only a GPU kernel can call}}
  llvm.call @nvshmem_int_wait_until(%p, %c0, %c0) : (!llvm.ptr, i32, i32) -> ()
  // expected-error @+1 {{no such routine: the host signals only through nvshmemx_<T>_put_signal_on_stream and nvshmemx_<T>_put_signal_nbi_on_stream, which take a CUDA stream}}
  llvm.call @nvshmem_int_put_signal(%p, %p, %n, %p, %n, %c0, %pe) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr, i64, i32, i32) -> ()
  // expected-error @+1 {{no such routine: NVSHMEM reduces bitwise over the unsigned and the fixed-width integer types only}}
  %r = llvm.call @nvshmem_int_and_reduce(%c0, %p, %p, %n) : (i32, !llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.return
}

// -----

// The stand-in defines a device stub for every host routine it declares
// (`__device__ inline int nvshmem_barrier(...) { return 0; }`), which a C++
// host compile turns into a linkonce_odr definition of the host routine.  A
// call the raiser did not raise would reach the stub; the pass drops its body
// so that the call reaches the library.  A definition of the program's own is
// left alone.
// CHECK:       llvm.func @nvshmem_barrier(i32) -> i32
// CHECK-NOT:   comdat
// CHECK:       llvm.func @nvshmem_helper_of_mine() -> i32 {
// CHECK:       llvm.func @stub_user
// CHECK:         llvm.call @nvshmem_barrier(
llvm.comdat @__llvm_global_comdat {
  llvm.comdat_selector @nvshmem_barrier any
}
llvm.func linkonce_odr @nvshmem_barrier(%team: i32) -> i32 comdat(@__llvm_global_comdat::@nvshmem_barrier) attributes {dso_local} {
  %0 = llvm.mlir.constant(0 : i32) : i32
  llvm.return %0 : i32
}
llvm.func @nvshmem_helper_of_mine() -> i32 {
  %0 = llvm.mlir.constant(7 : i32) : i32
  llvm.return %0 : i32
}
llvm.func @stub_user() -> i32 {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  %r = llvm.call @nvshmem_barrier(%c0) : (i32) -> i32
  %s = llvm.call @nvshmem_helper_of_mine() : () -> i32
  llvm.return %r : i32
}

// -----

// A GPU kernel calls NVSHMEM's device routines, so a device-only routine is
// what it should call; only host code is held to the host library.  The
// frontend links the device module in, so the kernel is an llvm.func
// compiled for the GPU.
// CHECK:       llvm.func @kernel_stub
// CHECK:         llvm.call @nvshmem_int_wait_until(
llvm.func @nvshmem_int_wait_until(!llvm.ptr, i32, i32)
llvm.func @kernel_stub(%p: !llvm.ptr) attributes {target_cpu = "sm_89"} {
  %c0 = llvm.mlir.constant(0 : i32) : i32
  llvm.call @nvshmem_int_wait_until(%p, %c0, %c0) : (!llvm.ptr, i32, i32) -> ()
  llvm.return
}

// -----

// A declaration of another type than the headers give the routine is not
// given a body it does not match.
// expected-error @+1 {{'nvshmem_init' is declared as '!llvm.func<i32 ()>', not as the NVSHMEM headers declare it ('!llvm.func<void ()>')}}
llvm.func @nvshmem_init() -> i32

// -----

// A module that uses no NVSHMEM is left as it is.
// CHECK-LABEL: llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
// CHECK-NEXT:  }
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
