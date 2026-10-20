// RUN: spmd-opt %s -split-input-file -lower-spmd-to-nvshmem -verify-diagnostics
//
// Everything NVSHMEM cannot express is reported before the module is
// rewritten, each with its own reason.  What OpenSHMEM cannot express NVSHMEM
// cannot either (shmem-inexpressible.mlir); the reasons name NVSHMEM's
// routines.  NVSHMEM adds three: its symmetric memory is what nvshmem_malloc
// allocates on the GPU, so a global variable is not symmetric; a GPU kernel is
// not lowered; and the API the raiser reads has fewer routines.

// A rooted reduction: NVSHMEM's reductions deliver to every PE.
func.func @rooted_reduce(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64, %root: i32) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 2 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.reduce' op is a rooted reduction; every PE of an NVSHMEM team receives the result of nvshmem_<T>_<op>_reduce}}
  %e = "spmd.reduce"(%world, %sbuf, %rbuf, %n, %dt, %op, %root) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op, i32) -> !spmd.error
  return
}

// -----

// NVSHMEM has no long double.
func.func @no_type_name(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = f80, usedModel = 2 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 2 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op has a datatype f80 that no NVSHMEM call of this shape takes}}
  %e = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %dt, %op) <{isBlocking = true, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> !spmd.error
  return
}

// -----

// NVSHMEM's host collectives are all blocking.
func.func @nonblocking_collective(%sbuf: !llvm.ptr, %rbuf: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  %op = "spmd.constOpType"() <{stringAttr = "sum", usedModel = 2 : i32}> : () -> !spmd.op
  // expected-error @+1 {{'spmd.allreduce' op is a non-blocking collective, which NVSHMEM does not have}}
  %req, %e = "spmd.allreduce"(%world, %sbuf, %rbuf, %n, %dt, %op) <{isBlocking = false, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, !llvm.ptr, i64, !spmd.datatype, !spmd.op) -> (!spmd.req, !spmd.error)
  return
}

// -----

// Two-sided messaging, which NVSHMEM does not have.
func.func @send(%buf: !llvm.ptr, %n: i64, %dest: i32, %tag: i32) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  // expected-error @+1 {{'spmd.send' op is two-sided messaging, which NVSHMEM does not have; its transfers name an address on the symmetric heap, not a peer that receives}}
  %e = "spmd.send"(%world, %buf, %n, %dt, %dest, %tag) <{isBlocking = true, isBuffered = false, isReadyMode = false, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i64, !spmd.datatype, i32, i32) -> !spmd.error
  return
}

// -----

// A window over the stack, which no other PE can reach.
func.func @stack_window() {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c1 = arith.constant 1 : i32
  %buf = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  // expected-error @+1 {{'spmd.winCreate' op creates a window over the stack, which is not symmetric; NVSHMEM reaches only the symmetric heap nvshmem_malloc allocates from another PE}}
  %win, %e = "spmd.winCreate"(%world, %buf, %c1) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  return
}

// -----

// A window over a global variable.  OpenSHMEM makes global variables
// symmetric; NVSHMEM does not, since its symmetric heap is on the GPU.
llvm.mlir.global internal @counter(0 : i32) : i32
func.func @global_window() {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i32
  %buf = llvm.mlir.addressof @counter : !llvm.ptr
  // expected-error @+1 {{'spmd.winCreate' op creates a window over a global variable, which is not symmetric; NVSHMEM reaches only the symmetric heap nvshmem_malloc allocates from another PE}}
  %win, %e = "spmd.winCreate"(%world, %buf, %c0) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  return
}

// -----

// A window freed where the lowering cannot see what made it.
func.func @free_unknown_window(%win: !spmd.win) {
  // expected-error @+1 {{'spmd.winFree' op frees a window made neither by spmd.winAlloc nor by spmd.winCreate in sight; whether nvshmem_free has to free its memory depends on which, and guessing either way frees memory twice or leaks it}}
  %e = "spmd.winFree"(%win) <{usedModel = 2 : i32}> : (!spmd.win) -> !spmd.error
  return
}

// -----

// nvshmem_quiet() completes everything and reports on nothing.
func.func @wait_status(%req: !spmd.req) {
  // expected-error @+1 {{'spmd.wait' op asks for a status; nvshmem_quiet() reports nothing about the operation it completes}}
  %status, %e = "spmd.wait"(%req) <{usedModel = 2 : i32}> : (!spmd.req) -> (!spmd.status, !spmd.error)
  return
}

// -----

// A thread level NVSHMEM does not define.
func.func @thread_level() {
  // expected-error @+1 {{'spmd.constThreadLevel' op 'OVERSUBSCRIBED' is not an NVSHMEM thread support level}}
  %l = "spmd.constThreadLevel"() <{level = "OVERSUBSCRIBED", usedModel = 2 : i32}> : () -> !spmd.threadLevel
  %e = "spmd.init"(%l) <{usedModel = 2 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// -----

// The API the raiser reads has no realloc, and NVIDIA's library has none
// either, so nothing raises to this and there is no call to lower it to.  It
// is not one of the operations the check knows a reason for, so the
// conversion reports it.
func.func @realloc(%p: !llvm.ptr, %n: i64) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  // expected-error @+1 {{failed to legalize operation 'spmd.realloc'}}
  %q, %e = "spmd.realloc"(%world, %p, %n) <{usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i64) -> (!llvm.ptr, !spmd.error)
  return
}

// -----

// A window kept in an int: a window is a pointer.  A team, unlike OpenSHMEM's,
// is an int32_t and fits (nvshmem-management.mlir).
func.func @window_in_int(%sym: !llvm.ptr, %store: !llvm.ptr) {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i32
  %win, %e = "spmd.winCreate"(%world, %sym, %c0) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  // expected-error @+1 {{the program keeps an NVSHMEM window in 'i32', but a window is a pointer in this ABI; a variable declared for an MPI handle is too narrow for it}}
  %handle = builtin.unrealized_conversion_cast %win : !spmd.win to i32
  llvm.store %handle, %store : i32, !llvm.ptr
  return
}

// -----

// A call a GPU kernel makes is device-initiated NVSHMEM, which is not lowered:
// the lowering emits host calls, and nothing compiles the kernel.
gpu.module @kernels {
  gpu.func @kernel(%buf: !llvm.ptr, %n: i64, %pe: i32, %win: !spmd.win) kernel {
    %c0 = arith.constant 0 : i64
    %c1 = arith.constant 1 : i64
    %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
    // expected-error @+1 {{'spmd.put' op is inside a GPU kernel; the lowering emits host-initiated NVSHMEM calls only, and compiles no device code}}
    %e = "spmd.put"(%buf, %n, %dt, %c1, %pe, %win, %c0, %n, %dt, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
    gpu.return
  }
}

// -----

// The frontend links a CUDA program's device module into the host module, so
// a kernel is usually not in a gpu.module at all: it is an llvm.func compiled
// for the GPU, which a gpu.launch in host code calls.
llvm.func internal @kernel_stub(%buf: !llvm.ptr, %sym: !llvm.ptr, %n: i64, %pe: i32) attributes {target_cpu = "sm_89"} {
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c0_i32 = arith.constant 0 : i32
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 2 : i32}> : () -> !spmd.datatype
  // expected-error @+1 {{'spmd.winCreate' op is inside a GPU kernel}}
  %win, %e0 = "spmd.winCreate"(%world, %sym, %c0_i32) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  // expected-error @+1 {{'spmd.put' op is inside a GPU kernel; the lowering emits host-initiated NVSHMEM calls only, and compiles no device code}}
  %e = "spmd.put"(%buf, %n, %dt, %c1, %pe, %win, %c0, %n, %dt, %c1) <{isAtomic = false, isBlocking = true, usedModel = 2 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  llvm.return
}

// -----

// A window over an element of a global variable is still over the global.
llvm.mlir.global internal @array(dense<0> : tensor<8xi32>) : !llvm.array<8 x i32>
func.func @global_element_window() {
  %world = "spmd.commWorld"() <{usedModel = 2 : i32}> : () -> !spmd.comm
  %c0 = arith.constant 0 : i32
  %base = llvm.mlir.addressof @array : !llvm.ptr
  %elem = llvm.getelementptr %base[0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.array<8 x i32>
  // expected-error @+1 {{'spmd.winCreate' op creates a window over a global variable, which is not symmetric}}
  %win, %e = "spmd.winCreate"(%world, %elem, %c0) <{operandSegmentSizes = array<i32: 1, 1, 0, 1>, usedModel = 2 : i32}> : (!spmd.comm, !llvm.ptr, i32) -> (!spmd.win, !spmd.error)
  return
}
