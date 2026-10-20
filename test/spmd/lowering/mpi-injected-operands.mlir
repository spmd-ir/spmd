// RUN: spmd-opt %s -split-input-file -lower-spmd-to-mpi -verify-diagnostics
//
// Operands the raiser injects because the MPI call has no argument for them.
// Read backwards they are conditions on the operation: it can only be that
// call if the operand still is what the raiser would have put there.  The
// lowering used to ignore them, so an operation that said something else
// became a call that did something else.

// MPI_Alloc_mem allocates from the process's own memory and takes no
// communicator, so the raiser gives spmd.malloc the world one.
func.func @malloc_on_sub_communicator(%size: i64, %color: i32, %key: i32) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %sub, %e0 = "spmd.commSplit"(%world, %color, %key) <{usedModel = 0 : i32}> : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  // expected-error @+1 {{'spmd.malloc' op has a communicator other than the world communicator}}
  %mem, %e1 = "spmd.malloc"(%sub, %size) <{usedModel = 0 : i32}> : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  // expected-error @+1 {{'spmd.free' op has a communicator other than the world communicator}}
  %e2 = "spmd.free"(%sub, %mem) <{usedModel = 0 : i32}> : (!spmd.comm, !llvm.ptr) -> !spmd.error
  return
}

// -----

// A thread level other than the one MPI_Init fixes is MPI_Init_thread, not an
// error: the injected operand only rules out the call that injects it.
func.func @multiple_is_init_thread() {
  %multiple = "spmd.constThreadLevel"() <{level = "MULTIPLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %e = "spmd.init"(%multiple) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// -----

// An injected constant is matched by its value, not by the operation holding
// it: a stride of 1 is a stride of 1 whichever constant-like operation the
// program spells it with.  The raiser writes arith.constant, but IR that has
// been through the LLVM dialect writes llvm.mlir.constant.
func.func @llvm_constant_stride(%origin: !llvm.ptr, %rank: i32, %disp: i64, %win: !spmd.win) {
  %dt = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %count = arith.constant 1 : i64
  %one = llvm.mlir.constant(1 : i64) : i64
  %e = "spmd.put"(%origin, %count, %dt, %one, %rank, %win, %disp, %count, %dt, %one) <{isAtomic = true, isBlocking = true, usedModel = 0 : i32}> : (!llvm.ptr, i64, !spmd.datatype, i64, i32, !spmd.win, i64, i64, !spmd.datatype, i64) -> !spmd.error
  return
}
