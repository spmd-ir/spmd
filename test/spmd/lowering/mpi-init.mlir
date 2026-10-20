// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// spmd.init always carries the thread level it asks for.  The raiser records
// plain MPI_Init as a constant SINGLE with no $threadProvided, and that is the
// one shape that lowers back to MPI_Init; every other one is MPI_Init_thread,
// with MPICH's MPI_THREAD_* constants (SINGLE 0 ... MULTIPLE 3).

// CHECK-LABEL: func.func @init
// CHECK:         %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         llvm.call @MPI_Init(%[[NULL]], %[[NULL]]) : (!llvm.ptr, !llvm.ptr) -> i32
// CHECK:         llvm.call @MPI_Finalize() : () -> i32
func.func @init() {
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %e0 = "spmd.init"(%single) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  %e1 = "spmd.finalize"() <{usedModel = 0 : i32}> : () -> !spmd.error
  return
}

// The granted level is written through a pointer and read back; the casts of
// the level and of the error code have nothing left to do.
// CHECK-LABEL: func.func @init_thread
// CHECK:         %[[PROVIDED:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[MULTIPLE:.*]] = llvm.mlir.constant(3 : i32) : i32
// CHECK:         %[[NULL:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:         %[[RC:.*]] = llvm.call @MPI_Init_thread(%[[NULL]], %[[NULL]], %[[MULTIPLE]], %[[PROVIDED]]) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
// CHECK:         %[[LEVEL:.*]] = llvm.load %[[PROVIDED]] : !llvm.ptr -> i32
// CHECK:         return %[[LEVEL]], %[[RC]] : i32, i32
func.func @init_thread() -> (i32, i32) {
  %multiple = "spmd.constThreadLevel"() <{level = "MULTIPLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %provided, %e = "spmd.init"(%multiple) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
  %level = "spmd.castThreadLevelToI32"(%provided) : (!spmd.threadLevel) -> i32
  %rc = "spmd.castSPMDErrorToI32"(%e) : (!spmd.error) -> i32
  return %level, %rc : i32, i32
}

// Asking for SINGLE but reading the granted level back is MPI_Init_thread:
// MPI_Init has nowhere to report it.
// CHECK-LABEL: func.func @single_read_back
// CHECK:         %[[SINGLE:.*]] = llvm.mlir.constant(0 : i32) : i32
// CHECK:         llvm.call @MPI_Init_thread(%{{.*}}, %{{.*}}, %[[SINGLE]], %{{.*}})
// CHECK:         llvm.load
func.func @single_read_back() -> i32 {
  %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %provided, %e = "spmd.init"(%single) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
  %level = "spmd.castThreadLevelToI32"(%provided) : (!spmd.threadLevel) -> i32
  return %level : i32
}

// A level other than SINGLE is MPI_Init_thread even when nobody reads back what
// was granted; the pointer is still MPI's to write.
// CHECK-LABEL: func.func @funneled_unread
// CHECK:         %[[PROVIDED:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[FUNNELED:.*]] = llvm.mlir.constant(1 : i32) : i32
// CHECK:         llvm.call @MPI_Init_thread(%{{.*}}, %{{.*}}, %[[FUNNELED]], %[[PROVIDED]])
// CHECK-NOT:     llvm.load
// CHECK:         return
func.func @funneled_unread() {
  %funneled = "spmd.constThreadLevel"() <{level = "FUNNELED", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  %e = "spmd.init"(%funneled) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// A level that is not a constant reaches MPI_Init_thread as the int it lowers
// to.
// CHECK-LABEL: func.func @level_argument
// CHECK:         %[[LEVEL:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.threadLevel to i32
// CHECK:         llvm.call @MPI_Init_thread(%{{.*}}, %{{.*}}, %[[LEVEL]], %{{.*}})
func.func @level_argument(%level: !spmd.threadLevel) {
  %e = "spmd.init"(%level) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @serialized
// CHECK:         llvm.mlir.constant(2 : i32) : i32
func.func @serialized() -> !spmd.threadLevel {
  %serialized = "spmd.constThreadLevel"() <{level = "SERIALIZED", usedModel = 0 : i32}> : () -> !spmd.threadLevel
  return %serialized : !spmd.threadLevel
}
