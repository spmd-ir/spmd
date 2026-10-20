// RUN: spmd-opt %s -split-input-file -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// MPI_Init and MPI_Init_thread start the process-level parallelism, so raising
// them does more than replace the call: main's body moves into an
// spmd.parallel over the world communicator, which records the thread level
// the program asked for. Every return from main becomes a barrier that matches
// exactly once, then the yield of main's exit code: all ranks leave together,
// the way MPI_Finalize (collective) makes them.
//
// The MPI standard defines MPI_Init as MPI_Init_thread with MPI_THREAD_SINGLE.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         %[[OUTER:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK:         %[[P:.*]] = "spmd.parallel"(%[[OUTER]]) <{isProcessLevel = true, {{.*}}threadLevel = 0 : i32, usedModel = 0 : i32}>
// CHECK:           %[[L:.*]] = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}>
// CHECK:           "spmd.init"(%[[L]]) <{usedModel = 0 : i32}>
// CHECK-SAME:        : (!spmd.threadLevel) -> !spmd.error
// CHECK:           "spmd.finalize"() <{usedModel = 0 : i32}>
// CHECK:           "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}>
// CHECK:           "spmd.parallel_yield"(%c7_i32) : (i32) -> ()
// CHECK:         }) : (!spmd.comm) -> i32
// CHECK:         llvm.return %[[P]] : i32
llvm.func @main() -> i32 {
  %c7 = arith.constant 7 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c7 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32

// -----

// The level granted is written back through `provided`.
// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         "spmd.parallel"({{.*}}threadLevel = 1 : i32
// CHECK-DAG:       %[[PROVIDED:.*]] = llvm.alloca
// CHECK-DAG:       %[[L:.*]] = "spmd.constThreadLevel"() <{level = "FUNNELED", usedModel = 0 : i32}>
// CHECK:           %[[GOT:.*]], %{{.*}} = "spmd.init"(%[[L]])
// CHECK-SAME:        : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
// CHECK:           %[[INT:.*]] = "spmd.castThreadLevelToI32"(%[[GOT]]) : (!spmd.threadLevel) -> i32
// CHECK:           %[[SLOT:.*]] = "enzymexla.pointer2memref"(%[[PROVIDED]]) : (!llvm.ptr) -> memref<1xi32>
// CHECK:           memref.store %[[INT]], %[[SLOT]][%c0] : memref<1xi32>
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // MPI_THREAD_FUNNELED
  %funneled = arith.constant 1 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %provided = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init_thread(%null, %null, %funneled, %provided) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init_thread(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32

// -----

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         "spmd.parallel"({{.*}}threadLevel = 2 : i32
// CHECK:           "spmd.constThreadLevel"() <{level = "SERIALIZED", usedModel = 0 : i32}>
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // MPI_THREAD_SERIALIZED
  %serialized = arith.constant 2 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %provided = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init_thread(%null, %null, %serialized, %provided) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init_thread(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32

// -----

// MPI initialised in a function main calls, and a main without an exit code:
// the parallel is still around main's body, and it yields nothing.
// CHECK-LABEL: llvm.func @setup()
// CHECK:         "spmd.constThreadLevel"() <{level = "MULTIPLE", usedModel = 0 : i32}>
// CHECK:         "spmd.init"
// CHECK-NOT:     "spmd.parallel"
// CHECK:         llvm.return
// CHECK-LABEL: llvm.func @main()
// CHECK:         %[[OUTER:.*]] = "spmd.commWorld"
// CHECK:         "spmd.parallel"(%[[OUTER]]) <{{{.*}}threadLevel = 3 : i32
// CHECK:           llvm.call @setup()
// CHECK:           "spmd.finalize"
// CHECK:           "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, isUniqueMatching = true
// CHECK:           "spmd.parallel_yield"() : () -> ()
// CHECK:         }) : (!spmd.comm) -> ()
// CHECK:         llvm.return
llvm.func @setup() {
  %c1 = arith.constant 1 : i32
  // MPI_THREAD_MULTIPLE
  %multiple = arith.constant 3 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %provided = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init_thread(%null, %null, %multiple, %provided) : (!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
  llvm.return
}
llvm.func @main() {
  llvm.call @setup() : () -> ()
  %0 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return
}
llvm.func @MPI_Init_thread(!llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32

// -----

// A main that returns in two places, one of them before MPI_Finalize: each
// return gets its own exit barrier and yields its own exit code.
// CHECK-LABEL: llvm.func @main(
// CHECK:         %[[OUTER:.*]] = "spmd.commWorld"
// CHECK:         "spmd.parallel"(%[[OUTER]])
// CHECK:           llvm.cond_br %{{.*}}, ^[[EARLY:.*]], ^[[LATE:.*]]
// CHECK:         ^[[EARLY]]:
// CHECK-NEXT:      "spmd.barrier"(%{{.*}}) <{isBlocking = true, usedModel = 0 : i32}>
// CHECK-NEXT:      "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, isUniqueMatching = true
// CHECK-NEXT:      "spmd.parallel_yield"(%c1_i32)
// CHECK:         ^[[LATE]]:
// CHECK-NEXT:      "spmd.finalize"
// CHECK-NEXT:      "spmd.barrier"(%[[OUTER]]) <{isBlocking = true, isUniqueMatching = true
// CHECK-NEXT:      "spmd.parallel_yield"(%c0_i32)
llvm.func @main(%argc: i32) -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // MPI_COMM_WORLD in the bundled MPICH ABI.
  %world = arith.constant 1140850688 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %cond = arith.cmpi sgt, %argc, %c1 : i32
  llvm.cond_br %cond, ^early, ^late
^early:
  %1 = llvm.call @MPI_Barrier(%world) : (i32) -> i32
  llvm.return %c1 : i32
^late:
  %2 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Barrier(i32) -> i32
llvm.func @MPI_Finalize() -> i32

// -----

// No MPI_Init reachable from main: the calls are still raised, but nothing
// starts a parallel region.
// CHECK-LABEL: llvm.func @main() -> i32
// CHECK-NOT:     "spmd.parallel"
// CHECK:         "spmd.getRankInComm"
// CHECK:         llvm.return
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  // MPI_COMM_WORLD in the bundled MPICH ABI.
  %world = arith.constant 1140850688 : i32
  %rank = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Comm_rank(%world, %rank) : (i32, !llvm.ptr) -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
