// RUN: spmd-opt %s -lower-spmd | FileCheck %s --implicit-check-not='"spmd.' --implicit-check-not=unrealized_conversion_cast
// RUN: spmd-opt %s -lower-spmd="models=omp=keep" | FileCheck %s --check-prefix=KEEP
//
// A hybrid MPI+OpenMP program as -convert-apiCalls-to-spmd raises it: main in
// MPI's process-level spmd.parallel, and inside it an OpenMP thread team.  The
// default table lowers both.  A thread-number query outside every thread team
// is raised on the communicator of the process team around it -- the
// spmd.getParallelComm the raiser injects folds to that -- and is still the
// current team's to OpenMP: omp_get_thread_num().  omp=keep leaves the OpenMP
// operations alone.

// CHECK-LABEL: llvm.func @main
// CHECK:         llvm.call @MPI_Init
// CHECK:         %[[OUTSIDE:.*]] = llvm.call @omp_get_thread_num() : () -> i32
// CHECK:         omp.parallel num_threads(%{{.*}} : i32) {
// CHECK-NEXT:      %[[INSIDE:.*]] = llvm.call @omp_get_thread_num() : () -> i32
// CHECK-NEXT:      llvm.call @use(%[[OUTSIDE]], %[[INSIDE]])
// CHECK-NEXT:      omp.terminator
// CHECK-NEXT:    }
// CHECK:         llvm.call @MPI_Finalize

// KEEP-LABEL: llvm.func @main
// KEEP:         llvm.call @MPI_Init
// KEEP:         "spmd.getRankInComm"({{.*}}) <{usedModel = 5 : i32}>
// KEEP:         "spmd.parallel"({{.*}}) <{isProcessLevel = false
llvm.func @use(i32, i32)
llvm.func @main() -> i32 {
  %zero = arith.constant 0 : i32
  %four = arith.constant 4 : i32
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %code = "spmd.parallel"(%world) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 0 : i32}> ({
    %funneled = "spmd.constThreadLevel"() <{level = "FUNNELED", usedModel = 0 : i32}> : () -> !spmd.threadLevel
    %provided, %e0 = "spmd.init"(%funneled) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> (!spmd.threadLevel, !spmd.error)
    %outside, %e1 = "spmd.getRankInComm"(%world) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
    %ompWorld = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
    %team, %e2 = "spmd.commDup"(%ompWorld) <{usedModel = 5 : i32}> : (!spmd.comm) -> (!spmd.comm, !spmd.error)
    "spmd.parallel"(%team, %four) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 1, 0, 0>, usedModel = 5 : i32}> ({
      %inside, %e3 = "spmd.getRankInComm"(%team) <{usedModel = 5 : i32}> {isBlocking = true} : (!spmd.comm) -> (i32, !spmd.error)
      llvm.call @use(%outside, %inside) : (i32, i32) -> ()
      "spmd.parallel_yield"() : () -> ()
    }) : (!spmd.comm, i32) -> ()
    %e4 = "spmd.finalize"() <{usedModel = 0 : i32}> : () -> !spmd.error
    %e5 = "spmd.barrier"(%world) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%zero) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  llvm.return %code : i32
}
