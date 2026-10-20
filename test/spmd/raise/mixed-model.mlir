// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s
//
// A program that uses MPI and OpenSHMEM, and calls what the raiser does not
// map. Each model gets its own process-level spmd.parallel, world communicator
// and exit barrier, and its operations keep its usedModel (MPI 0, OpenSHMEM
// 1). The parallels nest: each model initialised wraps main once more, so the
// model initialised last is outermost -- OpenSHMEM here, as the lowering tests
// (lowering/driver-mixed-model.mlir, mpi-nested-process-parallel.mlir) start
// from.
//
// A call no map names stays a call, whichever model it belongs to:
// MPI_Get_count and printf. MPI_COMM_WORLD passed to it is the MPI world
// communicator, cast back to the program's handle.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         %[[SW:.*]] = "spmd.commWorld"() <{usedModel = 1 : i32}>
// CHECK:         %[[OUT:.*]] = "spmd.parallel"(%[[SW]]) <{isProcessLevel = true, {{.*}}usedModel = 1 : i32}> ({
// CHECK:           %[[MW:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK:           %[[IN:.*]] = "spmd.parallel"(%[[MW]]) <{isProcessLevel = true, {{.*}}usedModel = 0 : i32}> ({
// CHECK:             %[[MWORLD:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK:             %[[HANDLE:.*]] = builtin.unrealized_conversion_cast %[[MWORLD]] : !spmd.comm to i32
// CHECK:             "spmd.init"({{.*}}) <{usedModel = 0 : i32}>
// CHECK:             "spmd.init"({{.*}}) <{usedModel = 1 : i32}>
// CHECK:             "spmd.getRankInComm"(%[[MWORLD]]) <{usedModel = 0 : i32}>
// CHECK:             %[[SWORLD:.*]] = "spmd.commWorld"() <{usedModel = 1 : i32}>
// CHECK:             %[[ME:.*]], %{{.*}} = "spmd.getRankInComm"(%[[SWORLD]]) <{usedModel = 1 : i32}>
// CHECK:             llvm.call @MPI_Get_count(%{{.*}}, %[[HANDLE]], %{{.*}})
// CHECK:             llvm.call @printf(
// CHECK:             "spmd.finalize"() <{usedModel = 1 : i32}>
// CHECK:             "spmd.finalize"() <{usedModel = 0 : i32}>
// CHECK:             "spmd.barrier"(%[[MW]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}>
// CHECK:             "spmd.parallel_yield"(%[[ME]]) : (i32) -> ()
// CHECK:           }) : (!spmd.comm) -> i32
// CHECK:           "spmd.barrier"(%[[SW]]) <{isBlocking = true, isUniqueMatching = true, usedModel = 1 : i32}>
// CHECK:           "spmd.parallel_yield"(%[[IN]]) : (i32) -> ()
// CHECK:         }) : (!spmd.comm) -> i32
// CHECK:         llvm.return %[[OUT]] : i32
llvm.func @main() -> i32 {
  %c1 = arith.constant 1 : i32
  // MPI_COMM_WORLD in the bundled MPICH ABI.
  %world = arith.constant 1140850688 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %rank = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  llvm.call @shmem_init() : () -> ()
  %1 = llvm.call @MPI_Comm_rank(%world, %rank) : (i32, !llvm.ptr) -> i32
  %me = llvm.call @shmem_my_pe() : () -> i32
  %2 = llvm.call @MPI_Get_count(%null, %world, %rank) : (!llvm.ptr, i32, !llvm.ptr) -> i32
  %3 = llvm.call @printf(%null) vararg(!llvm.func<i32 (ptr, ...)>) : (!llvm.ptr) -> i32
  llvm.call @shmem_finalize() : () -> ()
  %4 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %me : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
llvm.func @MPI_Get_count(!llvm.ptr, i32, !llvm.ptr) -> i32
llvm.func @shmem_init()
llvm.func @shmem_finalize()
llvm.func @shmem_my_pe() -> i32
llvm.func @printf(!llvm.ptr, ...) -> i32
