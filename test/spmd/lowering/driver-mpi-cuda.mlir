// RUN: spmd-opt %s -lower-spmd | FileCheck %s --implicit-check-not='"spmd.'
//
// The driver's default table now lowers CUDA too, so an MPI program that also
// copies to and from the device comes out as calls all the way down.  Before
// this MR the CUDA operations were kept, the module still had an spmd.memset
// after -lower-spmd, and `spmd-verify compile` refused it: that refusal is
// what stopped every NCCL and NVSHMEM benchmark, which all call cudaMemset,
// from reaching a binary.
//
// The two models are lowered one after the other and the process-level
// spmd.parallel of MPI is spliced in before either of them, so the CUDA calls
// end up in the function body next to the MPI ones.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:         llvm.call @MPI_Init(
// CHECK:         llvm.call @cudaMemset(%{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i32, i64) -> i32
// CHECK:         llvm.call @cudaMemcpy(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, i64, i32) -> i32
// CHECK:         llvm.call @MPI_Finalize()
llvm.func @main() -> i32 {
  %mpiWorld = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %code = "spmd.parallel"(%mpiWorld) <{isProcessLevel = true, operandSegmentSizes = array<i32: 1, 0, 0, 0>, threadLevel = 0 : i32, usedModel = 0 : i32}> ({
    %c0 = llvm.mlir.constant(0 : i32) : i32
    %one = llvm.mlir.constant(1 : i32) : i32
    %bytes = llvm.mlir.constant(4096 : i64) : i64
    %single = "spmd.constThreadLevel"() <{level = "SINGLE", usedModel = 0 : i32}> : () -> !spmd.threadLevel
    %e0 = "spmd.init"(%single) <{usedModel = 0 : i32}> : (!spmd.threadLevel) -> !spmd.error
    %device = llvm.alloca %one x i8 : (i32) -> !llvm.ptr
    %host = llvm.alloca %one x i8 : (i32) -> !llvm.ptr
    %e1 = "spmd.memset"(%device, %c0, %bytes) <{usedModel = 4 : i32}> : (!llvm.ptr, i32, i64) -> !spmd.error
    %e2 = "spmd.memcpy"(%device, %host, %bytes, %one) <{usedModel = 4 : i32}> : (!llvm.ptr, !llvm.ptr, i64, i32) -> !spmd.error
    %e3 = "spmd.finalize"() <{usedModel = 0 : i32}> : () -> !spmd.error
    %e4 = "spmd.barrier"(%mpiWorld) <{isBlocking = true, isUniqueMatching = true, usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
    "spmd.parallel_yield"(%c0) : (i32) -> ()
  }) : (!spmd.comm) -> i32
  llvm.return %code : i32
}
