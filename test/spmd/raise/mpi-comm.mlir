// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// Communicator management, process topologies, MPI's memory allocation and
// MPI_Abort. The out-parameters (rank, size, the new communicator, the
// allocated base address) become results of the op, stored back where the
// program passed them; a handle is stored in the program's int form.
//
// MPI_Comm_free takes the communicator's address. It raises to a destroy of
// that address, cast to !spmd.comm, and does not model that MPI resets the
// variable to MPI_COMM_NULL (MR 11, part A).

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK-DAG:     %[[W:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK-DAG:     %[[RANK:.*]] = llvm.alloca %c1_i32 x i32
// CHECK-DAG:     %[[SIZE:.*]] = llvm.alloca %c1_i32 x i32
// CHECK-DAG:     %[[NEW:.*]] = llvm.alloca %c1_i32 x i32
// CHECK-DAG:     %[[DIMS:.*]] = llvm.alloca %c2_i32 x i32
// CHECK-DAG:     %[[PERIODS:.*]] = llvm.alloca %c3_i32 x i32
// CHECK-DAG:     %[[BASE:.*]] = llvm.alloca %c1_i32 x !llvm.ptr
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c2 = arith.constant 2 : i32
  %c3 = arith.constant 3 : i32
  %c9 = arith.constant 9 : i32
  %c64 = arith.constant 64 : i64
  // MPI_COMM_WORLD and MPI_INFO_NULL in the bundled MPICH ABI.
  %world = arith.constant 1140850688 : i32
  %infonull = arith.constant 469762048 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %rank = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %size = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %newcomm = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %dims = llvm.alloca %c2 x i32 : (i32) -> !llvm.ptr
  %periods = llvm.alloca %c3 x i32 : (i32) -> !llvm.ptr
  %base = llvm.alloca %c1 x !llvm.ptr : (i32) -> !llvm.ptr
  %init = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[R:.*]], %{{.*}} = "spmd.getRankInComm"(%[[W]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm) -> (i32, !spmd.error)
  // CHECK: %[[RS:.*]] = "enzymexla.pointer2memref"(%[[RANK]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: memref.store %[[R]], %[[RS]][%c0] : memref<1xi32>
  %0 = llvm.call @MPI_Comm_rank(%world, %rank) : (i32, !llvm.ptr) -> i32
  // CHECK: %[[S:.*]], %{{.*}} = "spmd.getSizeOfComm"(%[[W]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm) -> (i32, !spmd.error)
  // CHECK: %[[SS:.*]] = "enzymexla.pointer2memref"(%[[SIZE]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: memref.store %[[S]], %[[SS]][%c0] : memref<1xi32>
  %1 = llvm.call @MPI_Comm_size(%world, %size) : (i32, !llvm.ptr) -> i32

  // CHECK: %[[DUP:.*]], %{{.*}} = "spmd.commDup"(%[[W]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm) -> (!spmd.comm, !spmd.error)
  // CHECK: %[[DS:.*]] = "enzymexla.pointer2memref"(%[[NEW]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[DI:.*]] = builtin.unrealized_conversion_cast %[[DUP]] : !spmd.comm to i32
  // CHECK: memref.store %[[DI]], %[[DS]][%c0] : memref<1xi32>
  %2 = llvm.call @MPI_Comm_dup(%world, %newcomm) : (i32, !llvm.ptr) -> i32

  // Color 9, key 2.
  // CHECK: %[[SPLIT:.*]], %{{.*}} = "spmd.commSplit"(%[[W]], %c9_i32, %c2_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, i32, i32) -> (!spmd.comm, !spmd.error)
  // CHECK: %[[SI:.*]] = builtin.unrealized_conversion_cast %[[SPLIT]] : !spmd.comm to i32
  // CHECK: memref.store %[[SI]]
  %3 = llvm.call @MPI_Comm_split(%world, %c9, %c2, %newcomm) : (i32, i32, i32, !llvm.ptr) -> i32

  // CHECK: %[[FREED:.*]] = builtin.unrealized_conversion_cast %[[NEW]] : !llvm.ptr to !spmd.comm
  // CHECK: "spmd.commDestroy"(%[[FREED]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm) -> !spmd.error
  %4 = llvm.call @MPI_Comm_free(%newcomm) : (!llvm.ptr) -> i32

  // 9 processes over 2 dimensions; the op fills dims in place.
  // CHECK: "spmd.dimsCreate"(%c9_i32, %c2_i32, %[[DIMS]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (i32, i32, !llvm.ptr) -> !spmd.error
  %5 = llvm.call @MPI_Dims_create(%c9, %c2, %dims) : (i32, i32, !llvm.ptr) -> i32
  // CHECK: %[[CART:.*]], %{{.*}} = "spmd.cartCreate"(%[[W]], %c2_i32, %[[DIMS]], %[[PERIODS]], %c1_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, i32, !llvm.ptr, !llvm.ptr, i32) -> (!spmd.comm, !spmd.error)
  // CHECK: %[[CI:.*]] = builtin.unrealized_conversion_cast %[[CART]] : !spmd.comm to i32
  // CHECK: memref.store %[[CI]]
  %6 = llvm.call @MPI_Cart_create(%world, %c2, %dims, %periods, %c1, %newcomm) : (i32, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32

  // MPI_Alloc_mem and MPI_Free_mem name no communicator; the op allocates on
  // the world. The info argument is dropped.
  // CHECK: %[[AW:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
  // CHECK: %[[ADDR:.*]], %{{.*}} = "spmd.malloc"(%[[AW]], %c64_i64) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, i64) -> (!llvm.ptr, !spmd.error)
  // CHECK: %[[BS:.*]] = "enzymexla.pointer2memref"(%[[BASE]]) : (!llvm.ptr) -> memref<1x!llvm.ptr>
  // CHECK: memref.store %[[ADDR]], %[[BS]][%c0] : memref<1x!llvm.ptr>
  %7 = llvm.call @MPI_Alloc_mem(%c64, %infonull, %base) : (i64, i32, !llvm.ptr) -> i32
  // CHECK: %[[P:.*]] = llvm.load %[[BASE]] : !llvm.ptr -> !llvm.ptr
  // CHECK: %[[FW:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
  // CHECK: "spmd.free"(%[[FW]], %[[P]]) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, !llvm.ptr) -> !spmd.error
  %p = llvm.load %base : !llvm.ptr -> !llvm.ptr
  %8 = llvm.call @MPI_Free_mem(%p) : (!llvm.ptr) -> i32

  // CHECK: "spmd.abort"(%[[W]], %c9_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, i32) -> !spmd.error
  %9 = llvm.call @MPI_Abort(%world, %c9) : (i32, i32) -> i32

  %fin = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Comm_rank(i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_size(i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_dup(i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_split(i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_free(!llvm.ptr) -> i32
llvm.func @MPI_Dims_create(i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Cart_create(i32, i32, !llvm.ptr, !llvm.ptr, i32, !llvm.ptr) -> i32
llvm.func @MPI_Alloc_mem(i64, i32, !llvm.ptr) -> i32
llvm.func @MPI_Free_mem(!llvm.ptr) -> i32
llvm.func @MPI_Abort(i32, i32) -> i32
llvm.func @MPI_Finalize() -> i32
