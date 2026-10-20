// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// Matched probe and receive. MPI_Mprobe dequeues a message from source 3 with
// tag 7 and hands out its handle, which is written back into the program's
// MPI_Message; MPI_Mrecv and MPI_Imrecv read it from there. The receive names
// no communicator or envelope of its own: the message carries them.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK:         %[[W:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
// CHECK:         llvm.alloca %c5_i32 x i32
// CHECK-NEXT:    llvm.alloca %c1_i32 x i32
// CHECK-NEXT:    %[[MSG:.*]] = llvm.alloca %c1_i32 x i32
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c1 = arith.constant 1 : i32
  %c3 = arith.constant 3 : i32
  %c5 = arith.constant 5 : i32
  %c7 = arith.constant 7 : i32
  %c1_i64 = arith.constant 1 : i64
  // MPI_INT and MPI_COMM_WORLD in the bundled MPICH ABI; MPI_STATUS_IGNORE.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %buf = llvm.alloca %c5 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %msg = llvm.alloca %c1 x i32 : (i32) -> !llvm.ptr
  %init = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[M0:.*]], %{{.*}} = "spmd.mProbe"(%[[W]], %c3_i32, %c7_i32) <{usedModel = 0 : i32}>
  // CHECK-SAME: : (!spmd.comm, i32, i32) -> (!spmd.msg, !spmd.error)
  // CHECK: %[[S0:.*]] = "enzymexla.pointer2memref"(%[[MSG]]) : (!llvm.ptr) -> memref<1xi32>
  // CHECK: %[[C0:.*]] = builtin.unrealized_conversion_cast %[[M0]] : !spmd.msg to i32
  // CHECK: memref.store %[[C0]], %[[S0]][%c0] : memref<1xi32>
  %0 = llvm.call @MPI_Mprobe(%c3, %c7, %world, %msg, %ignore) : (i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[L1:.*]] = memref.load %{{.*}}[%c0] : memref<1xi32>
  // CHECK: %[[H1:.*]] = builtin.unrealized_conversion_cast %[[L1]] : i32 to !spmd.msg
  // CHECK: "spmd.mRecv"(%{{.*}}, %c5_i64, %{{.*}}, %[[H1]]) <{isBlocking = true, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<5xi32>, i64, !spmd.datatype, !spmd.msg) -> !spmd.error
  %1 = llvm.call @MPI_Mrecv(%buf, %c5, %int, %msg, %ignore) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  // CHECK: %[[L2:.*]] = memref.load %{{.*}}[%c0] : memref<1xi32>
  // CHECK: %[[H2:.*]] = builtin.unrealized_conversion_cast %[[L2]] : i32 to !spmd.msg
  // CHECK: %[[R2:.*]], %{{.*}} = "spmd.imRecv"(%{{.*}}, %c5_i64, %{{.*}}, %[[H2]]) <{isBlocking = false, usedModel = 0 : i32}>
  // CHECK-SAME: : (memref<5xi32>, i64, !spmd.datatype, !spmd.msg) -> (!spmd.req, !spmd.error)
  // CHECK: %[[Q2:.*]] = builtin.unrealized_conversion_cast %[[R2]] : !spmd.req to i32
  // CHECK: memref.store %[[Q2]]
  %2 = llvm.call @MPI_Imrecv(%buf, %c5, %int, %msg, %req) : (!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32

  %fin = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Mprobe(i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Mrecv(!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Imrecv(!llvm.ptr, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
