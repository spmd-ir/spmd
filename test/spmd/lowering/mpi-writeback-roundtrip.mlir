// RUN: spmd-opt %s -split-input-file -convert-apiCalls-to-spmd -lower-spmd-to-mpi | FileCheck %s
//
// MPI writes the null handle back into a request it completes or frees and
// into a communicator or window it frees.  The raiser reads each of those
// handles out of the program's variable, so the lowering hands MPI that
// variable's address: afterwards the program reads MPI_REQUEST_NULL,
// MPI_COMM_NULL or MPI_WIN_NULL, as it would have without the round trip.  A
// stack slot in between would take the write, and the program would keep a
// handle MPI had already released.
//
// The raiser spells the communicator of MPI_Comm_free as the variable's
// address itself; the request and the window as a load from it.

// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:       ^bb1:
// CHECK:         %[[BUF:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[REQ:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[DUP:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[BASE:.*]] = llvm.alloca %{{.*}} x !llvm.ptr
// CHECK:         %[[WIN:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         llvm.call @MPI_Wait(%[[REQ]], %{{.*}})
// CHECK:         llvm.call @MPI_Request_free(%[[REQ]])
// CHECK:         llvm.call @MPI_Comm_free(%[[DUP]])
// CHECK:         llvm.call @MPI_Win_free(%[[WIN]])
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c1_i64 = arith.constant 1 : i64
  %c64_i64 = arith.constant 64 : i64
  %c4_i32 = arith.constant 4 : i32
  // MPI_INT, MPI_COMM_WORLD and MPI_INFO_NULL in the bundled MPICH ABI;
  // MPI_STATUS_IGNORE.
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %info = arith.constant 469762048 : i32
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %buf = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %req = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %dup = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %base = llvm.alloca %c1_i32 x !llvm.ptr : (i32) -> !llvm.ptr
  %win = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Isend(%buf, %c1_i32, %int, %c0_i32, %c0_i32, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  %2 = llvm.call @MPI_Wait(%req, %ignore) : (!llvm.ptr, !llvm.ptr) -> i32
  %3 = llvm.call @MPI_Irecv(%buf, %c1_i32, %int, %c0_i32, %c0_i32, %world, %req) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  %4 = llvm.call @MPI_Request_free(%req) : (!llvm.ptr) -> i32
  %5 = llvm.call @MPI_Comm_dup(%world, %dup) : (i32, !llvm.ptr) -> i32
  %6 = llvm.call @MPI_Comm_free(%dup) : (!llvm.ptr) -> i32
  %7 = llvm.call @MPI_Win_allocate(%c64_i64, %c4_i32, %info, %world, %base, %win) : (i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
  %8 = llvm.call @MPI_Win_free(%win) : (!llvm.ptr) -> i32
  %9 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Irecv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Request_free(!llvm.ptr) -> i32
llvm.func @MPI_Comm_dup(i32, !llvm.ptr) -> i32
llvm.func @MPI_Comm_free(!llvm.ptr) -> i32
llvm.func @MPI_Win_allocate(i64, i32, i32, i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Win_free(!llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32

// -----

// One element of an array of requests: the raiser reads reqs[1] through a
// one-element view at the element's own address, so MPI_Wait gets that
// address, and the MPI_Waitall after it sees MPI_REQUEST_NULL in reqs[1]
// instead of a request MPI already released.
// CHECK-LABEL: llvm.func @main() -> i32
// CHECK:       ^bb1:
// CHECK:         %[[REQS:.*]] = llvm.alloca %{{.*}} x !llvm.array<2 x i32>
// CHECK:         %[[ELEM:.*]] = llvm.getelementptr inbounds %[[REQS]][4]
// CHECK:         llvm.call @MPI_Wait(%[[ELEM]], %{{.*}})
// CHECK:         llvm.call @MPI_Waitall(%{{.*}}, %{{.*}}, %{{.*}})
llvm.func @main() -> i32 {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c2_i32 = arith.constant 2 : i32
  %c1_i64 = arith.constant 1 : i64
  %int = arith.constant 1275069445 : i32
  %world = arith.constant 1140850688 : i32
  %ignore = llvm.inttoptr %c1_i64 : i64 to !llvm.ptr
  %null = llvm.mlir.zero : !llvm.ptr
  %a = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %b = llvm.alloca %c1_i32 x i32 : (i32) -> !llvm.ptr
  %reqs = llvm.alloca %c1_i32 x !llvm.array<2 x i32> : (i32) -> !llvm.ptr
  %0 = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32
  %1 = llvm.call @MPI_Isend(%a, %c1_i32, %int, %c1_i32, %c0_i32, %world, %reqs) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  %reqs1 = llvm.getelementptr inbounds %reqs[4] : (!llvm.ptr) -> !llvm.ptr, i8
  %2 = llvm.call @MPI_Irecv(%b, %c1_i32, %int, %c0_i32, %c0_i32, %world, %reqs1) : (!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
  %3 = llvm.call @MPI_Wait(%reqs1, %ignore) : (!llvm.ptr, !llvm.ptr) -> i32
  %4 = llvm.call @MPI_Waitall(%c2_i32, %reqs, %ignore) : (i32, !llvm.ptr, !llvm.ptr) -> i32
  %5 = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0_i32 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Isend(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Irecv(!llvm.ptr, i32, i32, i32, i32, i32, !llvm.ptr) -> i32
llvm.func @MPI_Wait(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Waitall(i32, !llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Finalize() -> i32
