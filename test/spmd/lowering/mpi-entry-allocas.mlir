// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// The storage for an out-parameter, a status or a handle taken by address is
// allocated in the entry block of the function.  An alloca next to a call
// inside a loop would grow the stack on every iteration: nothing frees it
// before the function returns.  Here the rank, the request of the
// non-blocking barrier, the request MPI_Wait takes by address and the status
// it writes all get their storage before the loop's first branch, and the
// loop body holds only the calls.

// CHECK-LABEL: func.func @loop
// CHECK:         llvm.alloca %{{.*}} x !llvm.array<5 x i32>
// CHECK:         llvm.alloca %{{.*}} x i32
// CHECK:         llvm.alloca %{{.*}} x i32
// CHECK:         llvm.alloca %{{.*}} x i32
// CHECK:         llvm.br ^bb1
// CHECK-NOT:     llvm.alloca
// CHECK:         llvm.call @MPI_Comm_rank(
// CHECK-NOT:     llvm.alloca
// CHECK:         llvm.call @MPI_Ibarrier(
// CHECK-NOT:     llvm.alloca
// CHECK:         llvm.call @MPI_Wait(
// CHECK-NOT:     llvm.alloca
// CHECK:         return
func.func @loop(%n: i32) {
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  llvm.br ^cond(%zero : i32)
^cond(%i: i32):
  %more = arith.cmpi slt, %i, %n : i32
  llvm.cond_br %more, ^body, ^exit
^body:
  %rank, %e0 = "spmd.getRankInComm"(%comm) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
  %req, %e1 = "spmd.barrier"(%comm) <{isBlocking = false, usedModel = 0 : i32}> : (!spmd.comm) -> (!spmd.req, !spmd.error)
  %st, %e2 = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> (!spmd.status, !spmd.error)
  %next = arith.addi %i, %one : i32
  llvm.br ^cond(%next : i32)
^exit:
  return
}

// A region that owns its own stack allocations - here the thread team of an
// OpenMP parallel region, which another model lowers - keeps the storage of
// the MPI calls inside it.  Hoisting it to the function would share one
// request slot among all threads of the team.
// CHECK-LABEL: func.func @thread_team
// CHECK-NOT:     llvm.alloca
// CHECK:         "spmd.parallel"
// CHECK:         llvm.alloca %{{.*}} x i32
// CHECK:         llvm.call @MPI_Comm_rank(
// CHECK:         "spmd.parallel_yield"
func.func @thread_team() {
  %comm = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %team = "spmd.commWorld"() <{usedModel = 5 : i32}> : () -> !spmd.comm
  "spmd.parallel"(%team) <{isProcessLevel = false, operandSegmentSizes = array<i32: 1, 0, 0, 0>, usedModel = 5 : i32}> ({
    %rank, %e = "spmd.getRankInComm"(%comm) <{usedModel = 0 : i32}> : (!spmd.comm) -> (i32, !spmd.error)
    "spmd.parallel_yield"() : () -> ()
  }) : (!spmd.comm) -> ()
  return
}
