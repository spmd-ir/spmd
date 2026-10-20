// RUN: spmd-opt %s -lower-spmd-to-mpi | FileCheck %s
//
// A handle MPI takes by address goes to MPI at the address of the program's
// variable it was read from, so MPI's write-back lands there.  That address is
// only used when it provably still holds the handle the op names: the load is
// in the op's block, with nothing between the two that could write memory.
// Otherwise MPI gets a stack slot holding the handle, as it does for a handle
// that never lived in a variable.

// CHECK-LABEL: func.func @no_variable
// CHECK:         %[[HANDLE:.*]] = builtin.unrealized_conversion_cast %arg0 : !spmd.req to i32
// CHECK:         llvm.store %[[HANDLE]], %[[SLOT:[0-9]+]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Wait(%[[SLOT]], %{{.*}})
func.func @no_variable(%req: !spmd.req) {
  %e = "spmd.wait"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// CHECK-LABEL: func.func @llvm_load
// CHECK-NOT:     llvm.alloca
// CHECK:         llvm.call @MPI_Request_free(%arg0)
func.func @llvm_load(%slot: !llvm.ptr) {
  %handle = llvm.load %slot : !llvm.ptr -> i32
  %req = builtin.unrealized_conversion_cast %handle : i32 to !spmd.req
  %e = "spmd.reqFree"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// A store between the load and the op may have changed the variable, so the
// value loaded, not the variable, is what the op frees.
// CHECK-LABEL: func.func @store_between
// CHECK:         %[[HANDLE:.*]] = llvm.load %arg0 : !llvm.ptr -> i32
// CHECK:         llvm.store %arg1, %arg0
// CHECK:         llvm.store %[[HANDLE]], %[[SLOT:[0-9]+]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Request_free(%[[SLOT]])
func.func @store_between(%slot: !llvm.ptr, %other: i32) {
  %handle = llvm.load %slot : !llvm.ptr -> i32
  llvm.store %other, %slot : i32, !llvm.ptr
  %req = builtin.unrealized_conversion_cast %handle : i32 to !spmd.req
  %e = "spmd.reqFree"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  return
}

// A load in another block may be separated from the op by anything.
// CHECK-LABEL: func.func @other_block
// CHECK:         %[[SLOT:.*]] = llvm.alloca %{{.*}} x i32
// CHECK:         %[[HANDLE:.*]] = llvm.load %arg0 : !llvm.ptr -> i32
// CHECK:       ^bb1:
// CHECK:         llvm.store %[[HANDLE]], %[[SLOT]] : i32, !llvm.ptr
// CHECK:         llvm.call @MPI_Request_free(%[[SLOT]])
func.func @other_block(%slot: !llvm.ptr) {
  %handle = llvm.load %slot : !llvm.ptr -> i32
  %req = builtin.unrealized_conversion_cast %handle : i32 to !spmd.req
  llvm.br ^next
^next:
  %e = "spmd.reqFree"(%req) <{usedModel = 0 : i32}> : (!spmd.req) -> !spmd.error
  llvm.return
}

// The raiser's spelling of a communicator MPI_Comm_free takes by address: the
// variable's address cast to the handle type.
// CHECK-LABEL: func.func @address_cast
// CHECK-NOT:     llvm.alloca
// CHECK:         llvm.call @MPI_Comm_free(%arg0)
func.func @address_cast(%var: !llvm.ptr) {
  %comm = builtin.unrealized_conversion_cast %var : !llvm.ptr to !spmd.comm
  %e = "spmd.commDestroy"(%comm) <{usedModel = 0 : i32}> : (!spmd.comm) -> !spmd.error
  return
}
