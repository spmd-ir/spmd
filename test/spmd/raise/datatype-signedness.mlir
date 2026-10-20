// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s
//
// spmd.constDatatype records an integer's signedness when the datatype says
// it: si32 for MPI_INT and shmem_int_*, ui32 for MPI_UNSIGNED and
// shmem_uint_*.  A datatype that does not say it is a signless integer: C
// leaves char's signedness to the platform, MPI_BYTE is no number, and
// shmem_put32 moves words.  All of them used to raise to the signless type, so
// a maximum over MPI_UNSIGNED lowered to one over MPI_INT.

// CHECK-LABEL: llvm.func @mpi_datatypes
// CHECK: "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = si64, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = si16, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = ui16, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = si8, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = ui8, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 0 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = tuple<si32, si32>, usedModel = 0 : i32}>
llvm.func @mpi_datatypes(%buf: !llvm.ptr) {
  %c1 = arith.constant 1 : i32
  %c0 = arith.constant 0 : i32
  %world = arith.constant 1140850688 : i32
  %int = arith.constant 1275069445 : i32
  %unsigned = arith.constant 1275069446 : i32
  %long = arith.constant 1275070471 : i32
  %ulong = arith.constant 1275070472 : i32
  %short = arith.constant 1275068931 : i32
  %ushort = arith.constant 1275068932 : i32
  %schar = arith.constant 1275068696 : i32
  %uchar = arith.constant 1275068674 : i32
  %char = arith.constant 1275068673 : i32
  %byte = arith.constant 1275068685 : i32
  %uint64 = arith.constant 1275070526 : i32
  %twoInt = arith.constant 1275070486 : i32
  %0 = llvm.call @MPI_Send(%buf, %c1, %int, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %1 = llvm.call @MPI_Send(%buf, %c1, %unsigned, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %2 = llvm.call @MPI_Send(%buf, %c1, %long, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %3 = llvm.call @MPI_Send(%buf, %c1, %ulong, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %4 = llvm.call @MPI_Send(%buf, %c1, %short, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %5 = llvm.call @MPI_Send(%buf, %c1, %ushort, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %6 = llvm.call @MPI_Send(%buf, %c1, %schar, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %7 = llvm.call @MPI_Send(%buf, %c1, %uchar, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %8 = llvm.call @MPI_Send(%buf, %c1, %char, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %9 = llvm.call @MPI_Send(%buf, %c1, %byte, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %10 = llvm.call @MPI_Send(%buf, %c1, %uint64, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  %11 = llvm.call @MPI_Send(%buf, %c1, %twoInt, %c0, %c0, %world) : (!llvm.ptr, i32, i32, i32, i32, i32) -> i32
  llvm.return
}
llvm.func @MPI_Send(!llvm.ptr, i32, i32, i32, i32, i32) -> i32

// CHECK-LABEL: llvm.func @shmem_datatypes
// CHECK: "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 1 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = si32, usedModel = 1 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 1 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = si64, usedModel = 1 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 1 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = si8, usedModel = 1 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = i8, usedModel = 1 : i32}>
// CHECK: "spmd.constDatatype"() <{typeAttr = i32, usedModel = 1 : i32}>
llvm.mlir.global external @SHMEM_TEAM_WORLD() : !llvm.ptr
llvm.func @shmem_datatypes(%dst: !llvm.ptr, %src: !llvm.ptr, %pe: i32) {
  %n = arith.constant 4 : i64
  %wa = llvm.mlir.addressof @SHMEM_TEAM_WORLD : !llvm.ptr
  %w = llvm.load %wa : !llvm.ptr -> !llvm.ptr
  %0 = llvm.call @shmem_uint_max_reduce(%w, %dst, %src, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  %1 = llvm.call @shmem_int_max_reduce(%w, %dst, %src, %n) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
  llvm.call @shmem_ulong_put(%dst, %src, %n, %pe) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_ptrdiff_put(%dst, %src, %n, %pe) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_size_put(%dst, %src, %n, %pe) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_int8_put(%dst, %src, %n, %pe) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_char_put(%dst, %src, %n, %pe) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.call @shmem_put32(%dst, %src, %n, %pe) : (!llvm.ptr, !llvm.ptr, i64, i32) -> ()
  llvm.return
}
llvm.func @shmem_uint_max_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_int_max_reduce(!llvm.ptr, !llvm.ptr, !llvm.ptr, i64) -> i32
llvm.func @shmem_ulong_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_ptrdiff_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_size_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_int8_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_char_put(!llvm.ptr, !llvm.ptr, i64, i32)
llvm.func @shmem_put32(!llvm.ptr, !llvm.ptr, i64, i32)
