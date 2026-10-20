// RUN: spmd-opt %s -allow-unregistered-dialect -lower-spmd-to-mpi | FileCheck %s
//
// The predefined handles, from include/spmd/TargetABI/MPICH.h.
// spmd.constDatatype keeps an MLIR type, so a type lowers to the first
// predefined datatype of that shape: MPI_CHAR against MPI_BYTE, and
// MPI_LONG against MPI_LONG_LONG_INT, are not recorded.  Signedness is: si32
// is MPI_INT and ui32 MPI_UNSIGNED.  An integer of unknown signedness that no
// datatype records as such -- i32 -- is the signed datatype of its width,
// which moves the same bits; MAX and MIN over it are errors
// (mpi-unlowerable.mlir).  A type or name without a predefined handle is an
// error too, never a stand-in.

// CHECK-LABEL: func.func @datatypes
// CHECK:         llvm.mlir.constant(1275068735 : i32) : i32
// CHECK:         llvm.mlir.constant(1275068673 : i32) : i32
// CHECK:         llvm.mlir.constant(1275068696 : i32) : i32
// CHECK:         llvm.mlir.constant(1275068674 : i32) : i32
// CHECK:         llvm.mlir.constant(1275068931 : i32) : i32
// CHECK:         llvm.mlir.constant(1275068932 : i32) : i32
// CHECK:         llvm.mlir.constant(1275069445 : i32) : i32
// CHECK:         llvm.mlir.constant(1275069446 : i32) : i32
// CHECK:         llvm.mlir.constant(1275070471 : i32) : i32
// CHECK:         llvm.mlir.constant(1275070472 : i32) : i32
// CHECK:         llvm.mlir.constant(1275069450 : i32) : i32
// CHECK:         llvm.mlir.constant(1275070475 : i32) : i32
// CHECK:         llvm.mlir.constant(1275072524 : i32) : i32
// CHECK:         llvm.mlir.constant(-1946157056 : i32) : i32
// CHECK:         llvm.mlir.constant(-1946157055 : i32) : i32
// CHECK:         llvm.mlir.constant(1275070486 : i32) : i32
// CHECK:         llvm.mlir.constant(1275068931 : i32) : i32
// CHECK:         llvm.mlir.constant(1275069445 : i32) : i32
// CHECK:         llvm.mlir.constant(1275070471 : i32) : i32
// CHECK:         "test.keep"
func.func @datatypes() {
  %d0 = "spmd.constDatatype"() <{typeAttr = i1, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d1 = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d2 = "spmd.constDatatype"() <{typeAttr = si8, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d3 = "spmd.constDatatype"() <{typeAttr = ui8, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d4 = "spmd.constDatatype"() <{typeAttr = si16, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d5 = "spmd.constDatatype"() <{typeAttr = ui16, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d6 = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d7 = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d8 = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d9 = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d10 = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d11 = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d12 = "spmd.constDatatype"() <{typeAttr = f128, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d13 = "spmd.constDatatype"() <{typeAttr = tuple<f32, si32>, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d14 = "spmd.constDatatype"() <{typeAttr = tuple<f64, si32>, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d15 = "spmd.constDatatype"() <{typeAttr = tuple<si32, si32>, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d16 = "spmd.constDatatype"() <{typeAttr = i16, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d17 = "spmd.constDatatype"() <{typeAttr = i32, usedModel = 0 : i32}> : () -> !spmd.datatype
  %d18 = "spmd.constDatatype"() <{typeAttr = i64, usedModel = 0 : i32}> : () -> !spmd.datatype
  "test.keep"(%d0, %d1, %d2, %d3, %d4, %d5, %d6, %d7, %d8, %d9, %d10, %d11, %d12, %d13, %d14, %d15, %d16, %d17, %d18) : (!spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype, !spmd.datatype) -> ()
  return
}

// CHECK-LABEL: func.func @reduce_ops
// CHECK:         llvm.mlir.constant(1476395009 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395010 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395011 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395012 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395013 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395014 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395015 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395016 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395017 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395018 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395019 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395020 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395021 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395022 : i32) : i32
// CHECK:         llvm.mlir.constant(1476395023 : i32) : i32
// CHECK:         "test.keep"
func.func @reduce_ops() {
  %MAX = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 0 : i32}> : () -> !spmd.op
  %MIN = "spmd.constOpType"() <{stringAttr = "MIN", usedModel = 0 : i32}> : () -> !spmd.op
  %SUM = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}> : () -> !spmd.op
  %PROD = "spmd.constOpType"() <{stringAttr = "PROD", usedModel = 0 : i32}> : () -> !spmd.op
  %LAND = "spmd.constOpType"() <{stringAttr = "LAND", usedModel = 0 : i32}> : () -> !spmd.op
  %BAND = "spmd.constOpType"() <{stringAttr = "BAND", usedModel = 0 : i32}> : () -> !spmd.op
  %LOR = "spmd.constOpType"() <{stringAttr = "LOR", usedModel = 0 : i32}> : () -> !spmd.op
  %BOR = "spmd.constOpType"() <{stringAttr = "BOR", usedModel = 0 : i32}> : () -> !spmd.op
  %LXOR = "spmd.constOpType"() <{stringAttr = "LXOR", usedModel = 0 : i32}> : () -> !spmd.op
  %BXOR = "spmd.constOpType"() <{stringAttr = "BXOR", usedModel = 0 : i32}> : () -> !spmd.op
  %MINLOC = "spmd.constOpType"() <{stringAttr = "MINLOC", usedModel = 0 : i32}> : () -> !spmd.op
  %MAXLOC = "spmd.constOpType"() <{stringAttr = "MAXLOC", usedModel = 0 : i32}> : () -> !spmd.op
  %REPLACE = "spmd.constOpType"() <{stringAttr = "REPLACE", usedModel = 0 : i32}> : () -> !spmd.op
  %NO_OP = "spmd.constOpType"() <{stringAttr = "NO_OP", usedModel = 0 : i32}> : () -> !spmd.op
  %EQUAL = "spmd.constOpType"() <{stringAttr = "EQUAL", usedModel = 0 : i32}> : () -> !spmd.op
  "test.keep"(%MAX, %MIN, %SUM, %PROD, %LAND, %BAND, %LOR, %BOR, %LXOR, %BXOR, %MINLOC, %MAXLOC, %REPLACE, %NO_OP, %EQUAL) : (!spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op, !spmd.op) -> ()
  return
}

// MPI_COMM_WORLD is 0x44000000 and MPI_REQUEST_NULL 0x2c000000.
// CHECK-LABEL: func.func @handles
// CHECK:         llvm.mlir.constant(1140850688 : i32) : i32
// CHECK:         llvm.mlir.constant(738197504 : i32) : i32
func.func @handles() -> (!spmd.comm, !spmd.req) {
  %world = "spmd.commWorld"() <{usedModel = 0 : i32}> : () -> !spmd.comm
  %null = "spmd.requestNull"() <{usedModel = 0 : i32}> : () -> !spmd.req
  return %world, %null : !spmd.comm, !spmd.req
}
