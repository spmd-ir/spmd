// RUN: spmd-opt %s -convert-apiCalls-to-spmd | FileCheck %s --implicit-check-not="llvm.call @MPI_"
//
// The predefined MPI handles the raiser decodes into operations of their own,
// in the bundled MPICH ABI (include/spmd/TargetABI/MPICH.h, checked against
// mpi.h by //test/abi:mpich_abi_test). Each call has its own count, so every
// CHECK pairs a handle with the call that used it.
//
// A datatype keeps the MLIR type of one element: signed or unsigned where the
// datatype says so, signless where it does not (C leaves the sign of char to
// the platform, and MPI_BYTE is no number), and a tuple for the pair types
// of MINLOC and MAXLOC. A reduction keeps the name MPI gives it.
//
// A handle that is not one of these constants (a datatype the program built,
// or one passed into a helper function) is not decoded; see MR 16's
// description.

// CHECK-LABEL: llvm.func @main
// CHECK:         "spmd.parallel"
// CHECK:         %[[W:.*]] = "spmd.commWorld"() <{usedModel = 0 : i32}>
llvm.func @main() -> i32 {
  %c0 = arith.constant 0 : i32
  %c8 = arith.constant 8 : i32
  // MPI_COMM_WORLD, MPI_INT and MPI_SUM.
  %world = arith.constant 1140850688 : i32
  %int = arith.constant 1275069445 : i32
  %sum = arith.constant 1476395011 : i32
  %null = llvm.mlir.zero : !llvm.ptr
  %sbuf = llvm.alloca %c8 x i64 : (i32) -> !llvm.ptr
  %rbuf = llvm.alloca %c8 x i64 : (i32) -> !llvm.ptr
  %init = llvm.call @MPI_Init(%null, %null) : (!llvm.ptr, !llvm.ptr) -> i32

  // Datatypes, with MPI_SUM.
  // CHECK: %[[T1:.*]] = "spmd.constDatatype"() <{typeAttr = i1, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c1_i64, %[[T1]], %{{.*}})
  %n1 = arith.constant 1 : i32
  %dt1 = arith.constant 1275068735 : i32  // MPI_C_BOOL = 0x4c00013f
  %r1 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n1, %dt1, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T2:.*]] = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c2_i64, %[[T2]], %{{.*}})
  %n2 = arith.constant 2 : i32
  %dt2 = arith.constant 1275068673 : i32  // MPI_CHAR = 0x4c000101
  %r2 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n2, %dt2, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T3:.*]] = "spmd.constDatatype"() <{typeAttr = si8, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c3_i64, %[[T3]], %{{.*}})
  %n3 = arith.constant 3 : i32
  %dt3 = arith.constant 1275068696 : i32  // MPI_SIGNED_CHAR = 0x4c000118
  %r3 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n3, %dt3, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T4:.*]] = "spmd.constDatatype"() <{typeAttr = ui8, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c4_i64, %[[T4]], %{{.*}})
  %n4 = arith.constant 4 : i32
  %dt4 = arith.constant 1275068674 : i32  // MPI_UNSIGNED_CHAR = 0x4c000102
  %r4 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n4, %dt4, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T5:.*]] = "spmd.constDatatype"() <{typeAttr = i8, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c5_i64, %[[T5]], %{{.*}})
  %n5 = arith.constant 5 : i32
  %dt5 = arith.constant 1275068685 : i32  // MPI_BYTE = 0x4c00010d
  %r5 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n5, %dt5, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T6:.*]] = "spmd.constDatatype"() <{typeAttr = si16, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c6_i64, %[[T6]], %{{.*}})
  %n6 = arith.constant 6 : i32
  %dt6 = arith.constant 1275068931 : i32  // MPI_SHORT = 0x4c000203
  %r6 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n6, %dt6, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T7:.*]] = "spmd.constDatatype"() <{typeAttr = ui16, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c7_i64, %[[T7]], %{{.*}})
  %n7 = arith.constant 7 : i32
  %dt7 = arith.constant 1275068932 : i32  // MPI_UNSIGNED_SHORT = 0x4c000204
  %r7 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n7, %dt7, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T8:.*]] = "spmd.constDatatype"() <{typeAttr = si32, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c8_i64, %[[T8]], %{{.*}})
  %n8 = arith.constant 8 : i32
  %dt8 = arith.constant 1275069445 : i32  // MPI_INT = 0x4c000405
  %r8 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n8, %dt8, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T9:.*]] = "spmd.constDatatype"() <{typeAttr = ui32, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c9_i64, %[[T9]], %{{.*}})
  %n9 = arith.constant 9 : i32
  %dt9 = arith.constant 1275069446 : i32  // MPI_UNSIGNED = 0x4c000406
  %r9 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n9, %dt9, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T10:.*]] = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c10_i64, %[[T10]], %{{.*}})
  %n10 = arith.constant 10 : i32
  %dt10 = arith.constant 1275070471 : i32  // MPI_LONG = 0x4c000807
  %r10 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n10, %dt10, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T11:.*]] = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c11_i64, %[[T11]], %{{.*}})
  %n11 = arith.constant 11 : i32
  %dt11 = arith.constant 1275070472 : i32  // MPI_UNSIGNED_LONG = 0x4c000808
  %r11 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n11, %dt11, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T12:.*]] = "spmd.constDatatype"() <{typeAttr = si64, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c12_i64, %[[T12]], %{{.*}})
  %n12 = arith.constant 12 : i32
  %dt12 = arith.constant 1275070473 : i32  // MPI_LONG_LONG_INT = 0x4c000809
  %r12 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n12, %dt12, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T13:.*]] = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c13_i64, %[[T13]], %{{.*}})
  %n13 = arith.constant 13 : i32
  %dt13 = arith.constant 1275070489 : i32  // MPI_UNSIGNED_LONG_LONG = 0x4c000819
  %r13 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n13, %dt13, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T14:.*]] = "spmd.constDatatype"() <{typeAttr = ui64, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c14_i64, %[[T14]], %{{.*}})
  %n14 = arith.constant 14 : i32
  %dt14 = arith.constant 1275070526 : i32  // MPI_UINT64_T = 0x4c00083e
  %r14 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n14, %dt14, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T15:.*]] = "spmd.constDatatype"() <{typeAttr = f32, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c15_i64, %[[T15]], %{{.*}})
  %n15 = arith.constant 15 : i32
  %dt15 = arith.constant 1275069450 : i32  // MPI_FLOAT = 0x4c00040a
  %r15 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n15, %dt15, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T16:.*]] = "spmd.constDatatype"() <{typeAttr = f64, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c16_i64, %[[T16]], %{{.*}})
  %n16 = arith.constant 16 : i32
  %dt16 = arith.constant 1275070475 : i32  // MPI_DOUBLE = 0x4c00080b
  %r16 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n16, %dt16, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T17:.*]] = "spmd.constDatatype"() <{typeAttr = f128, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c17_i64, %[[T17]], %{{.*}})
  %n17 = arith.constant 17 : i32
  %dt17 = arith.constant 1275072524 : i32  // MPI_LONG_DOUBLE = 0x4c00100c
  %r17 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n17, %dt17, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T18:.*]] = "spmd.constDatatype"() <{typeAttr = tuple<f32, si32>, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c18_i64, %[[T18]], %{{.*}})
  %n18 = arith.constant 18 : i32
  %dt18 = arith.constant -1946157056 : i32  // MPI_FLOAT_INT = 0x8c000000
  %r18 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n18, %dt18, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T19:.*]] = "spmd.constDatatype"() <{typeAttr = tuple<f64, si32>, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c19_i64, %[[T19]], %{{.*}})
  %n19 = arith.constant 19 : i32
  %dt19 = arith.constant -1946157055 : i32  // MPI_DOUBLE_INT = 0x8c000001
  %r19 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n19, %dt19, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[T20:.*]] = "spmd.constDatatype"() <{typeAttr = tuple<si32, si32>, usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c20_i64, %[[T20]], %{{.*}})
  %n20 = arith.constant 20 : i32
  %dt20 = arith.constant 1275070486 : i32  // MPI_2INT = 0x4c000816
  %r20 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n20, %dt20, %sum, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32

  // Reductions (MPI_MAX = 0x58000001 counting up to MPI_EQUAL), over MPI_INT.
  // CHECK: %[[O101:.*]] = "spmd.constOpType"() <{stringAttr = "MAX", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c101_i64, %{{.*}}, %[[O101]])
  %n101 = arith.constant 101 : i32
  %op101 = arith.constant 1476395009 : i32  // MPI_MAX
  %r101 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n101, %int, %op101, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O102:.*]] = "spmd.constOpType"() <{stringAttr = "MIN", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c102_i64, %{{.*}}, %[[O102]])
  %n102 = arith.constant 102 : i32
  %op102 = arith.constant 1476395010 : i32  // MPI_MIN
  %r102 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n102, %int, %op102, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O103:.*]] = "spmd.constOpType"() <{stringAttr = "SUM", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c103_i64, %{{.*}}, %[[O103]])
  %n103 = arith.constant 103 : i32
  %op103 = arith.constant 1476395011 : i32  // MPI_SUM
  %r103 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n103, %int, %op103, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O104:.*]] = "spmd.constOpType"() <{stringAttr = "PROD", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c104_i64, %{{.*}}, %[[O104]])
  %n104 = arith.constant 104 : i32
  %op104 = arith.constant 1476395012 : i32  // MPI_PROD
  %r104 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n104, %int, %op104, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O105:.*]] = "spmd.constOpType"() <{stringAttr = "LAND", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c105_i64, %{{.*}}, %[[O105]])
  %n105 = arith.constant 105 : i32
  %op105 = arith.constant 1476395013 : i32  // MPI_LAND
  %r105 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n105, %int, %op105, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O106:.*]] = "spmd.constOpType"() <{stringAttr = "BAND", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c106_i64, %{{.*}}, %[[O106]])
  %n106 = arith.constant 106 : i32
  %op106 = arith.constant 1476395014 : i32  // MPI_BAND
  %r106 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n106, %int, %op106, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O107:.*]] = "spmd.constOpType"() <{stringAttr = "LOR", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c107_i64, %{{.*}}, %[[O107]])
  %n107 = arith.constant 107 : i32
  %op107 = arith.constant 1476395015 : i32  // MPI_LOR
  %r107 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n107, %int, %op107, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O108:.*]] = "spmd.constOpType"() <{stringAttr = "BOR", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c108_i64, %{{.*}}, %[[O108]])
  %n108 = arith.constant 108 : i32
  %op108 = arith.constant 1476395016 : i32  // MPI_BOR
  %r108 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n108, %int, %op108, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O109:.*]] = "spmd.constOpType"() <{stringAttr = "LXOR", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c109_i64, %{{.*}}, %[[O109]])
  %n109 = arith.constant 109 : i32
  %op109 = arith.constant 1476395017 : i32  // MPI_LXOR
  %r109 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n109, %int, %op109, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O110:.*]] = "spmd.constOpType"() <{stringAttr = "BXOR", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c110_i64, %{{.*}}, %[[O110]])
  %n110 = arith.constant 110 : i32
  %op110 = arith.constant 1476395018 : i32  // MPI_BXOR
  %r110 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n110, %int, %op110, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O111:.*]] = "spmd.constOpType"() <{stringAttr = "MINLOC", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c111_i64, %{{.*}}, %[[O111]])
  %n111 = arith.constant 111 : i32
  %op111 = arith.constant 1476395019 : i32  // MPI_MINLOC
  %r111 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n111, %int, %op111, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O112:.*]] = "spmd.constOpType"() <{stringAttr = "MAXLOC", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c112_i64, %{{.*}}, %[[O112]])
  %n112 = arith.constant 112 : i32
  %op112 = arith.constant 1476395020 : i32  // MPI_MAXLOC
  %r112 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n112, %int, %op112, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O113:.*]] = "spmd.constOpType"() <{stringAttr = "REPLACE", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c113_i64, %{{.*}}, %[[O113]])
  %n113 = arith.constant 113 : i32
  %op113 = arith.constant 1476395021 : i32  // MPI_REPLACE
  %r113 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n113, %int, %op113, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O114:.*]] = "spmd.constOpType"() <{stringAttr = "NO_OP", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c114_i64, %{{.*}}, %[[O114]])
  %n114 = arith.constant 114 : i32
  %op114 = arith.constant 1476395022 : i32  // MPI_NO_OP
  %r114 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n114, %int, %op114, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
  // CHECK: %[[O115:.*]] = "spmd.constOpType"() <{stringAttr = "EQUAL", usedModel = 0 : i32}>
  // CHECK: "spmd.allreduce"(%[[W]], %{{.*}}, %{{.*}}, %c115_i64, %{{.*}}, %[[O115]])
  %n115 = arith.constant 115 : i32
  %op115 = arith.constant 1476395023 : i32  // MPI_EQUAL
  %r115 = llvm.call @MPI_Allreduce(%sbuf, %rbuf, %n115, %int, %op115, %world) : (!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32

  // MPI_COMM_SELF is no communicator the dialect names: it stays the
  // program's handle, cast to a communicator.
  // CHECK: %[[SELF:.*]] = builtin.unrealized_conversion_cast %c1140850689_i32 : i32 to !spmd.comm
  // CHECK: "spmd.barrier"(%[[SELF]])
  %self = arith.constant 1140850689 : i32
  %b = llvm.call @MPI_Barrier(%self) : (i32) -> i32

  %fin = llvm.call @MPI_Finalize() : () -> i32
  llvm.return %c0 : i32
}
llvm.func @MPI_Init(!llvm.ptr, !llvm.ptr) -> i32
llvm.func @MPI_Allreduce(!llvm.ptr, !llvm.ptr, i32, i32, i32, i32) -> i32
llvm.func @MPI_Barrier(i32) -> i32
llvm.func @MPI_Finalize() -> i32
