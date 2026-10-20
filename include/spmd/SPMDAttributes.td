//===- SPMDAttributes.td - SPMD dialect attributes -------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_ATTRIBUTES
#define SPMD_ATTRIBUTES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/EnumAttr.td"
include "SPMDDialect.td"

//===----------------------------------------------------------------------===//
// SPMD type definitions
//===----------------------------------------------------------------------===//

class SPMD_Attribute<string attrName, string attrMnemonic, list<Trait> traits = []>
    : TypeDef<SPMD_Dialect, attrName, traits> {
  let mnemonic = attrMnemonic;
}

//def SPMD_CustomType : SPMD_Type<"Custom", "custom"> {
//    let summary = "SPMD custom type";
//    let description = "Custom type in SPMD dialect";
//    let parameters = (ins StringRefParameter<"the custom value">:$value);
//    let assemblyFormat = "`<` $value `>`";
//}

// def SPMD_DatatypeAttr : I64EnumAttr<
//     "ExecutorType", "",
//     [
//       I32EnumAttrCase<"INT", 0, "dynamic">,
//       I32EnumAttrCase<"static", 1, "dynamic">,

//     ]> {
//   let cppNamespace = "::mlir::arith";
// }

// def SPMD_ExecutorKindAttr : I32EnumAttr<
//     "ExecutorKind", "The kind of execution of an operation",
//     [
//       I32EnumAttrCase<"All", 0, "all">,
//       I32EnumAttrCase<"Dynamic", 1, "dynamic">,
//       I32EnumAttrCase<"Static", 2, "static">,
//       I32EnumAttrCase<"One", 3, "one">,
//     ]> {
//   let cppNamespace = "::mlir::spmd";
// }

//===----------------------------------------------------------------------===//
// Thread level enum attribute
// Used as a compile-time annotation on spmd.parallel for thread-safety analysis.
//===----------------------------------------------------------------------===//

def ThreadLevel_Single     : I32EnumAttrCase<"Single",     0, "single">;
def ThreadLevel_Funneled   : I32EnumAttrCase<"Funneled",   1, "funneled">;
def ThreadLevel_Serialized : I32EnumAttrCase<"Serialized", 2, "serialized">;
def ThreadLevel_Multiple   : I32EnumAttrCase<"Multiple",   3, "multiple">;

def SPMD_ThreadLevelAttr : I32EnumAttr<"ThreadLevel",
    "thread-support level required at MPI_Init_Thread",
    [ThreadLevel_Single, ThreadLevel_Funneled,
     ThreadLevel_Serialized, ThreadLevel_Multiple]> {
  let cppNamespace = "::mlir::spmd";
  let genSpecializedAttr = 1;
}



def SPMD_ScheduleKindStatic     : I32EnumAttrCase<"Static",     0, "static">;
//we may should use  DynamicLike   // includes dynamic/guided/runtime/auto instead
def SPMD_ScheduleKindDynamic    : I32EnumAttrCase<"Dynamic",    1, "dynamic">;
def SPMD_ScheduleKindGuided     : I32EnumAttrCase<"Guided",     2, "guided">;
def SPMD_ScheduleKindRuntime    : I32EnumAttrCase<"Runtime",    3, "runtime">;
def SPMD_ScheduleKindAuto       : I32EnumAttrCase<"Auto",    4, "auto">;

def SPMD_ScheduleKindAttr : I32EnumAttr<"ScheduleKind",
    "SPMD loop scheduling policy",
    [
      SPMD_ScheduleKindStatic,
      SPMD_ScheduleKindDynamic,
      SPMD_ScheduleKindGuided,
      SPMD_ScheduleKindRuntime,
      SPMD_ScheduleKindAuto
    ]> {
  let cppNamespace = "::mlir::spmd";
  let genSpecializedAttr = 1;
}


def SPMD_ReductionKindAttr : I32EnumAttr<"ReductionKind",
    "Well-known reduction operator (derived annotation, not authoritative)",
    [
      I32EnumAttrCase<"Add",   0, "add">,
      I32EnumAttrCase<"Mul",   1, "mul">,
      I32EnumAttrCase<"MinS",  2, "min_s">,
      I32EnumAttrCase<"MinU",  3, "min_u">,
      I32EnumAttrCase<"MinF",  4, "min_f">,
      I32EnumAttrCase<"MaxS",  5, "max_s">,
      I32EnumAttrCase<"MaxU",  6, "max_u">,
      I32EnumAttrCase<"MaxF",  7, "max_f">,
      I32EnumAttrCase<"And",   8, "and">,
      I32EnumAttrCase<"Or",    9, "or">,
      I32EnumAttrCase<"Xor",  10, "xor">,
      I32EnumAttrCase<"LAnd", 11, "land">,
      I32EnumAttrCase<"LOr",  12, "lor">,
      I32EnumAttrCase<"LXor", 13, "lxor">
    ]> {
  let cppNamespace = "mlir::spmd";
  let genSpecializedAttr = 1;
}


#endif // SPMD_ATTRIBUTES
