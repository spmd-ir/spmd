// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDAttributes.h"
#include "spmd/SPMDDialect.h"
#include "spmd/SPMDTypes.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "spmd/SPMDDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::spmd;

#define GET_ATTRDEF_CLASSES
#include "spmd/SPMDAttributes.cpp.inc"
#include "spmd/SPMDEnums.cpp.inc"

// void SPMDDialect::registerAttributes() {
//   addAttributes<
// #define GET_ATTRDEF_LIST
// #include "spmd/SPMDAttributes.cpp.inc"
//       >();
// }