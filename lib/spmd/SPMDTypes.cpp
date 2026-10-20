// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDTypes.h"
#include "spmd/SPMDAttributes.h"
#include "spmd/SPMDDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "spmd/SPMDDialect.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::spmd;

#define GET_TYPEDEF_CLASSES
#include "spmd/SPMDTypes.cpp.inc"

void SPMDDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "spmd/SPMDTypes.cpp.inc"
      >();
}
