// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDDialect.h"
#include "mlir/IR/DialectImplementation.h"
#include "spmd/SPMDOps.h"

using namespace mlir;
using namespace mlir::spmd;

#include "spmd/SPMDOpsDialect.cpp.inc"

//===----------------------------------------------------------------------===//
// SPMD dialect.
//===----------------------------------------------------------------------===//

void SPMDDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "spmd/SPMDOps.cpp.inc"
      >();
  registerTypes();
}
