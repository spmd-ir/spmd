// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ompTransformations.h"
#include "ompSpmdEntries.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"

#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"

namespace mlir::spmd {

std::optional<Value>
handleOMPOperand(const llvm::json::Array &params, StringRef kind,
                 const llvm::json::Object &paramObj, Value oldOperand,
                 unsigned operandIdx, LLVM::CallOp llvmCallOp,
                 OpBuilder &builder, Location loc, unsigned modelGroup) {

  // OMP runtime API calls mapped in ompToSpmdOperands have no model-specific
  // operand conversions beyond the communicator injection handled via
  // D("comm", getParallelComm) in the entries map. All operands that reach
  // this function are passed through unchanged.
  //
  // If future OMP API calls require operand-level transformations (e.g.
  // converting an OMP lock handle to an spmd type), add cases here following
  // the same pattern as handleMPIOperand.

  LLVM_DEBUG(
      llvm::dbgs()
      << "[OMPTransformations] no model-specific handler for operand kind: "
      << kind << " — passing through unchanged\n");

  return std::nullopt;
}

void populateOMPPatterns(RewritePatternSet &patterns,
                         const ApiRegistryEntry &entry) {
  // No OMP-specific rewrite patterns at this time. The generic
  // ConvertApiCallsToSPMD machinery handles all entries in ompToSpmdOperands
  // without requiring additional patterns.
  (void)entry;
}

} // namespace mlir::spmd