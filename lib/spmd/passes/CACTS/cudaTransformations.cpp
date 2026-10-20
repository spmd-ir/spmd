// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "cudaTransformations.h"
#include "cudaSpmdEntries.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "spmd/SPMDOps.h"

#include "common.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace mlir::spmd {

std::optional<Value>
handleCUDAOperand(const llvm::json::Array &params, StringRef kind,
                  const llvm::json::Object &paramObj, Value oldOperand,
                  unsigned operandIdx, LLVM::CallOp llvmCallOp,
                  OpBuilder &builder, Location loc, unsigned modelGroup) {
  LLVM_DEBUG(DBGS() << "Not Implemented yet: handleCUDAOperand " << kind
                    << "\n");
  return std::nullopt;
}

} // namespace mlir::spmd