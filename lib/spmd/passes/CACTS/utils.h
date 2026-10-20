/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- utils.h - CACTS related utilities ------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This implements miscellaneous functionality  useful for the CACTS pass
// and related components that do not fit into one component.
// Include several headers useful for multiple components.
//
//===----------------------------------------------------------------------===//

#ifndef UTILS_H
#define UTILS_H

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "spmd/SPMDOps.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/JSON.h"
#include <string>

// #include "polygeist/Ops.h"

namespace mlir::spmd {

/// Strip a trailing numeric suffix of the form ".<digits>" from a symbol name.
/// "MPI_Send.42" -> "MPI_Send",  "MPI_Send" -> "MPI_Send"
inline llvm::StringRef stripNumericSuffix(llvm::StringRef name) {
  auto dot = name.rfind('.');
  if (dot == llvm::StringRef::npos)
    return name;
  llvm::StringRef suffix = name.substr(dot + 1);
  if (!suffix.empty() && llvm::all_of(suffix, llvm::isDigit))
    return name.substr(0, dot);
  return name;
}

std::string toLowerStr(StringRef s);

std::string toUpperStr(llvm::StringRef s);

std::optional<std::string> getParamString(const llvm::json::Object *obj,
                                          llvm::StringRef key);

std::optional<bool> getParamBool(const llvm::json::Object *obj,
                                 llvm::StringRef key);

bool isArithConstantIntX(Value value, int x);

void getOriginalCommArg(Value commArg, SmallVector<Value> &origCommArgs,
                        SmallVector<Operation *> &branchLikeOps);

// Function type for model-specific single-communicator conversion.
// Takes the original comm Value and modelGroup, returns the new spmd::CommType
// Value.
using SingleCommHandler = std::function<Value(Value, int)>;

// Shared comm conversion logic. Handles both single and arith.select (branch)
// cases. Model-specific behavior is injected via `handleSingle`.
Value convertOldToSPMDComms(SmallVector<Value> origCommArgs,
                            SmallVector<Operation *> branchLikeOps,
                            int modelGroup,
                            const SingleCommHandler &handleSingle);

// Shared entry point: traces comm arg origins, then delegates to
// convertOldToSPMDComms.
Value getNewComm(Value commArg, int modelGroup,
                 const SingleCommHandler &handleSingle);

} // namespace mlir::spmd

#endif // UTILS_H