/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- cudaTransformations.h - CUDA api calls to spmd -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares functions for converting cuda func call operations to the spmd
// dialect
//
//===----------------------------------------------------------------------===//
#include "common.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/JSON.h"
#include <optional>

#ifndef CUDA_TRANSFORMATIONS_H
#define CUDA_TRANSFORMATIONS_H

namespace mlir::spmd {
std::optional<Value>
handleCUDAOperand(const llvm::json::Array &params, StringRef kind,
                  const llvm::json::Object &paramObj, Value oldOperand,
                  unsigned operandIdx, LLVM::CallOp llvmCallOp,
                  OpBuilder &builder, Location loc, unsigned modelGroup);

} // namespace mlir::spmd

#endif // CUDA_TRANSFORMATIONS_H