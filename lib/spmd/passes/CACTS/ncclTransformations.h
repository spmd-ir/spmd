/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- ncclTransformations.h - NCCL api calls to spmd -----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares functions for converting nccl func call operations to the spmd
// dialect
//
//===----------------------------------------------------------------------===//
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/JSON.h"

#ifndef NCCL_TRANSFORMATIONS_H
#define NCCL_TRANSFORMATIONS_H

namespace mlir::spmd {
// Entry function for the nccl transformation component that replaces
// func.CallOps by spmd.ops
bool convertApiCallNCCL(LLVM::CallOp &llvmCallOp, int modelGroup);

std::optional<Value>
handleNCCLOperand(const llvm::json::Array &params, StringRef kind,
                  const llvm::json::Object &paramObj, Value oldOperand,
                  unsigned operandIdx, LLVM::CallOp llvmCallOp,
                  OpBuilder &builder, Location loc, unsigned modelGroup);

} // namespace mlir::spmd

#endif // NCCL_TRANSFORMATIONS_H