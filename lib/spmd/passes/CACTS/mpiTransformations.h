/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- mpiTransformations.h - MPI api calls to spmd -------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This declares functions for converting mpi func call operations to the spmd
// dialect
//
//===----------------------------------------------------------------------===//
#include "common.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/JSON.h"
#include <optional>

#ifndef MPI_TRANSFORMATIONS_H
#define MPI_TRANSFORMATIONS_H

namespace mlir::spmd {

std::optional<Value>
handleMPIOperand(const llvm::json::Array &params, StringRef kind,
                 const llvm::json::Object &paramObj, Value oldOperand,
                 unsigned operandIdx, LLVM::CallOp llvmCallOp,
                 OpBuilder &builder, Location loc, unsigned modelGroup);

// TODO remove this from here
bool isStatusIgnore(Value statusArg);

void populateMPIPatterns(mlir::RewritePatternSet &patterns,
                         const ApiRegistryEntry &entry);

} // namespace mlir::spmd

#endif // MPI_TRANSFORMATIONS_H