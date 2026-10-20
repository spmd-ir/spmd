/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- ompTransformations.h - OMP runtime API -> SPMD - Header --*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Declares functions for converting OpenMP runtime API function calls
// (omp_get_thread_num, omp_get_num_threads, etc.) to the SPMD dialect.
// Structural OpenMP ops (omp.parallel, omp.barrier, etc.) are handled
// separately by the OMPToSPMD pass.
//
//===----------------------------------------------------------------------===//
#ifndef OMP_TRANSFORMATIONS_H
#define OMP_TRANSFORMATIONS_H

#include "common.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "llvm/Support/JSON.h"
#include <optional>

namespace mlir::spmd {

/// Model-specific operand handler for OpenMP runtime API calls.
/// Most OMP runtime functions need a communicator injected as a default
/// operand via spmd.getParallelComm — this function handles any remaining
/// OMP-specific operand conversions beyond that.
std::optional<Value>
handleOMPOperand(const llvm::json::Array &params, StringRef kind,
                 const llvm::json::Object &paramObj, Value oldOperand,
                 unsigned operandIdx, LLVM::CallOp llvmCallOp,
                 OpBuilder &builder, Location loc, unsigned modelGroup);

/// Populates OMP-specific rewrite patterns into the given pattern set.
/// Currently a no-op — OMP runtime calls are fully handled by the generic
/// ConvertApiCallsToSPMD machinery via ompToSpmdOperands. Reserved for
/// future OMP-specific pattern needs.
void populateOMPPatterns(mlir::RewritePatternSet &patterns,
                         const ApiRegistryEntry &entry);

} // namespace mlir::spmd

#endif // OMP_TRANSFORMATIONS_H