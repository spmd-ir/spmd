// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToNVSHMEM.h - Lower SPMD operations to NVSHMEM --*- C++ -*-===//
//
// The NVSHMEM lowering as a function, so that a driver lowering several models
// can run it without going through -lower-spmd-to-nvshmem.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDTONVSHMEM_H
#define SPMD_CONVERSION_LOWERSPMDTONVSHMEM_H

#include "LowerSPMDCommon.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::spmd {

/// Reports every operation tagged for NVSHMEM that NVSHMEM cannot express, as
/// checkSPMDToSHMEM does for OpenSHMEM, and every one inside a GPU kernel or
/// over a window no other PE can reach -- the stack or a global variable,
/// since NVSHMEM's symmetric memory is what nvshmem_malloc allocates.
LogicalResult checkSPMDToNVSHMEM(ModuleOp module);

/// Lowers every operation in \p module tagged for NVSHMEM to the call of
/// NVSHMEM's host API it models, as -lower-spmd-to-nvshmem does, and leaves
/// the operations of other models alone.  Every operation NVSHMEM cannot
/// express is reported before anything is rewritten; on failure the module
/// may be partly lowered.
LogicalResult
lowerSPMDToNVSHMEM(ModuleOp module,
                   StructureLowering structure = StructureLowering::PerModel);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDTONVSHMEM_H
