// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToSHMEM.h - Lower SPMD operations to OpenSHMEM --*- C++ -*-===//
//
// The OpenSHMEM lowering as a function, so that a driver lowering several
// models can run it without going through -lower-spmd-to-shmem.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDTOSHMEM_H
#define SPMD_CONVERSION_LOWERSPMDTOSHMEM_H

#include "LowerSPMDCommon.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::spmd {

/// Reports every operation tagged for OpenSHMEM that OpenSHMEM cannot express:
/// a rooted reduction, a communicator split by colour and key, a non-blocking
/// collective, two-sided messaging, a window over the stack, a free of a window
/// whose origin is not in sight, and an operand no OpenSHMEM call has a
/// parameter for.  A
/// driver lowering several models runs this for each of them before it
/// rewrites anything, so that no diagnostic arrives after half the module has
/// changed.
LogicalResult checkSPMDToSHMEM(ModuleOp module);

/// Lowers every operation in \p module tagged for OpenSHMEM to the call it
/// models, as -lower-spmd-to-shmem does, and leaves the operations of other
/// models alone.  Every operation OpenSHMEM cannot express is reported before
/// anything is rewritten; on failure the module may be partly lowered.
LogicalResult
lowerSPMDToSHMEM(ModuleOp module,
                 StructureLowering structure = StructureLowering::PerModel);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDTOSHMEM_H
