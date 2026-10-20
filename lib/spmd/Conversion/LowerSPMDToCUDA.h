// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToCUDA.h - Lower SPMD operations to CUDA --------*- C++ -*-===//
//
// The CUDA lowering as a function, so that a driver lowering several models
// can run it without going through -lower-spmd-to-cuda.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDTOCUDA_H
#define SPMD_CONVERSION_LOWERSPMDTOCUDA_H

#include "LowerSPMDCommon.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::spmd {

/// Reports every operation tagged for CUDA that CUDA cannot express: an
/// operation no call of the raiser's map raises to, an operand no call has a
/// parameter for, and a constant $kind on an spmd.memcpy that is no direction
/// of enum cudaMemcpyKind.  A driver lowering several models runs this for
/// each of them before it rewrites anything, so that no diagnostic arrives
/// after half the module has changed.
LogicalResult checkSPMDToCUDA(ModuleOp module);

/// Lowers every operation in \p module tagged for CUDA to the call it models,
/// as -lower-spmd-to-cuda does, and leaves the operations of other models
/// alone.  Every operation CUDA cannot express is reported before anything is
/// rewritten; on failure the module may be partly lowered.
LogicalResult
lowerSPMDToCUDA(ModuleOp module,
                StructureLowering structure = StructureLowering::PerModel);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDTOCUDA_H
