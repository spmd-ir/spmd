// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToNCCL.h - Lower SPMD operations to NCCL --------*- C++ -*-===//
//
// The NCCL lowering as a function, so that a driver lowering several models
// can run it without going through -lower-spmd-to-nccl.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDTONCCL_H
#define SPMD_CONVERSION_LOWERSPMDTONCCL_H

#include "LowerSPMDCommon.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::spmd {

/// Reports every operation tagged for NCCL that NCCL cannot express: an
/// operation NCCL has no call for, a group whose operations are not all
/// completed by one wait in their block, the world communicator read where
/// NCCL has not been initialised, an initialisation with no MPI or OpenSHMEM
/// in the module to bootstrap it over, and an operand no NCCL call has a
/// parameter for.  A driver lowering several models runs this for each of them
/// before it rewrites anything, so that no diagnostic arrives after half the
/// module has changed.
LogicalResult checkSPMDToNCCL(ModuleOp module);

/// Lowers every operation in \p module tagged for NCCL to the call it models,
/// as -lower-spmd-to-nccl does, and leaves the operations of other models
/// alone.  Every operation NCCL cannot express is reported before anything is
/// rewritten; on failure the module may be partly lowered.
LogicalResult
lowerSPMDToNCCL(ModuleOp module,
                StructureLowering structure = StructureLowering::PerModel);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDTONCCL_H
