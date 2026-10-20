// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToMPI.h - Lower SPMD operations to MPI calls ----*- C++ -*-===//
//
// The MPI lowering as a function, so that a driver lowering several models can
// run it without going through -lower-spmd-to-mpi.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDTOMPI_H
#define SPMD_CONVERSION_LOWERSPMDTOMPI_H

#include "LowerSPMDCommon.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::spmd {

/// Reports every operation tagged for MPI that MPI cannot express: a datatype,
/// reduction operation, lock type or thread level with no predefined MPICH
/// handle, and an operand MPI has no parameter for.  A driver lowering several
/// models runs this for each of them before it rewrites anything, so that no
/// diagnostic arrives after half the module has changed.
LogicalResult checkSPMDToMPI(ModuleOp module);

/// Lowers every operation in \p module tagged for MPI to the MPI call it
/// models, as -lower-spmd-to-mpi does, and leaves the operations of other
/// models alone.  Every operation MPI cannot express is reported before
/// anything is rewritten; on failure the module may be partly lowered.
LogicalResult
lowerSPMDToMPI(ModuleOp module,
               StructureLowering structure = StructureLowering::PerModel);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDTOMPI_H
