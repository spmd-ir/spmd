// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToOMP.h - Lower SPMD operations to OpenMP -------*- C++ -*-===//
//
// The OpenMP lowering as a function, so that a driver lowering several models
// can run it without going through -lower-spmd-to-omp.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDTOOMP_H
#define SPMD_CONVERSION_LOWERSPMDTOOMP_H

#include "LowerSPMDCommon.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir::spmd {

/// Reports every operation tagged for OpenMP that OpenMP cannot express: an
/// operation of another model's kind (a message, a split communicator), a
/// process team, a parallel region with private variables, a query or a
/// worksharing construct on a team other than the current one, and a team
/// used by another model.  A driver lowering several models runs this for each
/// of them before it rewrites anything.
LogicalResult checkSPMDToOMP(ModuleOp module);

/// Lowers every operation in \p module tagged for OpenMP back to what
/// -omp-to-spmd raised it from: the directives to the omp dialect, and the
/// runtime queries to the calls they model.  Operations of other models are
/// left alone.  Every operation OpenMP cannot express is reported before
/// anything is rewritten; on failure the module may be partly lowered.
LogicalResult
lowerSPMDToOMP(ModuleOp module,
               StructureLowering structure = StructureLowering::PerModel);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDTOOMP_H
