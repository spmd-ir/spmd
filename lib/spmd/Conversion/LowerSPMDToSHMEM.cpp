// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToSHMEM.cpp - Lower SPMD operations to OpenSHMEM ----------===//
//
// -lower-spmd-to-shmem: the OpenSHMEM family's lowering
// (LowerSPMDToSHMEMFamily.cpp) under the Sandia OpenSHMEM ABI of
// includesProgramModels/shmem/include, whose constants are
// include/spmd/TargetABI/OpenSHMEM.h: a team is a pointer, and the predefined
// teams are extern variables the lowering loads.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDToSHMEM.h"
#include "LowerSPMDToSHMEMFamily.h"

#include "spmd/SPMDPasses.h"
#include "spmd/TargetABI/OpenSHMEM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"

namespace mlir::spmd {

#define GEN_PASS_DEF_LOWERSPMDTOSHMEM
#include "spmd/SPMDPasses.h.inc"

namespace {

namespace openshmem = abi::openshmem;

std::optional<int32_t> threadLevel(StringRef name) {
  return openshmem::lookupHandle(openshmem::kThreadLevels, name);
}
std::optional<int32_t> signalOp(StringRef name) {
  return openshmem::lookupHandle(openshmem::kSignalOps, name);
}

const SHMEMTarget kOpenSHMEM = {
    /*model=*/ProgrammingModel::SHMEM,
    /*flavour=*/SHMEMFlavour::OpenSHMEM,
    /*passName=*/"lower-spmd-to-shmem",
    /*displayName=*/"OpenSHMEM",
    /*prefix=*/"shmem_",
    /*jsonVariable=*/"SHMEM_JSON_PATH",
    /*jsonFile=*/"includesProgramModels/shmem/api.json",
    /*teamIsPointer=*/true,
    /*worldTeamVariable=*/openshmem::kTeamWorld,
    // spmd.commNode, the PEs of this node, is the team of the PEs that share
    // memory with this one.
    /*nodeTeamVariable=*/openshmem::kTeamShared,
    /*worldTeam=*/0,
    /*nodeTeam=*/0,
    /*threadLevel=*/threadLevel,
    /*signalOp=*/signalOp,
    /*success=*/openshmem::kSuccess,
    /*returnTypeOverride=*/openshmem::lookupReturnTypeOverride,
    // OpenSHMEM makes the program's global and static variables symmetric.
    /*globalsAreSymmetric=*/true,
};

} // namespace

LogicalResult checkSPMDToSHMEM(ModuleOp module) {
  return checkSPMDToSHMEMFamily(module, kOpenSHMEM);
}

LogicalResult lowerSPMDToSHMEM(ModuleOp module, StructureLowering structure) {
  return lowerSPMDToSHMEMFamily(module, kOpenSHMEM, structure);
}

namespace {

struct LowerSPMDtoSHMEMPass
    : public impl::LowerSPMDtoSHMEMBase<LowerSPMDtoSHMEMPass> {
  using LowerSPMDtoSHMEMBase::LowerSPMDtoSHMEMBase;

  void runOnOperation() override {
    if (failed(lowerSPMDToSHMEM(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
