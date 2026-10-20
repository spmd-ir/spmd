// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToNVSHMEM.cpp - Lower SPMD operations to NVSHMEM ----------===//
//
// -lower-spmd-to-nvshmem: the OpenSHMEM family's lowering
// (LowerSPMDToSHMEMFamily.cpp) under the NVSHMEM ABI of
// includesProgramModels/mynvshmem/include, whose constants are
// include/spmd/TargetABI/NVSHMEM.h: a team is an int32_t, and the predefined
// teams are constants.
//
// The calls it emits are the host API those headers declare, which is the API
// the raiser reads -- so a raised NVSHMEM call lowers back to itself.  Those
// headers are a stand-in for NVIDIA's, though, and a few of their routines are
// not what NVIDIA's library exports: nvshmem_init is an inline function of
// NVIDIA's header, for one.  Mapping the stand-in's routines to the library's
// is a separate pass, -nvshmem-to-host-abi, which `spmd-verify compile` runs
// after this one; it does the same for calls the raiser never raised.
//
// Only host-initiated NVSHMEM is lowered.  An NVSHMEM operation inside a GPU
// kernel is reported: nothing here compiles device code.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDToNVSHMEM.h"
#include "LowerSPMDToSHMEMFamily.h"

#include "spmd/SPMDPasses.h"
#include "spmd/TargetABI/NVSHMEM.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"

namespace mlir::spmd {

#define GEN_PASS_DEF_LOWERSPMDTONVSHMEM
#include "spmd/SPMDPasses.h.inc"

namespace {

namespace nvshmem = abi::nvshmem;

std::optional<int32_t> threadLevel(StringRef name) {
  return nvshmem::lookupHandle(nvshmem::kThreadLevels, name);
}
std::optional<int32_t> signalOp(StringRef name) {
  return nvshmem::lookupHandle(nvshmem::kSignalOps, name);
}
// The stand-in headers and USIS agree on every return type.
std::optional<StringRef> noReturnTypeOverride(StringRef) {
  return std::nullopt;
}

const SHMEMTarget kNVSHMEM = {
    /*model=*/ProgrammingModel::NVSHMEM,
    /*flavour=*/SHMEMFlavour::NVSHMEM,
    /*passName=*/"lower-spmd-to-nvshmem",
    /*displayName=*/"NVSHMEM",
    /*prefix=*/"nvshmem_",
    /*jsonVariable=*/"NVSHMEM_JSON_PATH",
    /*jsonFile=*/"includesProgramModels/mynvshmem/api.json",
    /*teamIsPointer=*/false,
    /*worldTeamVariable=*/"",
    /*nodeTeamVariable=*/"",
    /*worldTeam=*/nvshmem::kTeamWorld,
    // spmd.commNode is what the raiser makes of NVSHMEMX_TEAM_NODE.
    /*nodeTeam=*/nvshmem::kTeamNode,
    /*threadLevel=*/threadLevel,
    /*signalOp=*/signalOp,
    /*success=*/nvshmem::kSuccess,
    /*returnTypeOverride=*/noReturnTypeOverride,
    // Another PE reaches only the symmetric heap nvshmem_malloc allocates on
    // the GPU; a global variable is host memory of this process.
    /*globalsAreSymmetric=*/false,
};

} // namespace

LogicalResult checkSPMDToNVSHMEM(ModuleOp module) {
  return checkSPMDToSHMEMFamily(module, kNVSHMEM);
}

LogicalResult lowerSPMDToNVSHMEM(ModuleOp module, StructureLowering structure) {
  return lowerSPMDToSHMEMFamily(module, kNVSHMEM, structure);
}

namespace {

struct LowerSPMDtoNVSHMEMPass
    : public impl::LowerSPMDtoNVSHMEMBase<LowerSPMDtoNVSHMEMPass> {
  using LowerSPMDtoNVSHMEMBase::LowerSPMDtoNVSHMEMBase;

  void runOnOperation() override {
    if (failed(lowerSPMDToNVSHMEM(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
