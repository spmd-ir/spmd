// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Transforms/Passes.h"
#include "src/enzyme_ad/jax/RegistryUtils.h"
#include "llvm/Support/TargetSelect.h"

#include "spmd/SPMDDialect.h"
#include "spmd/SPMDPasses.h"

#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/RegistryUtils.h"
#include "llvm/Support/TargetSelect.h"

// Defined only by //test/lib:spmd-opt, the build the unit tests run.
#ifdef SPMD_INCLUDE_TEST_PASSES
#include "test/lib/TestPasses.h"
#endif

int main(int argc, char **argv) {
  const char *exitCodeEnvVar = std::getenv("SPMD_IR_EXITCODE");
  int userExitcode = exitCodeEnvVar ? std::atoi(exitCodeEnvVar) : 1;
  // Register dialects that may be part of input to spmd-opt
  llvm::InitializeNativeTarget();
  llvm::InitializeNativeTargetAsmPrinter();
  mlir::DialectRegistry registry;

  mlir::enzyme::prepareRegistry(registry);
  mlir::enzyme::registerDialects(registry);
  mlir::enzyme::registerInterfaces(registry);
  registry.addExtension(+[](mlir::MLIRContext *ctx,
                            mlir::enzymexla::EnzymeXLADialect *) {
    ctx->getOrLoadDialect<mlir::gpu::GPUDialect>(); // due to custom op parser
                                                    // (memcpy by enzymexla)
    // ctx->loadAllAvailableDialects();
  });
  registry.insert<mlir::spmd::SPMDDialect>();
  // Add the following to include *all* MLIR Core dialects, or selectively
  // include what you need like above. You only need to register dialects that
  // will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);
  mlir::spmd::registerPasses();
  mlir::enzyme::initializePasses();
  mlir::registerTransformsPasses();
#ifdef SPMD_INCLUDE_TEST_PASSES
  mlir::spmd::test::registerTestPasses();
#endif

  auto res = mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "SPMD optimizer driver\n", registry));
  return res ? userExitcode : res;
}
