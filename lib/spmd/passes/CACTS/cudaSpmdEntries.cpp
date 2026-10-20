// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "cudaSpmdEntries.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir::spmd {

const ApiCallToSpmdMap createCudaToSpmdOperands() {
  ApiCallToSpmdMap map = {

      {"cudaMemcpy",
       {"spmd.memcpy",
        Operands(O("dst", 0), O("src", 1), O("count", 2), O("kind", 3)),
        Results(), newAttributes(), postProcessing()}},

      {"cudaMemset",
       {"spmd.memset", Operands(O("dst", 0), O("value", 1), O("count", 2)),
        Results(), newAttributes(), postProcessing()}},

      // {"cudaMalloc", {"spmd.malloc",
      //                 Operands(D("comm", getWorldComm),
      //                          O("size",    0),
      //                          O("baseptr", 1)),
      //                 Results(),
      //                 newAttributes(),
      //                 postProcessing()}},

      // {"cudaFree", {"spmd.free",
      //               Operands(D("comm", getWorldComm),
      //                        O("baseptr", 0)),
      //               Results(),
      //               newAttributes(),
      //               postProcessing()}},

      // {"cudaDeviceSynchronize", {"spmd.barrier",
      //                             Operands(D("comm", getWorldComm)),
      //                             Results(),
      //                             newAttributes(),
      //                             postProcessing()}},

  };
  return map;
}

const ApiCallToSpmdMap cudaToSpmdOperands = createCudaToSpmdOperands();

} // namespace mlir::spmd