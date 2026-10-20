// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ompSpmdEntries.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"
#include "llvm/Support/Debug.h"

namespace mlir::spmd {

/// Returns an spmd.getParallelComm op result, used as the default communicator
/// operand for all OMP runtime calls. The canonicalizer folds this to the
/// concrete comm value of the enclosing spmd.parallel.
static Value getParallelComm(OpBuilder &builder, Location loc, int modelGroup) {
  return builder.create<spmd::GetParallelCommOp>(loc).getComm();
}

const ApiCallToSpmdMap createOmpToSpmdOperands() {
  ApiCallToSpmdMap map = {

      // omp_get_thread_num() -> spmd.getRankInComm(comm)
      // Returns the thread number within the current team, i.e. the rank within
      // the sub-communicator created by the enclosing spmd.parallel.
      {"omp_get_thread_num",
       {"spmd.getRankInComm", Operands(D("comm", getParallelComm)),
        Results(RR("rank", 0)), newAttributes(), postProcessing()}},

      // omp_get_num_threads() -> spmd.getSizeOfComm(comm)
      // Returns the number of threads in the current team, i.e. the size of the
      // sub-communicator of the enclosing spmd.parallel.
      {"omp_get_num_threads",
       {"spmd.getSizeOfComm", Operands(D("comm", getParallelComm)),
        Results(RR("size", 0)), newAttributes(), postProcessing()}},

      // omp_get_max_threads() -> spmd.getSizeOfComm(comm)
      // Conservative approximation: maps to the current team size.
      // In principle omp_get_max_threads may be called outside a parallel
      // region
      // and return a global thread limit — the getParallelComm default handles
      // that by falling back to commWorld when no parallel is enclosing.
      // todo
      // {"omp_get_max_threads", {
      //   "spmd.getSizeOfComm",
      //   Operands(D("comm", getParallelComm)),
      //   Results(RR("size", 0)),
      //   newAttributes(),
      //   postProcessing()
      // }},

  };

  return map;
}

const ApiCallToSpmdMap ompToSpmdOperands = createOmpToSpmdOperands();

} // namespace mlir::spmd