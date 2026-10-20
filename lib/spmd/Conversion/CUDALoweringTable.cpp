// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CUDALoweringTable.cpp - The CUDA lowering, read off the raiser -----===//

#include "CUDALoweringTable.h"

#include "cudaSpmdEntries.h"

#include "llvm/Support/ErrorHandling.h"

namespace mlir::spmd {

namespace {

//===----------------------------------------------------------------------===//
// The CUDA ABI of one argument
//===----------------------------------------------------------------------===//

bool classifyCUDAArg(ArgSlot &slot, CallDescription &) {
  switch (slot.source) {
  case ArgSource::Operand:
    slot.abiKind = static_cast<int>(CUDAArgKind::In);
    return true;

  case ArgSource::Result:
  case ArgSource::Unused:
    // Neither call the raiser maps has an out-parameter or a parameter no
    // operand feeds, so there is no ABI rule to write and no way to test one.
    // Returning false drops such a call from the table rather than passing
    // storage or a zero the lowering has not thought about: if the raiser
    // grows an entry with one -- cudaMalloc writes the allocation through a
    // void ** -- it is better for the call to be missing, and reported as
    // such, than to be emitted wrongly.
    return false;
  }
  llvm_unreachable("unknown ArgSource");
}

/// Every callee in the map is the one its operation lowers to: CUDA's two
/// raised calls raise to an operation each.
bool isCanonicalCUDACallee(StringRef, const APICallEntry &) { return true; }

StringRef cudaKindName(int abiKind) {
  switch (static_cast<CUDAArgKind>(abiKind)) {
  case CUDAArgKind::In:
    return "in";
  }
  llvm_unreachable("unknown CUDAArgKind");
}

} // namespace

const TableSpec &getCUDATableSpec() {
  static const TableSpec spec = {
      /*model=*/"cuda",
      /*displayName=*/"CUDA",
      /*entries=*/&cudaToSpmdOperands,
      /*classify=*/classifyCUDAArg,
      /*isCanonical=*/isCanonicalCUDACallee,
      // Never consulted: spmd.memcpy and spmd.memset carry no $isBlocking (and
      // no $isNonBlocking), so callsInBlockingForm keeps every candidate, and
      // each of the two operations has exactly one.  USIS is what this would
      // invert to -- it calls cudaMemset non-blocking, because it only
      // enqueues onto the null stream -- so Execution is the form to name.
      /*blockingForm=*/BlockingForm::Execution,
      // The dialect gives neither operation a request, and CUDA's asynchronous
      // forms (cudaMemcpyAsync, cudaMemsetAsync) are not raised, so an
      // operation that did carry one is to be reported rather than quietly
      // made blocking.
      /*allowsRequestNull=*/false,
      /*alsoSameInjected=*/nullptr,
      // Neither operation has a stride operand, so no stride diagnostic can
      // arise for CUDA.
      /*contiguousOnlyNote=*/"",
      /*extraOperandNotes=*/{},
      /*missingOperandNotes=*/{},
  };
  return spec;
}

ArrayRef<CallDescription> getCUDALoweringTable() {
  return getLoweringTable(getCUDATableSpec());
}

const CallDescription *selectCUDACall(Operation *op, OpBuilder &builder,
                                      SmallVectorImpl<std::string> &reasons) {
  return selectCall(getCUDATableSpec(), op, builder, reasons);
}

void printCUDALoweringTable(raw_ostream &os, OpBuilder &builder) {
  printLoweringTable(getCUDATableSpec(), os, builder, cudaKindName);
}

} // namespace mlir::spmd
