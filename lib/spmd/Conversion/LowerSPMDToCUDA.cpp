// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToCUDA.cpp - Lower SPMD operations to CUDA calls ----------===//
//
// Every operation tagged for CUDA becomes a call to the CUDA runtime routine
// it models, under the ABI of CUDA 13.1's cuda_runtime_api.h: a buffer is a
// pointer, a count the size_t the dialect holds as an i64, a direction and a
// fill byte the ints the program computed, and every routine returns a
// cudaError_t, which is an int (include/spmd/TargetABI/CUDA.h).
//
// The calls are derived: CUDALoweringTable reads them from the CACTS map the
// raiser reads and the USIS API description.  There is nothing left for this
// file to keep, which is why it is the shortest of the lowerings -- CUDA's
// calls are the regular part all the way down.  What it does have is the
// scope:
//
// * -convert-apiCalls-to-spmd raises two of CUDA's ~330 runtime routines,
//   cudaMemcpy and cudaMemset (lib/spmd/passes/CACTS/cudaSpmdEntries.cpp), and
//   this lowering covers exactly those.  Everything else the program calls is
//   still the llvm.call the frontend produced and passes through untouched:
//   an operation that was never raised needs no lowering.  That is what makes
//   the pass worth having -- an NCCL or NVSHMEM program calls cudaMemset, and
//   until it lowers, `spmd-verify compile` stops at the kept spmd.memset and
//   no such program reaches a binary.
//
// * CUDA kernels are not in scope.  `compile` does no GPU kernel outlining,
//   and neither the raiser nor this pass sees a launch: a .cu program is
//   refused earlier, not lowered here.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDToCUDA.h"
#include "CUDALoweringTable.h"
#include "LowerSPMDCommon.h"

#include "spmd/SPMDOps.h"
#include "spmd/SPMDPasses.h"
#include "spmd/TargetABI/CUDA.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"

#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"

#include <string>

namespace mlir::spmd {

#define GEN_PASS_DEF_LOWERSPMDTOCUDA
#include "spmd/SPMDPasses.h.inc"

namespace {

namespace cuda = abi::cuda;

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// A buffer is a pointer and a cudaError_t is a C enum, which is an int.  No
// handle type of the dialect reaches CUDA: the two operations take addresses
// and integers only.
class CUDATypeConverter : public SPMDLoweringTypeConverter {
public:
  explicit CUDATypeConverter(MLIRContext *ctx)
      : SPMDLoweringTypeConverter(ctx) {
    auto i32 = IntegerType::get(ctx, 32);
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    addConversion([i32](ErrorType) -> Type { return i32; });
    addConversion([ptr](MemRefType) -> Type { return ptr; });
  }
};

//===----------------------------------------------------------------------===//
// Calls
//===----------------------------------------------------------------------===//

/// One operation lowered to the CUDA call CUDALoweringTable picks for it: the
/// slots of that call say, argument by argument, what to pass.  Every slot is
/// an operand, so there is no stack storage to make and nothing to load back;
/// the call's cudaError_t is the operation's only result.
struct CUDATableCallPattern : public ConversionPattern {
  CUDATableCallPattern(const TypeConverter &converter, StringRef rootName,
                       MLIRContext *ctx)
      : ConversionPattern(converter, rootName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<std::string> reasons;
    const CallDescription *call = selectCUDACall(op, rewriter, reasons);
    if (!call)
      return rewriter.notifyMatchFailure(
          op, reasons.empty() ? "no CUDA call of this shape" : reasons.front());

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();

    SmallVector<Value> args;
    for (const ArgSlot &slot : call->args) {
      if (static_cast<CUDAArgKind>(slot.abiKind) != CUDAArgKind::In)
        return rewriter.notifyMatchFailure(op, "unknown argument kind");
      if (!hasSlotOperand(op, slot))
        return rewriter.notifyMatchFailure(op, "missing operand");
      args.push_back(operands[slot.operand]);
    }
    if (call->returnKind != "ERROR_CODE")
      return rewriter.notifyMatchFailure(op, "the call returns no status");

    FailureOr<LLVM::CallOp> emitted =
        emitRuntimeCall(rewriter, op, loc, call->callee, i32, args);
    if (failed(emitted))
      return failure();

    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected number of results");
    if (emitted->getResult().getType() !=
        getTypeConverter()->convertType(op->getResultTypes().front()))
      return rewriter.notifyMatchFailure(op, "result of the wrong type");
    rewriter.replaceOp(op, emitted->getResult());
    return success();
  }
};

void populateCUDACallPatterns(const TypeConverter &c, RewritePatternSet &p) {
  SmallVector<StringRef> lowered;
  for (const CallDescription &call : getCUDALoweringTable())
    if (call.canonical && !llvm::is_contained(lowered, call.spmdOpName))
      lowered.push_back(call.spmdOpName);
  for (StringRef spmdOpName : lowered)
    p.add<CUDATableCallPattern>(c, spmdOpName, p.getContext());
}

//===----------------------------------------------------------------------===//
// The checks
//===----------------------------------------------------------------------===//

/// The barrier the raiser puts before an spmd.parallel_yield so the region
/// matches itself.  LowerProcessParallel erases it with the parallel.
bool isProcessTeamBarrier(Operation *op) {
  auto barrier = dyn_cast<BarrierOp>(op);
  if (!barrier || !barrier.getIsUniqueMatching().value_or(false))
    return false;
  auto parallel = dyn_cast_or_null<ParallelOp>(barrier->getParentOp());
  return parallel && parallel.getIsProcessLevel() &&
         isa_and_present<ParallelYieldOp>(barrier->getNextNode()) &&
         getUsedModel(parallel) == getUsedModel(barrier);
}

/// Whether the structure patterns lower \p op rather than the table.  CUDA's
/// map has no init and no finalize, so -convert-apiCalls-to-spmd never wraps a
/// process-level spmd.parallel around a CUDA program -- but
/// -spmd-retarget-model and -lower-spmd's models= can hand this pass one, and
/// populateSPMDStructureLoweringPatterns does lower it.  Reporting it as
/// something CUDA has no call for would be wrong: it is not a call at all.
bool isLoweredByHand(Operation *op) {
  if (auto parallel = dyn_cast<ParallelOp>(op))
    return parallel.getIsProcessLevel();
  return isProcessTeamBarrier(op);
}

bool hasOperationTaggedForCUDA(ModuleOp module) {
  return module
      .walk([](Operation *op) {
        return isTaggedFor(op, ProgrammingModel::CUDA) ? WalkResult::interrupt()
                                                       : WalkResult::advance();
      })
      .wasInterrupted();
}

/// Every operation CUDA cannot express at all: one no call of the raiser's map
/// raises to.  The message names the map rather than CUDA itself, because the
/// routine may well exist -- the scope is what the raiser maps, not what the
/// runtime has.
LogicalResult checkExpressible(Operation *op) {
  if (isLoweredByHand(op))
    return success();
  if (getCallsFor(getCUDATableSpec(), op->getName().getStringRef()).empty())
    return op->emitOpError()
           << "has no CUDA counterpart; -convert-apiCalls-to-spmd raises "
              "cudaMemcpy and cudaMemset, and the lowering covers what it "
              "raises";
  return success();
}

/// A $kind the program fixed to something that is no direction.  Only a
/// constant is checked: a direction the program computes -- from whether a
/// pointer is on the device, say -- is its own business, and passes through.
LogicalResult checkMemcpyKind(Operation *op) {
  auto memcpy = dyn_cast<MemcpyOp>(op);
  if (!memcpy)
    return success();
  APInt kind;
  if (!matchPattern(memcpy.getKind(), m_ConstantInt(&kind)))
    return success();
  if (cuda::lookupMemcpyKind(kind.getSExtValue()))
    return success();
  return memcpy.emitOpError()
         << "copies with kind " << kind.getSExtValue()
         << ", which is no direction of enum cudaMemcpyKind; cudaMemcpy would "
            "return cudaErrorInvalidMemcpyDirection";
}

/// An operand the dialect has and no CUDA call has a parameter for, and one
/// CUDA needs and the operation has not.  Both are what the CACTS entry says
/// read backwards, so the table finds them.
LogicalResult checkCallable(Operation *op) {
  OpBuilder builder(op->getContext());
  SmallVector<std::string> reasons;
  selectCUDACall(op, builder, reasons);
  for (const std::string &reason : reasons)
    op->emitOpError() << reason;
  return failure(!reasons.empty());
}

} // namespace

LogicalResult checkSPMDToCUDA(ModuleOp module) {
  if (failed(checkUsedModels(module, "lower-spmd-to-cuda")))
    return failure();
  // The calls are derived from the same API description the raiser reads, so
  // the lowering needs it too, but only for a module that has something to
  // lower: -lower-spmd runs this check for every model of its table, and CUDA
  // is in the default one.  The module is asked first because building the
  // table complains on stderr when CUDA_JSON_PATH is unset, which would then
  // greet every run that has no CUDA in it at all.
  if (hasOperationTaggedForCUDA(module) && getCUDALoweringTable().empty())
    return emitError(module.getLoc())
           << "lower-spmd-to-cuda: no CUDA API description; set "
              "CUDA_JSON_PATH to includesProgramModels/cuda/api.json";

  bool inexpressible = false;
  module.walk([&](Operation *op) {
    if (!isTaggedFor(op, ProgrammingModel::CUDA))
      return;
    // All of them run, so an operation wrong on more than one count reports
    // every reason at once.
    bool expressible = succeeded(checkExpressible(op));
    bool callable = succeeded(checkCallable(op));
    bool kind = succeeded(checkMemcpyKind(op));
    if (!expressible || !callable || !kind)
      inexpressible = true;
  });
  return failure(inexpressible);
}

LogicalResult lowerSPMDToCUDA(ModuleOp module, StructureLowering structure) {
  MLIRContext *ctx = module.getContext();

  if (failed(checkSPMDToCUDA(module)))
    return failure();
  // Nothing tagged for CUDA means nothing to convert -- not even a cast, since
  // the only !spmd.error a cast of this model's could read comes from one of
  // these operations.  The early return is what keeps building the table, and
  // the "CUDA_JSON_PATH env-var not set" it prints when the description is
  // missing, out of every run of -lower-spmd that has no CUDA in it: CUDA is
  // in the default table, so this pass runs on every module.
  if (!hasOperationTaggedForCUDA(module))
    return success();

  CUDATypeConverter converter(ctx);
  RewritePatternSet patterns(ctx);
  populateCUDACallPatterns(converter, patterns);
  // CUDA's map has no init and no finalize, so -convert-apiCalls-to-spmd never
  // wraps a process-level spmd.parallel around a CUDA program; the structure
  // patterns are here for the casts of the results, and for a module whose
  // CUDA operations sit inside another model's parallel.
  populateSPMDStructureLoweringPatterns(converter, patterns,
                                        ProgrammingModel::CUDA, structure);

  return applySPMDLowering(module, ProgrammingModel::CUDA, converter,
                           std::move(patterns));
}

namespace {

struct LowerSPMDtoCUDAPass
    : public impl::LowerSPMDtoCUDABase<LowerSPMDtoCUDAPass> {
  using LowerSPMDtoCUDABase::LowerSPMDtoCUDABase;

  void runOnOperation() override {
    if (failed(lowerSPMDToCUDA(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
