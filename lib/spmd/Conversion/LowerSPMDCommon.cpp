// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDCommon.cpp - Shared pieces of the SPMD lowerings ----------===//

#include "LowerSPMDCommon.h"

#include "spmd/SPMDDialect.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir::spmd {

bool isTaggedFor(Operation *op, ProgrammingModel model) {
  std::optional<int> usedModel = getUsedModel(op);
  return usedModel && *usedModel == static_cast<int>(model);
}

bool isInDeviceCode(Operation *op) {
  if (op->getParentOfType<gpu::GPUModuleOp>() ||
      op->getParentOfType<gpu::LaunchOp>())
    return true;
  auto func = dyn_cast<LLVM::LLVMFuncOp>(op);
  if (!func)
    func = op->getParentOfType<LLVM::LLVMFuncOp>();
  std::optional<StringRef> cpu = func ? func.getTargetCpu() : std::nullopt;
  return cpu && cpu->starts_with("sm_");
}

LogicalResult checkUsedModels(ModuleOp module, StringRef passName) {
  bool invalid = false;
  module.walk([&](Operation *op) {
    if (!isSpmdOp(op))
      return;
    std::optional<int> usedModel = getUsedModel(op);
    if (!usedModel)
      return;
    if (*usedModel < static_cast<int>(ProgrammingModel::MPI) ||
        *usedModel > static_cast<int>(ProgrammingModel::OMP)) {
      op->emitError() << passName << ": invalid 'usedModel' value "
                      << *usedModel << " (expected "
                      << static_cast<int>(ProgrammingModel::MPI) << " to "
                      << static_cast<int>(ProgrammingModel::OMP) << ")";
      invalid = true;
    }
  });
  return failure(invalid);
}

namespace {

/// The reduction \p op reduces with, when that is one whose result on integers
/// depends on their signedness.
ConstOpTypeOp getSignednessDependentReduction(Operation *op) {
  for (Value operand : op->getOperands()) {
    auto reduction = operand.getDefiningOp<ConstOpTypeOp>();
    if (!reduction)
      continue;
    std::string name = reduction.getStringAttr().upper();
    if (llvm::is_contained({"MAX", "MIN", "MAXLOC", "MINLOC", "AVG"}, name))
      return reduction;
  }
  return nullptr;
}

} // namespace

bool reducesBySignedness(Operation *op) {
  return static_cast<bool>(getSignednessDependentReduction(op));
}

bool reducesOverUnknownSignedness(
    Operation *op, std::string *reason,
    llvm::function_ref<bool(Type)> isOwnDatatype) {
  ConstOpTypeOp reduction = getSignednessDependentReduction(op);
  if (!reduction)
    return false;
  for (Value operand : op->getOperands()) {
    auto datatype = operand.getDefiningOp<ConstDatatypeOp>();
    if (!datatype)
      continue;
    auto integer = dyn_cast<IntegerType>(datatype.getTypeAttr());
    if (!integer || !integer.isSignless() ||
        (isOwnDatatype && isOwnDatatype(integer)))
      continue;
    if (reason) {
      llvm::raw_string_ostream os(*reason);
      os << "reduces with '" << reduction.getStringAttr() << "' over "
         << integer
         << ", whose signedness the result depends on and the datatype does "
            "not record";
    }
    return true;
  }
  return false;
}

SPMDLoweringTypeConverter::SPMDLoweringTypeConverter(MLIRContext *ctx) {
  addConversion([](Type type) { return type; });
  auto materialize = [](OpBuilder &builder, Type type, ValueRange inputs,
                        Location loc) -> Value {
    // A buffer the raiser saw through a memref view reaches the runtime as the
    // address the view starts at.  That is what enzymexla's own bridge ops
    // say, and -convert-polygeist-to-llvm lowers them; it crashes on an
    // unrealized cast between the two.  Folded as they are built, so that a
    // view of a pointer is that pointer again whether or not a later
    // conversion in the same module gets to fold it.
    if (inputs.size() == 1) {
      Type inputType = inputs.front().getType();
      if (isa<MemRefType>(inputType) && isa<LLVM::LLVMPointerType>(type))
        return builder.createOrFold<enzymexla::Memref2PointerOp>(
            loc, type, inputs.front());
      if (isa<LLVM::LLVMPointerType>(inputType) && isa<MemRefType>(type))
        return builder.createOrFold<enzymexla::Pointer2MemrefOp>(
            loc, type, inputs.front());
    }
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  };
  addSourceMaterialization(materialize);
  addTargetMaterialization(materialize);
}

namespace {

// spmd.castSPMDErrorToI32 and spmd.castThreadLevelToI32 carry no usedModel:
// they go with the operation whose result they cast.
bool castsValueOf(Value input, ProgrammingModel model) {
  Operation *def = input.getDefiningOp();
  return def && isTaggedFor(def, model);
}

template <typename CastOp>
struct LowerIntegerCast : public OpConversionPattern<CastOp> {
  using OpConversionPattern<CastOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastOp op, typename CastOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value input = adaptor.getOperands().front();
    if (input.getType() != op.getResult().getType())
      return rewriter.notifyMatchFailure(op, "operand not lowered to i32");
    rewriter.replaceOp(op, input);
    return success();
  }
};

// -convert-apiCalls-to-spmd moves main's body into a process-level
// spmd.parallel on the world communicator, and ends every path out of it with
// a barrier that matches only itself, followed by an spmd.parallel_yield of
// main's return value.  No programming model has a call for either: the team
// is the set of processes the launcher starts.  The region is spliced back
// into the block around it, every yield becomes a branch to the operations
// after the spmd.parallel, and the barrier in front of a yield is dropped.
struct LowerProcessParallel : public OpConversionPattern<ParallelOp> {
  using OpConversionPattern<ParallelOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ParallelOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getIsProcessLevel())
      return rewriter.notifyMatchFailure(
          op, "a thread-level parallel region has no runtime call");
    if (op.getNumThreads() || !op.getPrivateVars().empty() ||
        !op.getReductionVars().empty())
      return rewriter.notifyMatchFailure(
          op, "clauses on a process-level parallel region");

    SmallVector<ParallelYieldOp> yields;
    for (Block &block : op.getRegion())
      if (auto yield = dyn_cast_or_null<ParallelYieldOp>(
              block.empty() ? nullptr : &block.back()))
        yields.push_back(yield);

    for (ParallelYieldOp yield : yields) {
      // The barrier belongs to the same team as the parallel it ends, so it
      // carries the parallel's own usedModel, whichever model that is.
      auto barrier = dyn_cast_or_null<BarrierOp>(yield->getPrevNode());
      if (barrier && barrier.getIsUniqueMatching().value_or(false) &&
          getUsedModel(barrier) == getUsedModel(op) && barrier->use_empty())
        rewriter.eraseOp(barrier);
    }

    Location loc = op.getLoc();
    Block *opBlock = op->getBlock();
    Block *afterBlock = rewriter.splitBlock(opBlock, op->getIterator());
    Block *continueBlock = afterBlock;
    if (op->getNumResults()) {
      continueBlock =
          rewriter.createBlock(afterBlock, op->getResultTypes(),
                               SmallVector<Location>(op->getNumResults(), loc));
      LLVM::BrOp::create(rewriter, loc, ValueRange(), afterBlock);
    }

    Block *entry = &op.getRegion().front();
    rewriter.inlineRegionBefore(op.getRegion(), continueBlock);
    rewriter.setInsertionPointToEnd(opBlock);
    LLVM::BrOp::create(rewriter, loc, ValueRange(), entry);

    for (ParallelYieldOp yield : yields) {
      rewriter.setInsertionPoint(yield);
      rewriter.replaceOpWithNewOp<LLVM::BrOp>(yield, yield->getOperands(),
                                              continueBlock);
    }

    rewriter.replaceOp(op, continueBlock->getArguments());
    return success();
  }
};

} // namespace

LogicalResult applySPMDLowering(ModuleOp module, ProgrammingModel model,
                                const TypeConverter &converter,
                                RewritePatternSet &&patterns,
                                function_ref<bool(Operation *)> alsoLowers,
                                ArrayRef<StringRef> targetDialects) {
  MLIRContext *ctx = module.getContext();
  ConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  for (StringRef dialect : targetDialects)
    target.addLegalDialect(dialect);
  target.addLegalOp<UnrealizedConversionCastOp>();
  target.addDynamicallyLegalDialect<SPMDDialect>(
      [model, alsoLowers](Operation *op) {
        return !isTaggedFor(op, model) && !(alsoLowers && alsoLowers(op));
      });
  target.addDynamicallyLegalOp<CastSPMDErrorToI32Op>(
      [model](CastSPMDErrorToI32Op cast) {
        return !castsValueOf(cast.getError(), model);
      });
  target.addDynamicallyLegalOp<CastThreadLevelToI32Op>(
      [model](CastThreadLevelToI32Op cast) {
        return !castsValueOf(cast.getThreadLevel(), model);
      });
  return applyPartialConversion(module, target, std::move(patterns));
}

void populateSPMDStructureLoweringPatterns(const TypeConverter &converter,
                                           RewritePatternSet &patterns,
                                           ProgrammingModel model,
                                           StructureLowering structure) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<LowerIntegerCast<CastSPMDErrorToI32Op>,
               LowerIntegerCast<CastThreadLevelToI32Op>>(converter, ctx);
  if (structure == StructureLowering::PerModel)
    patterns.add<LowerProcessParallel>(converter, ctx);
}

LogicalResult lowerProcessParallels(ModuleOp module,
                                    ArrayRef<ProgrammingModel> models) {
  MLIRContext *ctx = module.getContext();
  ConversionTarget target(*ctx);
  target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });
  target.addDynamicallyLegalOp<ParallelOp>([models](ParallelOp op) {
    if (!op.getIsProcessLevel())
      return true;
    std::optional<int> model = getUsedModel(op);
    return !model ||
           !llvm::is_contained(models, static_cast<ProgrammingModel>(*model));
  });

  SPMDLoweringTypeConverter converter(ctx);
  RewritePatternSet patterns(ctx);
  patterns.add<LowerProcessParallel>(converter, ctx);
  return applyPartialConversion(module, target, std::move(patterns));
}

FailureOr<LLVM::LLVMFuncOp>
getOrDeclareRuntimeFunction(OpBuilder &builder, Operation *anchor,
                            StringRef name, LLVM::LLVMFunctionType type) {
  auto module = anchor->getParentOfType<ModuleOp>();
  if (Operation *symbol = module.lookupSymbol(name)) {
    auto existing = dyn_cast<LLVM::LLVMFuncOp>(symbol);
    if (!existing)
      return anchor->emitOpError()
             << "needs @" << name
             << " to be an LLVM function, but the module defines it as '"
             << symbol->getName() << "'";
    if (existing.getFunctionType() != type)
      return anchor->emitOpError() << "needs @" << name << " declared as "
                                   << type << ", but the module declares it as "
                                   << existing.getFunctionType();
    return existing;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  return LLVM::LLVMFuncOp::create(builder, module.getLoc(), name, type);
}

FailureOr<LLVM::CallOp> emitRuntimeCall(OpBuilder &builder, Operation *anchor,
                                        Location loc, StringRef callee,
                                        Type returnType, ValueRange args) {
  SmallVector<Type> argTypes;
  for (Value arg : args)
    argTypes.push_back(arg.getType());
  FailureOr<LLVM::LLVMFuncOp> func = getOrDeclareRuntimeFunction(
      builder, anchor, callee,
      LLVM::LLVMFunctionType::get(returnType, argTypes));
  if (failed(func))
    return failure();
  return LLVM::CallOp::create(builder, loc, *func, args);
}

Value createI32Constant(OpBuilder &builder, Location loc, int32_t value) {
  return LLVM::ConstantOp::create(builder, loc, builder.getI32Type(),
                                  builder.getI32IntegerAttr(value));
}

Value createI64Constant(OpBuilder &builder, Location loc, int64_t value) {
  return LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                  builder.getI64IntegerAttr(value));
}

Value createStackSlot(OpBuilder &builder, Location loc, Type type) {
  OpBuilder::InsertionGuard guard(builder);
  if (Block *block = builder.getInsertionBlock()) {
    // The region of the nearest op that owns stack allocations: a function,
    // or a region such as a thread team's, whose allocas are per thread.
    Region *region = block->getParent();
    while (
        region && region->getParentOp() &&
        !region->getParentOp()->hasTrait<OpTrait::AutomaticAllocationScope>())
      region = region->getParentOp()->getParentRegion();
    if (region && !region->empty())
      builder.setInsertionPointToStart(&region->front());
  }
  Value one = createI32Constant(builder, loc, 1);
  return LLVM::AllocaOp::create(
      builder, loc, LLVM::LLVMPointerType::get(builder.getContext()), type, one,
      /*alignment=*/0);
}

} // namespace mlir::spmd
