// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToSHMEMFamily.cpp - Lower to OpenSHMEM or NVSHMEM ---------===//
//
// Every operation tagged for a model of the OpenSHMEM family becomes a call to
// the routine it models, under the ABI its SHMEMTarget describes: Sandia
// OpenSHMEM's (includesProgramModels/shmem/include), where a team is a pointer
// and the predefined teams are extern variables, or NVSHMEM's
// (includesProgramModels/mynvshmem/include), where a team is an int32_t and
// the predefined teams are constants.  In both, the symmetric heap is what a
// window is.
//
// No call is described here.  SHMEMLoweringTable derives every one of them
// from the CACTS map the raiser reads and the USIS API description, so that
// the argument order exists once rather than once per direction; this file
// turns one of its descriptions into the call and keeps what neither source
// knows: the ABI values, the window model, and the operations the family has
// no counterpart for.
//
// Three things are worth knowing about the shape of this lowering.
//
// * Both models put the element type and the reduction operation in the
//   function name.  The lowering does not choose them: the CACTS map says, for
//   each of the ~1300 names, which spmd.constDatatype and which
//   spmd.constOpType the raiser injects for it, so the name follows from the
//   operation's operands.  A datatype or a reduction the model has no name for
//   is reported, not rounded to the nearest one.
//
// * A window is the symmetric address it is over.  !spmd.win lowers to that
//   address, spmd.winAlloc to the model's malloc and spmd.winCreate to its base
//   operand; a transfer's target address is the window, which is why the table
//   requires the target displacement to be zero -- the displacement unit is
//   the window's, and nothing of it survives the conversion.
//
// * Neither model hands out a request.  Every non-blocking operation of a PE is
//   completed by the model's quiet, so !spmd.req lowers to one value for all
//   of them and spmd.wait and spmd.waitAll are that call.  What the family has
//   no non-blocking form of at all -- the collectives -- is reported instead.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDToSHMEMFamily.h"
#include "LowerSPMDCommon.h"
#include "SHMEMLoweringTable.h"

#include "spmd/SPMDTraits.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"
#include "src/enzyme_ad/jax/Dialect/Ops.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"

#include <string>

namespace mlir::spmd {

namespace {

/// The value !spmd.req lowers to.  Neither model hands out a request: every
/// non-blocking operation a PE started is completed by the next quiet, which
/// is what spmd.wait and spmd.waitAll become, so one value stands for all of
/// them and nothing ever reads it.
constexpr int32_t kNoRequest = 0;

/// The value !spmd.datatype and !spmd.op lower to.  Neither is ever passed to
/// a call: both models spell the element type and the reduction operation in
/// the function name, which SHMEMLoweringTable picks from these operations
/// rather than from a handle.  The constant only keeps the conversion total.
constexpr int32_t kUnusedHandle = 0;

/// \p reason, a format string with the model's name as {0} and the prefix of
/// its routines as {1}, so that one text serves both models.
std::string forTarget(StringLiteral reason, const SHMEMTarget &target) {
  return llvm::formatv(reason.data(), target.displayName, target.prefix).str();
}

/// The model's routine \p name, such as "quiet": shmem_quiet or nvshmem_quiet.
std::string routine(const SHMEMTarget &target, StringRef name) {
  return (target.prefix + name).str();
}

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// A window and a symmetric address are pointers in both ABIs, and a team is a
// pointer in OpenSHMEM and an int32_t in NVSHMEM; everything else the dialect
// holds is already an integer of the width the model takes.
class SHMEMTypeConverter : public SPMDLoweringTypeConverter {
public:
  SHMEMTypeConverter(MLIRContext *ctx, const SHMEMTarget &target)
      : SPMDLoweringTypeConverter(ctx) {
    auto i32 = IntegerType::get(ctx, 32);
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    Type team = target.teamIsPointer ? Type(ptr) : Type(i32);
    addConversion([team](CommType) -> Type { return team; });
    // A window is the symmetric address it is over.
    addConversion([ptr](WinType) -> Type { return ptr; });
    addConversion([i32](ErrorType) -> Type { return i32; });
    addConversion([i32](ReqType) -> Type { return i32; });
    addConversion([i32](DatatypeType) -> Type { return i32; });
    addConversion([i32](OpType) -> Type { return i32; });
    addConversion([i32](SignalOpType) -> Type { return i32; });
    addConversion([i32](ThreadLevelType) -> Type { return i32; });
    addConversion([ptr](MemRefType) -> Type { return ptr; });
  }
};

//===----------------------------------------------------------------------===//
// What the family has no counterpart for
//===----------------------------------------------------------------------===//

struct NoCounterpart {
  StringLiteral spmdOpName;
  /// A format string for forTarget.
  StringLiteral reason;
};

// Operations neither model can express at all, reported before anything is
// rewritten so that each one arrives with its own reason instead of as the
// conversion's "failed to legalize".  The operations a *call* cannot express
// -- a stride, a displacement, a datatype -- are not here: the table finds
// those by inverting what the raiser injected.
constexpr NoCounterpart kNoCounterpart[] = {
    {"spmd.reduce", "is a rooted reduction; every PE of an {0} team "
                    "receives the result of {1}<T>_<op>_reduce"},
    {"spmd.reduceScatter",
     "has no {0} counterpart; its collectives reduce, broadcast, "
     "collect and exchange, and scatter no reduction"},
    {"spmd.scan", "is a prefix reduction, which {0} has no collective "
                  "for"},
    {"spmd.exscan", "is a prefix reduction, which {0} has no collective "
                    "for"},
    {"spmd.scatter", "distributes a different part of the buffer to each PE, "
                     "which {0} has no collective for"},
    {"spmd.gather", "collects to one PE; {1}<T>_fcollect collects to all "
                    "of them"},
    {"spmd.commSplit",
     "splits by a colour and a key; {0} splits a team by a triplet of "
     "PE numbers ({1}team_split_strided), which spmd.commSplitStrided is"},
    {"spmd.commDup", "has no {0} counterpart; a team is not duplicated"},
    {"spmd.accumulate",
     "accumulates a buffer; {0}'s atomic accumulate "
     "({1}<T>_atomic_add) is one element, which spmd.getAccumulate of a "
     "single element is"},
    {"spmd.fetchAndOp",
     "has no {0} counterpart in this shape; the fetching atomics are "
     "spmd.getAccumulate, spmd.swap and spmd.compareAndSwap"},
    {"spmd.lock", "locks a window; {0} locks a symmetric lock variable "
                  "({1}set_lock), which the dialect does not model"},
    {"spmd.lockAll", "locks a window; {0} locks a symmetric lock "
                     "variable ({1}set_lock), which the dialect does not "
                     "model"},
    {"spmd.unlock", "unlocks a window; {0} unlocks a symmetric lock "
                    "variable ({1}clear_lock), which the dialect does not "
                    "model"},
    {"spmd.unlockAll", "unlocks a window; {0} unlocks a symmetric lock "
                       "variable ({1}clear_lock), which the dialect does "
                       "not model"},
    {"spmd.winStart", "opens an access epoch on a window; {0} has no "
                      "epochs, only {1}quiet and {1}barrier_all"},
    {"spmd.winComplete", "closes an access epoch on a window; {0} has "
                         "no epochs, only {1}quiet and {1}barrier_all"},
    {"spmd.waitSome", "completes some of several requests; {1}quiet() "
                      "completes every outstanding operation of this PE and "
                      "names none of them"},
    {"spmd.waitAny", "completes one of several requests; {1}quiet() "
                     "completes every outstanding operation of this PE and "
                     "names none of them"},
    {"spmd.test", "asks whether an operation has completed; {0} only "
                  "waits for all of them ({1}quiet)"},
    {"spmd.testAll", "asks whether operations have completed; {0} only "
                     "waits for all of them ({1}quiet)"},
    {"spmd.testSome", "asks whether operations have completed; {0} only "
                      "waits for all of them ({1}quiet)"},
    {"spmd.testAny", "asks whether operations have completed; {0} only "
                     "waits for all of them ({1}quiet)"},
};

StringLiteral noCounterpartReason(Operation *op) {
  for (const NoCounterpart &entry : kNoCounterpart)
    if (entry.spmdOpName == op->getName().getStringRef())
      return entry.reason;
  return StringLiteral("");
}

//===----------------------------------------------------------------------===//
// Predefined handles
//===----------------------------------------------------------------------===//

/// The predefined team \p name, which the OpenSHMEM ABI keeps in an extern
/// variable of type shmem_team_t rather than in a constant: the module gets the
/// declaration if it has none, and the call gets what the variable holds.
Value loadPredefinedTeam(OpBuilder &builder, Operation *anchor,
                         StringRef name) {
  auto module = anchor->getParentOfType<ModuleOp>();
  auto ptr = LLVM::LLVMPointerType::get(builder.getContext());
  auto global = module.lookupSymbol<LLVM::GlobalOp>(name);
  if (!global) {
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(module.getBody());
    global = LLVM::GlobalOp::create(builder, anchor->getLoc(), ptr,
                                    /*isConstant=*/false,
                                    LLVM::Linkage::External, name, Attribute());
  }
  Location loc = anchor->getLoc();
  Value address = LLVM::AddressOfOp::create(builder, loc, global);
  return LLVM::LoadOp::create(builder, loc, ptr, address);
}

/// spmd.commWorld and spmd.commNode: the extern variable the OpenSHMEM ABI
/// keeps the team in, or the constant NVSHMEM gives it.
template <typename OpT>
struct LowerPredefinedTeam : public OpConversionPattern<OpT> {
  LowerPredefinedTeam(const TypeConverter &converter, MLIRContext *ctx,
                      const SHMEMTarget &target, bool world)
      : OpConversionPattern<OpT>(converter, ctx), target(target), world(world) {
  }

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (target.teamIsPointer)
      rewriter.replaceOp(op,
                         loadPredefinedTeam(rewriter, op,
                                            world ? target.worldTeamVariable
                                                  : target.nodeTeamVariable));
    else
      rewriter.replaceOp(
          op, createI32Constant(rewriter, op.getLoc(),
                                world ? target.worldTeam : target.nodeTeam));
    return success();
  }

private:
  const SHMEMTarget &target;
  bool world;
};

/// A handle the lowering emits as a constant, and nothing but a constant: the
/// null request and the two handles the family spells in a function name.
template <typename OpT, int32_t Value_>
struct LowerConstantHandle : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, createI32Constant(rewriter, op.getLoc(), Value_));
    return success();
  }
};

std::optional<int32_t> handleOf(ConstThreadLevelOp op,
                                const SHMEMTarget &target) {
  return target.threadLevel(op.getLevel());
}
std::optional<int32_t> handleOf(ConstSignalOpTypeOp op,
                                const SHMEMTarget &target) {
  return target.signalOp(op.getStringAttr());
}

template <typename OpT>
struct LowerNamedHandle : public OpConversionPattern<OpT> {
  LowerNamedHandle(const TypeConverter &converter, MLIRContext *ctx,
                   const SHMEMTarget &target)
      : OpConversionPattern<OpT>(converter, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<int32_t> handle = handleOf(op, target);
    if (!handle)
      return rewriter.notifyMatchFailure(op, "no predefined handle");
    rewriter.replaceOp(op, createI32Constant(rewriter, op.getLoc(), *handle));
    return success();
  }

private:
  const SHMEMTarget &target;
};

LogicalResult checkPredefinedHandle(Operation *op, const SHMEMTarget &target) {
  if (auto level = dyn_cast<ConstThreadLevelOp>(op)) {
    if (!handleOf(level, target))
      return level.emitOpError()
             << "'" << level.getLevel() << "' is not an " << target.displayName
             << " thread support level";
  } else if (auto signalOp = dyn_cast<ConstSignalOpTypeOp>(op)) {
    if (!handleOf(signalOp, target))
      return signalOp.emitOpError()
             << "'" << signalOp.getStringAttr() << "' is not an "
             << target.displayName << " signal operation";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Calls
//===----------------------------------------------------------------------===//

/// The MLIR type of the C type USIS gives the call's return, or nothing when
/// this ABI has no rule for it -- which drops the call rather than returning
/// the wrong width.
FailureOr<Type> getReturnType(OpBuilder &builder, StringRef cType) {
  cType = cType.trim();
  if (cType == "void")
    return Type(LLVM::LLVMVoidType::get(builder.getContext()));
  if (cType.ends_with("*"))
    return Type(LLVM::LLVMPointerType::get(builder.getContext()));
  if (cType == "float")
    return Type(builder.getF32Type());
  if (cType == "double")
    return Type(builder.getF64Type());
  if (cType == "long double")
    return Type(builder.getF80Type());
  for (StringRef name :
       {"char", "signed char", "unsigned char", "int8_t", "uint8_t", "_Bool"})
    if (cType == name)
      return Type(builder.getI8Type());
  for (StringRef name : {"short", "unsigned short", "int16_t", "uint16_t"})
    if (cType == name)
      return Type(builder.getI16Type());
  for (StringRef name :
       {"int", "unsigned", "unsigned int", "int32_t", "uint32_t"})
    if (cType == name)
      return Type(builder.getI32Type());
  for (StringRef name :
       {"long", "long long", "unsigned long", "unsigned long long", "int64_t",
        "uint64_t", "size_t", "ptrdiff_t"})
    if (cType == name)
      return Type(builder.getI64Type());
  return failure();
}

/// One operation lowered to the call SHMEMLoweringTable picks for it: the
/// slots of that call say, argument by argument, what to pass.
struct SHMEMTableCallPattern : public ConversionPattern {
  SHMEMTableCallPattern(const TypeConverter &converter, StringRef rootName,
                        MLIRContext *ctx, const SHMEMTarget &target)
      : ConversionPattern(converter, rootName, /*benefit=*/1, ctx),
        target(target) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<std::string> reasons;
    const CallDescription *call =
        selectSHMEMCall(target.flavour, op, rewriter, reasons);
    if (!call)
      return rewriter.notifyMatchFailure(
          op, reasons.empty() ? "no call of this shape" : reasons.front());

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    auto ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    Type team = target.teamIsPointer ? Type(ptr) : i32;

    // An out-parameter's stack slot, and which of the operation's results it
    // becomes.  The slots are in prototype order and the results in the
    // order the CACTS entry lists them, so they are put back in that order.
    struct OutParameter {
      int result;
      Value storage;
      Type type;
    };
    SmallVector<Value> args;
    SmallVector<OutParameter> outParameters;

    for (const ArgSlot &slot : call->args) {
      switch (static_cast<SHMEMArgKind>(slot.abiKind)) {
      case SHMEMArgKind::In:
        if (!hasSlotOperand(op, slot))
          return rewriter.notifyMatchFailure(op, "missing operand");
        args.push_back(operands[slot.operand]);
        break;
      case SHMEMArgKind::OutTeam:
      case SHMEMArgKind::OutInt: {
        Type type =
            static_cast<SHMEMArgKind>(slot.abiKind) == SHMEMArgKind::OutTeam
                ? team
                : i32;
        Value storage = createStackSlot(rewriter, loc, type);
        args.push_back(storage);
        if (producesSlotResult(op, slot))
          outParameters.push_back({slot.result, storage, type});
        break;
      }
      case SHMEMArgKind::NullPointer:
        args.push_back(LLVM::ZeroOp::create(rewriter, loc, ptr));
        break;
      case SHMEMArgKind::ZeroI32:
        args.push_back(createI32Constant(rewriter, loc, 0));
        break;
      case SHMEMArgKind::ZeroI64:
        args.push_back(createI64Constant(rewriter, loc, 0));
        break;
      }
    }

    // USIS describes the specification; the ABI may return a status where the
    // specification returns nothing (Sandia's shmem_team_destroy).
    std::optional<StringRef> returnOverride =
        target.returnTypeOverride(call->callee);
    FailureOr<Type> returnType = getReturnType(
        rewriter, returnOverride.value_or(StringRef(call->returnCType)));
    if (failed(returnType))
      return rewriter.notifyMatchFailure(op, "unknown C return type");

    FailureOr<LLVM::CallOp> emitted =
        emitRuntimeCall(rewriter, op, loc, call->callee, *returnType, args);
    if (failed(emitted))
      return LogicalResult::failure();
    Value returned =
        emitted->getNumResults() == 1 ? emitted->getResult() : Value();

    // The operation's results, in the order the CACTS entry lists them.
    SmallVector<std::pair<int, Value>> produced;
    for (const OutParameter &out : outParameters)
      produced.push_back(
          {out.result,
           LLVM::LoadOp::create(rewriter, loc, out.type, out.storage)});
    if (call->returnResult >= 0) {
      if (!returned)
        return rewriter.notifyMatchFailure(op, "the call returns nothing");
      produced.push_back({call->returnResult, returned});
    }
    if (call->returnOperand >= 0) {
      // The raiser gave the operation storage holding what the call returned,
      // so the lowering writes it back there.
      if (!returned ||
          call->returnOperand >= static_cast<int>(op->getNumOperands()))
        return rewriter.notifyMatchFailure(op, "no storage for the result");
      LLVM::StoreOp::create(rewriter, loc, returned,
                            operands[call->returnOperand]);
    }
    llvm::sort(produced, [](const std::pair<int, Value> &lhs,
                            const std::pair<int, Value> &rhs) {
      return lhs.first < rhs.first;
    });

    SmallVector<Value> results;
    for (const auto &[index, value] : produced)
      results.push_back(value);
    // A blocking call of an operation the optimisations gave a request to:
    // there is nothing to complete, and the quiet completes it anyway.
    if (needsRequestNull(getSHMEMTableSpec(target.flavour), op, *call))
      results.push_back(createI32Constant(rewriter, loc, kNoRequest));
    // Only the routines that return an int report anything; the rest cannot
    // fail, so their spmd.error is the model's success.
    bool returnsStatus =
        call->returnKind == "ERROR_CODE" || returnOverride == "int";
    results.push_back(returnsStatus && returned
                          ? returned
                          : createI32Constant(rewriter, loc, target.success));

    if (results.size() != op->getNumResults())
      return rewriter.notifyMatchFailure(op, "unexpected number of results");
    for (auto [value, type] : llvm::zip(results, op->getResultTypes()))
      if (value.getType() != getTypeConverter()->convertType(type))
        return rewriter.notifyMatchFailure(op, "result of the wrong type");
    rewriter.replaceOp(op, results);
    return success();
  }

private:
  const SHMEMTarget &target;
};

void populateSHMEMCallPatterns(const TypeConverter &c, RewritePatternSet &p,
                               const SHMEMTarget &target) {
  SmallVector<StringRef> lowered;
  for (const CallDescription &call : getSHMEMLoweringTable(target.flavour))
    if (call.canonical && !llvm::is_contained(lowered, call.spmdOpName))
      lowered.push_back(call.spmdOpName);
  for (StringRef spmdOpName : lowered)
    p.add<SHMEMTableCallPattern>(c, spmdOpName, p.getContext(), target);
}

//===----------------------------------------------------------------------===//
// Windows, which are the symmetric heap
//===----------------------------------------------------------------------===//

Value successCode(ConversionPatternRewriter &rewriter, Location loc,
                  const SHMEMTarget &target) {
  return createI32Constant(rewriter, loc, target.success);
}

// spmd.winCreate(comm, base, size, dispUnit) -> base.  The window is the
// symmetric address it is over; that the address is on the symmetric heap is
// what checkSymmetricWindow verifies, because the model cannot reach anything
// else.  The size and the displacement unit are the program's own bookkeeping
// and no call takes them.
struct LowerWinCreate : public OpConversionPattern<WinCreateOp> {
  LowerWinCreate(const TypeConverter &converter, MLIRContext *ctx,
                 const SHMEMTarget &target)
      : OpConversionPattern(converter, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(WinCreateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, {adaptor.getBaseAddress(),
                            successCode(rewriter, op.getLoc(), target)});
    return success();
  }

private:
  const SHMEMTarget &target;
};

// spmd.winAlloc(comm, size, dispUnit) -> <prefix>malloc(size), which is where
// a window has to live.  The base address and the window are the same value.
struct LowerWinAlloc : public OpConversionPattern<WinAllocOp> {
  LowerWinAlloc(const TypeConverter &converter, MLIRContext *ctx,
                const SHMEMTarget &target)
      : OpConversionPattern(converter, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(WinAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    FailureOr<LLVM::CallOp> call =
        emitRuntimeCall(rewriter, op, op.getLoc(), routine(target, "malloc"),
                        ptr, {adaptor.getSize()});
    if (failed(call))
      return failure();
    Value address = call->getResult();
    rewriter.replaceOp(
        op, {address, address, successCode(rewriter, op.getLoc(), target)});
    return success();
  }

private:
  const SHMEMTarget &target;
};

/// Where a window's memory comes from, which decides whether freeing the window
/// frees the memory.
enum class WindowOrigin {
  /// spmd.winAlloc: allocated on the symmetric heap for the window, so the
  /// window owns it.
  Allocated,
  /// spmd.winCreate: the program's own memory, which it frees itself.
  Created,
  /// Neither is visible, because the window reaches spmd.winFree through a
  /// variable, an argument or a block argument.
  Unknown,
};

WindowOrigin originOf(Value win) {
  while (auto cast = win.getDefiningOp<UnrealizedConversionCastOp>()) {
    if (cast->getNumOperands() != 1)
      return WindowOrigin::Unknown;
    win = cast->getOperand(0);
  }
  if (win.getDefiningOp<WinAllocOp>())
    return WindowOrigin::Allocated;
  if (win.getDefiningOp<WinCreateOp>())
    return WindowOrigin::Created;
  return WindowOrigin::Unknown;
}

// spmd.winFree(win) -> quiet(), and free(win) when the window owns its memory.
// Freeing a window completes the transfers still outstanding on it
// (MPI_Win_free), and neither model can complete those of one address only,
// so it completes all of them.  A window spmd.winCreate made over the
// program's own memory is that memory, which the program frees itself --
// MPI_Win_free does not free it, and freeing it here would free it twice.
//
// Where the window comes from is decided on the module before the conversion
// starts (\p ownsMemory), when the operations that made the windows are still
// there to look at; the check reports a free whose window is neither.
struct LowerWinFree : public OpConversionPattern<WinFreeOp> {
  LowerWinFree(const TypeConverter &converter, MLIRContext *ctx,
               const SHMEMTarget &target,
               const DenseSet<Operation *> &ownsMemory)
      : OpConversionPattern(converter, ctx), target(target),
        ownsMemory(ownsMemory) {}

  LogicalResult
  matchAndRewrite(WinFreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    if (failed(emitRuntimeCall(rewriter, op, op.getLoc(),
                               routine(target, kQuiet), voidType, {})))
      return failure();
    if (ownsMemory.contains(op) &&
        failed(emitRuntimeCall(rewriter, op, op.getLoc(),
                               routine(target, "free"), voidType,
                               {adaptor.getWin()})))
      return failure();
    rewriter.replaceOp(op, successCode(rewriter, op.getLoc(), target));
    return success();
  }

private:
  const SHMEMTarget &target;
  const DenseSet<Operation *> &ownsMemory;
};

//===----------------------------------------------------------------------===//
// Completion, which is process-wide
//===----------------------------------------------------------------------===//

/// One operation lowered to a call taking nothing and returning nothing, whose
/// single result is the error code.  The family's completion and ordering
/// primitives take no window and no request: they apply to everything this PE
/// has outstanding, which is stronger than what the operation asked for.
template <typename OpT>
struct LowerToProcessWideCall : public OpConversionPattern<OpT> {
  LowerToProcessWideCall(const TypeConverter &converter, MLIRContext *ctx,
                         const SHMEMTarget &target, StringRef name)
      : OpConversionPattern<OpT>(converter, ctx), target(target),
        callee(routine(target, name)) {}

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    if (failed(
            emitRuntimeCall(rewriter, op, op.getLoc(), callee, voidType, {})))
      return failure();
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected number of results");
    rewriter.replaceOp(op, successCode(rewriter, op.getLoc(), target));
    return success();
  }

private:
  const SHMEMTarget &target;
  std::string callee;
};

// spmd.reqFree -> nothing.  There is no request object to free; what the
// request stood for is completed by the quiet, which spmd.wait is.
struct LowerReqFree : public OpConversionPattern<ReqFreeOp> {
  LowerReqFree(const TypeConverter &converter, MLIRContext *ctx,
               const SHMEMTarget &target)
      : OpConversionPattern(converter, ctx), target(target) {}

  LogicalResult
  matchAndRewrite(ReqFreeOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, successCode(rewriter, op.getLoc(), target));
    return success();
  }

private:
  const SHMEMTarget &target;
};

//===----------------------------------------------------------------------===//
// The checks
//===----------------------------------------------------------------------===//

bool hasOperationTaggedFor(ModuleOp module, ProgrammingModel model) {
  return module
      .walk([model](Operation *op) {
        return isTaggedFor(op, model) ? WalkResult::interrupt()
                                      : WalkResult::advance();
      })
      .wasInterrupted();
}

/// The object a window's address is in, followed back through the casts and
/// views the raiser and the dialect wrap it in and the element addresses the
/// program takes into it (&buf[2]).
Value peelAddress(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(def)) {
      if (cast->getNumOperands() != 1)
        return value;
      value = cast->getOperand(0);
    } else if (auto view = dyn_cast<enzymexla::Pointer2MemrefOp>(def)) {
      value = view.getSource();
    } else if (auto element = dyn_cast<LLVM::GEPOp>(def)) {
      value = element.getBase();
    } else {
      return value;
    }
  }
  return value;
}

/// Where the window's base address provably is, when it is somewhere the model
/// cannot reach from another PE: the stack, and for NVSHMEM a global variable.
/// A remote PE can read and write the symmetric heap, and OpenSHMEM makes the
/// program's global and static variables symmetric too, so a window over
/// anything else is a transfer that goes nowhere.  Nothing, for an address the
/// model may reach.
///
/// Anything whose provenance the lowering cannot see is let through.  A
/// symmetric address reaches its window through a function argument or a
/// structure field often enough that rejecting what cannot be proved
/// symmetric would reject working programs; what this catches is the case
/// that is certainly wrong.
StringRef unreachableBase(Value base, const SHMEMTarget &target) {
  Operation *def = peelAddress(base).getDefiningOp();
  if (isa_and_present<LLVM::AllocaOp, memref::AllocaOp>(def))
    return "the stack";
  if (!target.globalsAreSymmetric &&
      isa_and_present<LLVM::AddressOfOp, memref::GetGlobalOp>(def))
    return "a global variable";
  return StringRef();
}

/// Every operation the model cannot express, with its own reason.
LogicalResult checkExpressible(Operation *op, const SHMEMTarget &target) {
  if (StringLiteral reason = noCounterpartReason(op); !reason.empty())
    return op->emitOpError() << forTarget(reason, target);

  // The lowering emits host calls.  A call a GPU kernel makes is part of the
  // kernel, which nothing here compiles.  The handles it uses are reported
  // with it, not each on its own.
  if (isInDeviceCode(op) && !isMemoryEffectFree(op))
    return op->emitOpError()
           << "is inside a GPU kernel; the lowering emits host-initiated "
           << target.displayName << " calls only, and compiles no device code";

  if (op->hasTrait<spmd::P2P>())
    return op->emitOpError()
           << "is two-sided messaging, which " << target.displayName
           << " does not have; its transfers name an address on the symmetric "
              "heap, not a peer that receives";

  // A collective the program started and completes later.  The family's
  // collectives are all blocking, and making this one blocking would move the
  // synchronisation to a point the program did not choose.
  if (op->hasTrait<spmd::Collective>())
    if (auto blocking = op->getAttrOfType<BoolAttr>("isBlocking");
        blocking && !blocking.getValue())
      return op->emitOpError() << "is a non-blocking collective, which "
                               << target.displayName << " does not have";

  // The quiet completes every outstanding operation of this PE and says
  // nothing about any single one of them.
  if (auto wait = dyn_cast<WaitOp>(op)) {
    if (wait.getStatus())
      return wait.emitOpError()
             << "asks for a status; " << routine(target, kQuiet)
             << "() reports nothing about the operation it completes";
  } else if (auto waitAll = dyn_cast<WaitAllOp>(op)) {
    if (waitAll.getStatuses())
      return waitAll.emitOpError()
             << "asks for statuses; " << routine(target, kQuiet)
             << "() reports nothing about the operations it completes";
  }

  if (auto winFree = dyn_cast<WinFreeOp>(op))
    if (originOf(winFree.getWin()) == WindowOrigin::Unknown)
      return winFree.emitOpError()
             << "frees a window made neither by spmd.winAlloc nor by "
                "spmd.winCreate in sight; whether "
             << routine(target, "free")
             << " has to free its memory depends on which, and guessing "
                "either way frees memory twice or leaks it";

  if (auto winCreate = dyn_cast<WinCreateOp>(op)) {
    StringRef where = unreachableBase(winCreate.getBaseAddress(), target);
    if (!where.empty()) {
      InFlightDiagnostic diag =
          winCreate.emitOpError()
          << "creates a window over " << where << ", which is not symmetric; "
          << target.displayName << " reaches only the symmetric heap";
      if (target.globalsAreSymmetric)
        diag << " and global variables";
      else
        diag << " " << routine(target, "malloc") << " allocates";
      diag << " from another PE";
      return diag;
    }
  }

  return success();
}

/// Whether \p type is a handle this ABI holds in a pointer, where MPICH, which
/// a translated program was written against, holds it in an int.
bool isPointerHandle(Type type, const SHMEMTarget &target) {
  return isa<WinType>(type) || (target.teamIsPointer && isa<CommType>(type));
}

/// A team or a window the program keeps in a variable of its own.  The raiser
/// leaves the store and the load of such a variable as casts between the
/// handle and the variable's type; a program written against MPICH declares
/// that variable an int, and a window -- and an OpenSHMEM team -- is a
/// pointer, which does not fit.  The casts would survive
/// -spmd-reconcile-unrealized-cast and fail the translation to LLVM IR with no
/// word about why, so the check says it here.
LogicalResult checkHandleWidth(UnrealizedConversionCastOp cast,
                               const SHMEMTarget &target) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return success();
  Type from = cast->getOperand(0).getType();
  Type to = cast->getResult(0).getType();

  Type handle, stored;
  bool touchesModel = false;
  if (isPointerHandle(from, target) && to.isIntOrIndex()) {
    handle = from;
    stored = to;
    if (Operation *def = cast->getOperand(0).getDefiningOp())
      touchesModel = isTaggedFor(def, target.model);
  } else if (from.isIntOrIndex() && isPointerHandle(to, target)) {
    handle = to;
    stored = from;
    touchesModel =
        llvm::any_of(cast->getResult(0).getUsers(), [&](Operation *user) {
          return isTaggedFor(user, target.model);
        });
  }
  // An index is as wide as a pointer, and has no bit width of its own.
  if (!touchesModel || isa<IndexType>(stored) ||
      stored.getIntOrFloatBitWidth() >= 64)
    return success();
  return cast.emitError()
         << "the program keeps an " << target.displayName << " "
         << (isa<WinType>(handle) ? "window" : "team") << " in " << stored
         << ", but " << (isa<WinType>(handle) ? "a window" : "a team")
         << " is a pointer in this ABI; a variable declared for an MPI "
            "handle is too narrow for it";
}

/// An operand the dialect has and no call of the model has a parameter for,
/// and one whose value a call fixes -- a stride, a displacement, a datatype, a
/// reduction operation.  All of them are what the CACTS entry says read
/// backwards, so the table finds them.
LogicalResult checkCallable(Operation *op, const SHMEMTarget &target) {
  OpBuilder builder(op->getContext());
  SmallVector<std::string> reasons;
  selectSHMEMCall(target.flavour, op, builder, reasons);
  for (const std::string &reason : reasons)
    op->emitOpError() << reason;
  return failure(!reasons.empty());
}

} // namespace

LogicalResult checkSPMDToSHMEMFamily(ModuleOp module,
                                     const SHMEMTarget &target) {
  if (failed(checkUsedModels(module, target.passName)))
    return failure();
  // -lower-spmd runs this for every model of its table, whether or not the
  // module holds any operation of that model.  Nothing tagged for it means
  // nothing to check, and no reason to read the API description.
  if (!hasOperationTaggedFor(module, target.model))
    return success();
  // The calls are derived from the same API description the raiser reads, so
  // the lowering needs it too.
  if (getSHMEMLoweringTable(target.flavour).empty())
    // Not module.emitError(): the module is not what is wrong with the run,
    // and printing it back as the offending operation buries the one line
    // that is.
    return emitError(module.getLoc())
           << target.passName << ": no " << target.displayName
           << " API description; set " << target.jsonVariable << " to "
           << target.jsonFile;

  bool inexpressible = false;
  module.walk([&](Operation *op) {
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op)) {
      if (failed(checkHandleWidth(cast, target)))
        inexpressible = true;
      return;
    }
    if (!isTaggedFor(op, target.model))
      return;
    // All three run, so an operation wrong on more than one count reports
    // every reason at once.
    bool handle = succeeded(checkPredefinedHandle(op, target));
    bool expressible = succeeded(checkExpressible(op, target));
    bool callable = succeeded(checkCallable(op, target));
    if (!handle || !expressible || !callable)
      inexpressible = true;
  });
  return failure(inexpressible);
}

LogicalResult lowerSPMDToSHMEMFamily(ModuleOp module, const SHMEMTarget &target,
                                     StructureLowering structure) {
  MLIRContext *ctx = module.getContext();

  if (failed(checkSPMDToSHMEMFamily(module, target)))
    return failure();
  // Nothing to convert: no pattern would apply, and building them would read
  // the API description.
  if (!hasOperationTaggedFor(module, target.model))
    return success();

  DenseSet<Operation *> ownsMemory;
  module.walk([&](WinFreeOp winFree) {
    if (isTaggedFor(winFree, target.model) &&
        originOf(winFree.getWin()) == WindowOrigin::Allocated)
      ownsMemory.insert(winFree);
  });

  SHMEMTypeConverter converter(ctx, target);
  RewritePatternSet patterns(ctx);
  patterns.add<LowerWinFree>(converter, ctx, target, ownsMemory);
  patterns.add<LowerPredefinedTeam<CommWorldOp>>(converter, ctx, target,
                                                 /*world=*/true);
  patterns.add<LowerPredefinedTeam<CommNodeOp>>(converter, ctx, target,
                                                /*world=*/false);
  patterns.add<LowerConstantHandle<RequestNullOp, kNoRequest>,
               LowerConstantHandle<ConstDatatypeOp, kUnusedHandle>,
               LowerConstantHandle<ConstOpTypeOp, kUnusedHandle>>(converter,
                                                                  ctx);
  patterns.add<LowerNamedHandle<ConstThreadLevelOp>,
               LowerNamedHandle<ConstSignalOpTypeOp>, LowerWinCreate,
               LowerWinAlloc, LowerReqFree>(converter, ctx, target);
  patterns.add<LowerToProcessWideCall<FenceOp>>(converter, ctx, target,
                                                kBarrierAll);
  patterns.add<LowerToProcessWideCall<FenceOrderingOp>>(converter, ctx, target,
                                                        kFence);
  patterns
      .add<LowerToProcessWideCall<FlushOp>, LowerToProcessWideCall<FlushAllOp>,
           LowerToProcessWideCall<WaitOp>, LowerToProcessWideCall<WaitAllOp>>(
          converter, ctx, target, kQuiet);
  populateSHMEMCallPatterns(converter, patterns, target);
  populateSPMDStructureLoweringPatterns(converter, patterns, target.model,
                                        structure);

  return applySPMDLowering(module, target.model, converter,
                           std::move(patterns));
}

} // namespace mlir::spmd
