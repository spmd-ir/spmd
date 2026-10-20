// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDToNCCL.cpp - Lower SPMD operations to NCCL calls ----------===//
//
// Every operation tagged for NCCL becomes a call to the NCCL routine it
// models, under the ABI of includesProgramModels/nccl/include/nccl.h: a
// communicator and a CUDA stream are pointers, a datatype and a reduction
// operation are enumerators (include/spmd/TargetABI/NCCL.h), and every routine
// returns an ncclResult_t.
//
// The calls themselves are derived: NCCLLoweringTable reads them from the CACTS
// map the raiser reads and the USIS API description.  This file keeps what
// neither source says, which for NCCL is two things.
//
// * NCCL has no world communicator.  The raiser makes ncclCommInitRank an
//   spmd.init followed by an spmd.commWorld and drops the size, the unique id
//   and the rank, so the lowering has to start the communicator the way every
//   NCCL program does: ncclGetUniqueId on rank 0, the id sent to the other
//   ranks over a channel NCCL does not have, and ncclCommInitRank with the real
//   rank and size.  The channel is the process model the program runs under --
//   MPI, or OpenSHMEM when there is no MPI -- whose world is the one the raiser
//   checked NCCL's size against before it raised the call at all.  spmd.init
//   calls one bootstrap function the lowering adds to the module, which leaves
//   the communicator in a global, and spmd.commWorld loads it.
//
// * NCCL has no requests.  The raiser turns a group -- ncclGroupStart, some
//   calls, ncclGroupEnd -- into operations with a request each and an
//   spmd.waitAll of all of them where ncclGroupEnd was.  The lowering turns
//   that back: every operation with a request opens a group of its own, and
//   every request a wait completes closes one.  NCCL counts nested groups and
//   runs them all at the outermost ncclGroupEnd, which is where the wait is.
//   That is only the program's meaning when the wait completes every operation
//   still open, so a wait that completes some of them, or a group that is not
//   completed in its block, is reported.
//
// A CUDA stream is the program's own: spmd.castStream lowers to the
// cudaStream_t it casts.
//
//===----------------------------------------------------------------------===//

#include "LowerSPMDToNCCL.h"
#include "LowerSPMDCommon.h"
#include "NCCLLoweringTable.h"

#include "spmd/SPMDPasses.h"
#include "spmd/SPMDTraits.h"
#include "spmd/TargetABI/MPICH.h"
#include "spmd/TargetABI/NCCL.h"
#include "spmd/TargetABI/OpenSHMEM.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Transforms/DialectConversion.h"
#include "src/enzyme_ad/jax/Dialect/Dialect.h"

#include "llvm/ADT/STLExtras.h"

#include <algorithm>
#include <optional>

namespace mlir::spmd {

#define GEN_PASS_DEF_LOWERSPMDTONCCL
#include "spmd/SPMDPasses.h.inc"

namespace {

namespace nccl = abi::nccl;

/// The value !spmd.req lowers to.  NCCL hands out no request: an operation
/// with a request is one of a group, which the ncclGroupEnd its wait lowers to
/// runs, so one value stands for all of them and nothing reads it.
constexpr int32_t kNoRequest = 0;

/// What the lowering adds to the module: the world communicator, the storage
/// of the unique id it is made from (and, over OpenSHMEM, of the copy the other
/// processes receive), and the function that makes it.
constexpr StringLiteral kCommWorldGlobal = "__spmd_nccl_comm_world";
constexpr StringLiteral kUniqueIdGlobal = "__spmd_nccl_unique_id";
constexpr StringLiteral kReceivedUniqueIdGlobal =
    "__spmd_nccl_unique_id_received";
constexpr StringLiteral kBootstrapFunction = "__spmd_nccl_bootstrap";

constexpr StringLiteral kGroupStart = "ncclGroupStart";
constexpr StringLiteral kGroupEnd = "ncclGroupEnd";

//===----------------------------------------------------------------------===//
// Types
//===----------------------------------------------------------------------===//

// A communicator and a stream are pointers in this ABI, and a datatype, a
// reduction operation and a status are C enums, which are ints.
class NCCLTypeConverter : public SPMDLoweringTypeConverter {
public:
  explicit NCCLTypeConverter(MLIRContext *ctx)
      : SPMDLoweringTypeConverter(ctx) {
    auto i32 = IntegerType::get(ctx, 32);
    auto ptr = LLVM::LLVMPointerType::get(ctx);
    addConversion([ptr](CommType) -> Type { return ptr; });
    addConversion([ptr](StreamType) -> Type { return ptr; });
    addConversion([i32](ErrorType) -> Type { return i32; });
    addConversion([i32](ReqType) -> Type { return i32; });
    addConversion([i32](DatatypeType) -> Type { return i32; });
    addConversion([i32](OpType) -> Type { return i32; });
    addConversion([i32](ThreadLevelType) -> Type { return i32; });
    addConversion([ptr](MemRefType) -> Type { return ptr; });
  }
};

LLVM::LLVMStructType getUniqueIdType(MLIRContext *ctx) {
  return LLVM::LLVMStructType::getLiteral(
      ctx, {LLVM::LLVMArrayType::get(IntegerType::get(ctx, 8),
                                     nccl::kUniqueIdBytes)});
}

//===----------------------------------------------------------------------===//
// What NCCL has no counterpart for
//===----------------------------------------------------------------------===//

struct NoCounterpart {
  StringLiteral spmdOpName;
  StringLiteral reason;
};

// Operations NCCL cannot express at all, reported before anything is
// rewritten so that each arrives with its own reason.  Any other operation of
// the dialect that neither the table nor a pattern here lowers is reported as
// having no counterpart, without one.
constexpr NoCounterpart kNoCounterpart[] = {
    {"spmd.finalize",
     "has no NCCL counterpart; NCCL keeps no process-wide state, and each "
     "communicator is destroyed by ncclCommDestroy, which spmd.commDestroy is"},
    {"spmd.abort", "ends the program; ncclCommAbort only abandons the "
                   "operations of one communicator"},
    {"spmd.barrier", "has no NCCL counterpart; NCCL has no barrier, and an "
                     "empty group synchronises nothing"},
    {"spmd.commDup",
     "has no NCCL counterpart; NCCL duplicates a communicator by splitting it "
     "with one colour, which spmd.commSplit is"},
    {"spmd.reqFree",
     "lets an operation complete on its own; an NCCL group runs only at the "
     "ncclGroupEnd that closes it, which spmd.wait is"},
    {"spmd.waitSome", "completes some of several operations; ncclGroupEnd "
                      "runs every operation of its group"},
    {"spmd.waitAny", "completes one of several operations; ncclGroupEnd runs "
                     "every operation of its group"},
    {"spmd.test", "asks whether an operation has completed; an NCCL group "
                  "runs at its ncclGroupEnd and cannot be asked"},
    {"spmd.testAll", "asks whether operations have completed; an NCCL group "
                     "runs at its ncclGroupEnd and cannot be asked"},
    {"spmd.testSome", "asks whether operations have completed; an NCCL group "
                      "runs at its ncclGroupEnd and cannot be asked"},
    {"spmd.testAny", "asks whether operations have completed; an NCCL group "
                     "runs at its ncclGroupEnd and cannot be asked"},
};

StringRef noCounterpartReason(Operation *op) {
  for (const NoCounterpart &entry : kNoCounterpart)
    if (entry.spmdOpName == op->getName().getStringRef())
      return entry.reason;
  return StringRef();
}

/// The barrier the raiser puts in front of the yield of a process-level
/// spmd.parallel.  It matches only itself and goes with the parallel, so it is
/// not an NCCL barrier to report.
bool isProcessTeamBarrier(Operation *op) {
  auto barrier = dyn_cast<BarrierOp>(op);
  if (!barrier || !barrier.getIsUniqueMatching().value_or(false))
    return false;
  auto parallel = dyn_cast_or_null<ParallelOp>(barrier->getParentOp());
  return parallel && parallel.getIsProcessLevel() &&
         isa_and_present<ParallelYieldOp>(barrier->getNextNode()) &&
         getUsedModel(parallel) == getUsedModel(barrier);
}

/// The operations this file lowers itself rather than through the table.
bool isLoweredByHand(Operation *op) {
  if (auto parallel = dyn_cast<ParallelOp>(op))
    return parallel.getIsProcessLevel();
  return isa<CommWorldOp, ConstDatatypeOp, ConstOpTypeOp, ConstThreadLevelOp,
             CastStreamOp, RequestNullOp, InitOp, WaitOp, WaitAllOp>(op) ||
         isProcessTeamBarrier(op);
}

//===----------------------------------------------------------------------===//
// The bootstrap
//===----------------------------------------------------------------------===//

/// The model whose processes NCCL's world communicator is made over, and over
/// whose collectives rank 0 sends the unique id to the others.
enum class BootstrapModel { MPI, SHMEM };

/// Whether \p module starts \p model: an spmd.init tagged for it or, once that
/// model's own lowering has run, a call to one of \p initFunctions.  A
/// declaration proves nothing, because the raiser leaves the program's
/// declarations behind whether or not their calls survive: a program
/// translated from MPI to OpenSHMEM still declares MPI_Init.
bool startsModel(ModuleOp module, ProgrammingModel model,
                 ArrayRef<StringRef> initFunctions) {
  return module
      .walk([&](Operation *op) {
        if (isa<InitOp>(op) && isTaggedFor(op, model))
          return WalkResult::interrupt();
        if (auto call = dyn_cast<LLVM::CallOp>(op))
          if (std::optional<StringRef> callee = call.getCallee();
              callee && llvm::is_contained(initFunctions, *callee))
            return WalkResult::interrupt();
        return WalkResult::advance();
      })
      .wasInterrupted();
}

/// The process model of \p module: MPI if the module starts it, OpenSHMEM
/// otherwise.  A driver lowering several models lowers MPI before NCCL, which
/// is why a call to the initialisation counts as much as the operation.
std::optional<BootstrapModel> findBootstrapModel(ModuleOp module) {
  if (startsModel(module, ProgrammingModel::MPI,
                  {"MPI_Init", "MPI_Init_thread"}))
    return BootstrapModel::MPI;
  if (startsModel(module, ProgrammingModel::SHMEM,
                  {"shmem_init", "shmem_init_thread"}))
    return BootstrapModel::SHMEM;
  return std::nullopt;
}

/// An internal, zero-initialised global the lowering keeps its state in.  A
/// symbol of that name that is something else is reported on \p anchor.
FailureOr<LLVM::GlobalOp> getOrCreateGlobal(OpBuilder &builder,
                                            Operation *anchor, StringRef name,
                                            Type type) {
  auto module = anchor->getParentOfType<ModuleOp>();
  if (Operation *symbol = module.lookupSymbol(name)) {
    auto global = dyn_cast<LLVM::GlobalOp>(symbol);
    if (!global || global.getGlobalType() != type)
      return anchor->emitOpError()
             << "needs @" << name << " to be a global of type " << type
             << ", but the module defines it otherwise";
    return global;
  }
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(module.getBody());
  Location loc = module.getLoc();
  auto global =
      LLVM::GlobalOp::create(builder, loc, type, /*isConstant=*/false,
                             LLVM::Linkage::Internal, name, Attribute());
  Block *init = builder.createBlock(&global.getInitializerRegion());
  builder.setInsertionPointToStart(init);
  LLVM::ReturnOp::create(builder, loc,
                         LLVM::ZeroOp::create(builder, loc, type));
  return global;
}

/// Emits a call to \p callee, which the module gets a declaration of if it has
/// none.
/// The rank and the size of the process in the world of \p model, and a
/// broadcast of the unique id from rank 0, as the program would write them.
struct ProcessChannel {
  Value rank;
  Value size;
};

FailureOr<ProcessChannel> emitRankAndSize(OpBuilder &builder, Operation *anchor,
                                          Location loc, BootstrapModel model) {
  Type i32 = builder.getI32Type();
  auto ptr = LLVM::LLVMPointerType::get(builder.getContext());
  if (model == BootstrapModel::SHMEM) {
    FailureOr<LLVM::CallOp> rank =
        emitRuntimeCall(builder, anchor, loc, "shmem_my_pe", i32, {});
    FailureOr<LLVM::CallOp> size =
        emitRuntimeCall(builder, anchor, loc, "shmem_n_pes", i32, {});
    if (failed(rank) || failed(size))
      return failure();
    return ProcessChannel{rank->getResult(), size->getResult()};
  }

  Value one = createI32Constant(builder, loc, 1);
  Value rankSlot = LLVM::AllocaOp::create(builder, loc, ptr, i32, one);
  Value sizeSlot = LLVM::AllocaOp::create(builder, loc, ptr, i32, one);
  Value world = createI32Constant(builder, loc, abi::mpich::kCommWorld);
  if (failed(emitRuntimeCall(builder, anchor, loc, "MPI_Comm_rank", i32,
                             {world, rankSlot})) ||
      failed(emitRuntimeCall(builder, anchor, loc, "MPI_Comm_size", i32,
                             {world, sizeSlot})))
    return failure();
  return ProcessChannel{LLVM::LoadOp::create(builder, loc, i32, rankSlot),
                        LLVM::LoadOp::create(builder, loc, i32, sizeSlot)};
}

std::optional<int32_t> mpiByte() {
  for (const abi::mpich::DatatypeHandle &entry : abi::mpich::kDatatypes)
    if (entry.name == "MPI_BYTE")
      return entry.handle;
  return std::nullopt;
}

/// Sends the unique id at \p id from rank 0 to every process, and returns the
/// address of the id this process passes to ncclCommInitRank.
FailureOr<Value> emitIdBroadcast(OpBuilder &builder, Operation *anchor,
                                 Location loc, BootstrapModel model, Value id,
                                 Value isRoot) {
  Type i32 = builder.getI32Type();
  auto ptr = LLVM::LLVMPointerType::get(builder.getContext());
  Value root = createI32Constant(builder, loc, 0);
  if (model == BootstrapModel::SHMEM) {
    // SHMEM_TEAM_WORLD is an extern variable holding the team, and the ids are
    // globals, which OpenSHMEM makes symmetric.  Whether shmem_broadcastmem
    // may take one buffer as both source and destination, and whether it
    // writes the root's destination at all, neither the vendored header nor
    // USIS says; with a buffer of each, and the root keeping the id it made,
    // the answer does not matter.
    auto module = anchor->getParentOfType<ModuleOp>();
    auto team = module.lookupSymbol<LLVM::GlobalOp>(abi::openshmem::kTeamWorld);
    if (!team) {
      OpBuilder::InsertionGuard guard(builder);
      builder.setInsertionPointToStart(module.getBody());
      team = LLVM::GlobalOp::create(builder, loc, ptr, /*isConstant=*/false,
                                    LLVM::Linkage::External,
                                    abi::openshmem::kTeamWorld, Attribute());
    }
    FailureOr<LLVM::GlobalOp> receivedGlobal =
        getOrCreateGlobal(builder, anchor, kReceivedUniqueIdGlobal,
                          getUniqueIdType(builder.getContext()));
    if (failed(receivedGlobal))
      return failure();
    Value received = LLVM::AddressOfOp::create(builder, loc, *receivedGlobal);
    Value teamAddress = LLVM::AddressOfOp::create(builder, loc, team);
    Value world = LLVM::LoadOp::create(builder, loc, ptr, teamAddress);
    Value bytes = createI64Constant(builder, loc, nccl::kUniqueIdBytes);
    if (failed(emitRuntimeCall(builder, anchor, loc, "shmem_broadcastmem", i32,
                               {world, received, id, bytes, root})))
      return failure();
    return Value(LLVM::SelectOp::create(builder, loc, isRoot, id, received));
  }

  // MPI_Bcast has one buffer, which it writes on every process but the root.
  Value world = createI32Constant(builder, loc, abi::mpich::kCommWorld);
  Value bytes = createI32Constant(builder, loc, nccl::kUniqueIdBytes);
  Value byte = createI32Constant(builder, loc, *mpiByte());
  if (failed(emitRuntimeCall(builder, anchor, loc, "MPI_Bcast", i32,
                             {id, bytes, byte, root, world})))
    return failure();
  return id;
}

/// Adds the function spmd.init lowers a call to:
///
///   rank, size = the process's rank and size in the world of \p model
///   if (rank == 0) ncclGetUniqueId(&id)
///   broadcast id from rank 0
///   return ncclCommInitRank(&commWorld, size, id, rank)
///
/// The id and the communicator are globals, so that spmd.commWorld can read
/// the communicator anywhere after it and OpenSHMEM can broadcast the id.  The
/// result of ncclGetUniqueId is not checked: rank 0 cannot stop the others,
/// which are already waiting for the id, and the ncclCommInitRank of a rank
/// with a bad id reports the failure.
FailureOr<LLVM::LLVMFuncOp>
getOrCreateBootstrap(ModuleOp module, Operation *anchor, BootstrapModel model) {
  MLIRContext *ctx = module.getContext();
  OpBuilder builder(ctx);
  if (Operation *symbol = module.lookupSymbol(kBootstrapFunction)) {
    auto existing = dyn_cast<LLVM::LLVMFuncOp>(symbol);
    if (!existing)
      return anchor->emitOpError() << "needs @" << kBootstrapFunction
                                   << " to be the NCCL bootstrap, but the "
                                      "module defines it otherwise";
    return existing;
  }

  Type i32 = builder.getI32Type();
  auto ptr = LLVM::LLVMPointerType::get(ctx);
  LLVM::LLVMStructType idType = getUniqueIdType(ctx);
  FailureOr<LLVM::GlobalOp> commGlobal =
      getOrCreateGlobal(builder, anchor, kCommWorldGlobal, ptr);
  FailureOr<LLVM::GlobalOp> idGlobal =
      getOrCreateGlobal(builder, anchor, kUniqueIdGlobal, idType);
  if (failed(commGlobal) || failed(idGlobal))
    return failure();

  Location loc = module.getLoc();
  builder.setInsertionPointToEnd(module.getBody());
  auto bootstrap = LLVM::LLVMFuncOp::create(
      builder, loc, kBootstrapFunction, LLVM::LLVMFunctionType::get(i32, {}),
      LLVM::Linkage::Internal);
  Block *entry = bootstrap.addEntryBlock(builder);
  Block *root = builder.createBlock(&bootstrap.getBody());
  Block *share = builder.createBlock(&bootstrap.getBody());

  builder.setInsertionPointToStart(entry);
  FailureOr<ProcessChannel> channel =
      emitRankAndSize(builder, anchor, loc, model);
  if (failed(channel))
    return failure();
  Value zero = createI32Constant(builder, loc, 0);
  Value isRoot = LLVM::ICmpOp::create(builder, loc, LLVM::ICmpPredicate::eq,
                                      channel->rank, zero);
  LLVM::CondBrOp::create(builder, loc, isRoot, root, share);

  builder.setInsertionPointToStart(root);
  Value id = LLVM::AddressOfOp::create(builder, loc, *idGlobal);
  if (failed(
          emitRuntimeCall(builder, anchor, loc, "ncclGetUniqueId", i32, {id})))
    return failure();
  LLVM::BrOp::create(builder, loc, ValueRange(), share);

  builder.setInsertionPointToStart(share);
  id = LLVM::AddressOfOp::create(builder, loc, *idGlobal);
  FailureOr<Value> sharedId =
      emitIdBroadcast(builder, anchor, loc, model, id, isRoot);
  if (failed(sharedId))
    return failure();
  Value comm = LLVM::AddressOfOp::create(builder, loc, *commGlobal);
  FailureOr<LLVM::CallOp> init =
      emitRuntimeCall(builder, anchor, loc, "ncclCommInitRank", i32,
                      {comm, channel->size, *sharedId, channel->rank});
  if (failed(init))
    return failure();

  // ncclCommInitRank takes the 128-byte ncclUniqueId by value, which the
  // x86-64 C ABI passes in memory: a pointer to a copy, marked byval, as a C
  // compiler emits it.  A declaration the program already has carries the
  // mark; one the lowering adds gets it here.
  auto byval = NamedAttribute(
      StringAttr::get(ctx, LLVM::LLVMDialect::getByValAttrName()),
      TypeAttr::get(idType));
  auto initFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("ncclCommInitRank");
  if (!initFunc.getArgAttr(2, LLVM::LLVMDialect::getByValAttrName()))
    initFunc.setArgAttr(2, byval.getName(), byval.getValue());
  SmallVector<Attribute> argAttrs(4, DictionaryAttr::get(ctx));
  argAttrs[2] = DictionaryAttr::get(ctx, {byval});
  init->setArgAttrsAttr(ArrayAttr::get(ctx, argAttrs));
  LLVM::ReturnOp::create(builder, loc, init->getResult());
  return bootstrap;
}

//===----------------------------------------------------------------------===//
// Handles
//===----------------------------------------------------------------------===//

Value loadCommWorld(OpBuilder &builder, Operation *anchor) {
  auto module = anchor->getParentOfType<ModuleOp>();
  auto global = module.lookupSymbol<LLVM::GlobalOp>(kCommWorldGlobal);
  Location loc = anchor->getLoc();
  Value address = LLVM::AddressOfOp::create(builder, loc, global);
  return LLVM::LoadOp::create(
      builder, loc, LLVM::LLVMPointerType::get(builder.getContext()), address);
}

// spmd.commWorld -> the communicator the bootstrap left in its global.  The
// global exists by the time the conversion runs (lowerSPMDToNCCL), and the
// check has made sure an spmd.init ran before every load that matters.
struct LowerCommWorld : public OpConversionPattern<CommWorldOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CommWorldOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, loadCommWorld(rewriter, op));
    return success();
  }
};

std::optional<int32_t> handleOf(ConstDatatypeOp op) {
  return nccl::getDatatypeHandle(op.getTypeAttr());
}
std::optional<int32_t> handleOf(ConstOpTypeOp op) {
  return nccl::lookupHandle(nccl::kReduceOps, op.getStringAttr());
}
// NCCL has no thread support levels.  The raiser gives the spmd.init it makes
// the level SINGLE, and NCCL has no parameter to pass it to, so any level
// lowers to the same unused constant; checkPredefinedHandle makes sure nothing
// but that spmd.init reads it.
std::optional<int32_t> handleOf(ConstThreadLevelOp) { return 0; }

template <typename OpT>
struct LowerEnumerator : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<int32_t> handle = handleOf(op);
    if (!handle)
      return rewriter.notifyMatchFailure(op, "no NCCL enumerator");
    rewriter.replaceOp(op, createI32Constant(rewriter, op.getLoc(), *handle));
    return success();
  }
};

struct LowerRequestNull : public OpConversionPattern<RequestNullOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(RequestNullOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op,
                       createI32Constant(rewriter, op.getLoc(), kNoRequest));
    return success();
  }
};

// spmd.castStream -> the cudaStream_t it casts, which the raiser takes from the
// program's own call.
struct LowerCastStream : public OpConversionPattern<CastStreamOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(CastStreamOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value stream = adaptor.getOldStream();
    if (!isa<LLVM::LLVMPointerType>(stream.getType()))
      return rewriter.notifyMatchFailure(op, "the stream is not a pointer");
    rewriter.replaceOp(op, stream);
    return success();
  }
};

LogicalResult checkPredefinedHandle(Operation *op) {
  if (auto datatype = dyn_cast<ConstDatatypeOp>(op)) {
    if (!handleOf(datatype))
      return datatype.emitOpError()
             << "has no NCCL datatype for " << datatype.getTypeAttr();
  } else if (auto reduceOp = dyn_cast<ConstOpTypeOp>(op)) {
    if (!handleOf(reduceOp))
      return reduceOp.emitOpError()
             << "'" << reduceOp.getStringAttr()
             << "' is not a built-in NCCL reduction operation";
  } else if (auto level = dyn_cast<ConstThreadLevelOp>(op)) {
    for (Operation *user : level->getUsers())
      if (!isa<InitOp>(user) || !isTaggedFor(user, ProgrammingModel::NCCL))
        return level.emitOpError()
               << "is read by '" << user->getName()
               << "'; NCCL has no thread support levels, so the only place "
                  "it can lower to is the spmd.init NCCL ignores it in";
  } else if (auto cast = dyn_cast<CastStreamOp>(op)) {
    if (!isa<LLVM::LLVMPointerType>(cast.getOldStream().getType()))
      return cast.emitOpError()
             << "casts " << cast.getOldStream().getType()
             << " to a stream; NCCL takes the cudaStream_t itself, which is a "
                "pointer";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// Initialisation
//===----------------------------------------------------------------------===//

// spmd.init -> a call to the bootstrap, whose ncclCommInitRank result is the
// operation's error code.
struct LowerInit : public OpConversionPattern<InitOp> {
  LowerInit(const TypeConverter &converter, MLIRContext *ctx,
            LLVM::LLVMFuncOp bootstrap)
      : OpConversionPattern(converter, ctx), bootstrap(bootstrap) {}

  LogicalResult
  matchAndRewrite(InitOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!bootstrap || op.getThreadProvided())
      return rewriter.notifyMatchFailure(op, "no bootstrap for this init");
    auto call =
        LLVM::CallOp::create(rewriter, op.getLoc(), bootstrap, ValueRange());
    rewriter.replaceOp(op, call.getResult());
    return success();
  }

private:
  LLVM::LLVMFuncOp bootstrap;
};

//===----------------------------------------------------------------------===//
// Calls
//===----------------------------------------------------------------------===//

/// One operation lowered to the NCCL call NCCLLoweringTable picks for it: the
/// slots of that call say, argument by argument, what to pass.  An operation
/// with a request is one of a group, so its call is preceded by an
/// ncclGroupStart, which the wait that completes the request closes.
struct NCCLTableCallPattern : public ConversionPattern {
  NCCLTableCallPattern(const TypeConverter &converter, StringRef rootName,
                       MLIRContext *ctx)
      : ConversionPattern(converter, rootName, /*benefit=*/1, ctx) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<std::string> reasons;
    const CallDescription *call = selectNCCLCall(op, rewriter, reasons);
    if (!call)
      return rewriter.notifyMatchFailure(
          op, reasons.empty() ? "no NCCL call of this shape" : reasons.front());

    Location loc = op->getLoc();
    Type i32 = rewriter.getI32Type();
    auto ptr = LLVM::LLVMPointerType::get(rewriter.getContext());
    bool grouped = needsRequestNull(getNCCLTableSpec(), op, *call);

    // An out-parameter's stack slot, and which of the operation's results it
    // becomes.
    struct OutParameter {
      int result;
      Value storage;
      Type type;
    };
    SmallVector<Value> args;
    SmallVector<OutParameter> outParameters;

    for (const ArgSlot &slot : call->args) {
      switch (static_cast<NCCLArgKind>(slot.abiKind)) {
      case NCCLArgKind::In:
        if (!hasSlotOperand(op, slot))
          return rewriter.notifyMatchFailure(op, "missing operand");
        args.push_back(operands[slot.operand]);
        break;
      case NCCLArgKind::OutComm:
      case NCCLArgKind::OutPtr:
      case NCCLArgKind::OutInt: {
        Type type =
            static_cast<NCCLArgKind>(slot.abiKind) == NCCLArgKind::OutInt
                ? i32
                : Type(ptr);
        Value storage = createStackSlot(rewriter, loc, type);
        args.push_back(storage);
        if (producesSlotResult(op, slot))
          outParameters.push_back({slot.result, storage, type});
        break;
      }
      case NCCLArgKind::NullPointer:
        args.push_back(LLVM::ZeroOp::create(rewriter, loc, ptr));
        break;
      }
    }
    if (call->returnKind != "ERROR_CODE")
      return rewriter.notifyMatchFailure(op, "the call returns no status");

    // Its result is not read: an ncclGroupStart fails only on misuse the
    // ncclGroupEnd reports as well.
    if (grouped && failed(emitRuntimeCall(rewriter, op, loc, kGroupStart, i32,
                                          ValueRange())))
      return failure();
    FailureOr<LLVM::CallOp> emitted =
        emitRuntimeCall(rewriter, op, loc, call->callee, i32, args);
    if (failed(emitted))
      return failure();

    SmallVector<std::pair<int, Value>> produced;
    for (const OutParameter &out : outParameters)
      produced.push_back(
          {out.result,
           LLVM::LoadOp::create(rewriter, loc, out.type, out.storage)});
    llvm::sort(produced, [](const std::pair<int, Value> &lhs,
                            const std::pair<int, Value> &rhs) {
      return lhs.first < rhs.first;
    });

    SmallVector<Value> results;
    for (const auto &[index, value] : produced)
      results.push_back(value);
    if (grouped)
      results.push_back(createI32Constant(rewriter, loc, kNoRequest));
    results.push_back(emitted->getResult());

    if (results.size() != op->getNumResults())
      return rewriter.notifyMatchFailure(op, "unexpected number of results");
    for (auto [value, type] : llvm::zip(results, op->getResultTypes()))
      if (value.getType() != getTypeConverter()->convertType(type))
        return rewriter.notifyMatchFailure(op, "result of the wrong type");
    rewriter.replaceOp(op, results);
    return success();
  }
};

void populateNCCLCallPatterns(const TypeConverter &c, RewritePatternSet &p) {
  SmallVector<StringRef> lowered;
  for (const CallDescription &call : getNCCLLoweringTable())
    if (call.canonical && !llvm::is_contained(lowered, call.spmdOpName))
      lowered.push_back(call.spmdOpName);
  for (StringRef spmdOpName : lowered)
    p.add<NCCLTableCallPattern>(c, spmdOpName, p.getContext());
}

//===----------------------------------------------------------------------===//
// Completion, which closes groups
//===----------------------------------------------------------------------===//

/// How many requests \p op completes, when that is a constant.
std::optional<uint64_t> completedRequests(Operation *op) {
  if (isa<WaitOp>(op))
    return 1;
  APInt count;
  if (auto waitAll = dyn_cast<WaitAllOp>(op))
    if (matchPattern(waitAll.getCount(), m_ConstantInt(&count)) &&
        !count.isNegative())
      return count.getZExtValue();
  return std::nullopt;
}

// spmd.wait and spmd.waitAll -> one ncclGroupEnd per request.  The operations
// of the requests each opened a group, and NCCL runs them all at the
// outermost ncclGroupEnd, which is the last of these; its status is the
// operation's.
template <typename OpT>
struct LowerWaitToGroupEnd : public OpConversionPattern<OpT> {
  using OpConversionPattern<OpT>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(OpT op, typename OpT::Adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<uint64_t> count = completedRequests(op);
    if (!count || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "not a wait the check accepts");
    // A wait of no requests completes nothing, as MPI_Waitall(0, ...) does,
    // and closes no group.
    Value status = createI32Constant(rewriter, op.getLoc(), nccl::kSuccess);
    for (uint64_t i = 0; i < *count; ++i) {
      FailureOr<LLVM::CallOp> end =
          emitRuntimeCall(rewriter, op, op.getLoc(), kGroupEnd,
                          rewriter.getI32Type(), ValueRange());
      if (failed(end))
        return failure();
      status = end->getResult();
    }
    rewriter.replaceOp(op, status);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// The checks
//===----------------------------------------------------------------------===//

bool hasOperationTaggedForNCCL(ModuleOp module) {
  return module
      .walk([](Operation *op) {
        return isTaggedFor(op, ProgrammingModel::NCCL) ? WalkResult::interrupt()
                                                       : WalkResult::advance();
      })
      .wasInterrupted();
}

/// Every operation NCCL cannot express at all, with its own reason.
LogicalResult checkExpressible(Operation *op) {
  if (isLoweredByHand(op))
    return success();
  if (StringRef reason = noCounterpartReason(op); !reason.empty())
    return op->emitOpError() << reason;
  if (getCallsFor(getNCCLTableSpec(), op->getName().getStringRef()).empty())
    return op->emitOpError() << "has no NCCL counterpart";
  return success();
}

LogicalResult checkInit(InitOp init, bool hasBootstrapModel) {
  if (init.getThreadProvided())
    return init.emitOpError()
           << "asks for the thread level granted; NCCL has no thread support "
              "levels";
  if (!hasBootstrapModel)
    return init.emitOpError()
           << "starts NCCL, whose world communicator is made from a unique id "
              "that rank 0 sends the other processes over another model's "
              "collectives; the module starts neither MPI nor OpenSHMEM";
  return success();
}

LogicalResult checkWait(Operation *op) {
  if (auto wait = dyn_cast<WaitOp>(op); wait && wait.getStatus())
    return wait.emitOpError()
           << "asks for a status; ncclGroupEnd reports nothing about a single "
              "operation";
  if (auto waitAll = dyn_cast<WaitAllOp>(op); waitAll && waitAll.getStatuses())
    return waitAll.emitOpError()
           << "asks for statuses; ncclGroupEnd reports nothing about a single "
              "operation";
  return success();
}

/// Whether \p op is an operation of a group: one the raiser, or an
/// optimisation, gave a request.
bool startsGroupedOperation(Operation *op) {
  return isTaggedFor(op, ProgrammingModel::NCCL) && !isa<RequestNullOp>(op) &&
         llvm::any_of(op->getResultTypes(),
                      [](Type type) { return isa<ReqType>(type); });
}

/// The groups of \p block.  Every operation with a request opens a group of its
/// own and every request a wait completes closes one, so the lowering means
/// what the program does only where a wait completes every operation opened
/// since the last one, in the same block.  An operation of a group that no
/// such wait completes would never run, one outside a group that runs between
/// would join it, and a wait that completes fewer would not run the operations
/// it completes.
LogicalResult checkGroups(Block &block) {
  bool ok = true;
  SmallVector<Operation *> open;
  for (Operation &op : block) {
    if (startsGroupedOperation(&op)) {
      open.push_back(&op);
      continue;
    }
    if (!isTaggedFor(&op, ProgrammingModel::NCCL))
      continue;

    if (!isa<WaitOp, WaitAllOp>(op)) {
      if (!open.empty() &&
          (op.hasTrait<Communication>() || isa<GroupStartOp, GroupEndOp>(op))) {
        op.emitOpError() << "runs while " << open.size()
                         << " grouped NCCL operation(s) wait for their "
                            "ncclGroupEnd, and would be grouped with them";
        ok = false;
      }
      continue;
    }

    std::optional<uint64_t> count = completedRequests(&op);
    if (!count) {
      op.emitOpError() << "completes a number of requests that is not a "
                          "constant; the lowering emits one ncclGroupEnd per "
                          "request";
      ok = false;
      open.clear();
      continue;
    }
    if (*count != open.size()) {
      op.emitOpError()
          << "completes " << *count << " request(s), but " << open.size()
          << " grouped NCCL operation(s) are open in its block; ncclGroupEnd "
             "runs every operation of its group, so a wait has to complete all "
             "of them";
      ok = false;
    }
    // What it completes is no longer open, so that the wait completing the
    // rest is not reported as well.
    open.erase(open.begin(),
               open.begin() + std::min<uint64_t>(*count, open.size()));
  }
  for (Operation *op : open) {
    op->emitOpError()
        << "is an operation of a group that no spmd.wait in its block "
           "completes; NCCL runs a group at its ncclGroupEnd, which the "
           "lowering can place only where it sees every operation of the group";
    ok = false;
  }
  return success(ok);
}

/// Whether the communicator \p use reads is passed on to something that needs
/// it: everything except the process team, whose parallel and barrier go with
/// it, and ncclMemAlloc and ncclMemFree, which the raiser gives the world
/// communicator only because spmd.malloc and spmd.free have one.
bool passesCommunicatorOn(OpOperand &use) {
  Operation *user = use.getOwner();
  if (auto parallel = dyn_cast<ParallelOp>(user))
    return !parallel.getIsProcessLevel();
  if (isa<MallocOp, FreeOp>(user) && isTaggedFor(user, ProgrammingModel::NCCL))
    return false;
  return !isProcessTeamBarrier(user);
}

/// NCCL's world communicator exists once spmd.init has run, and spmd.commWorld
/// reads it where it stands.  A read that nothing initialised before would
/// load a null communicator, so every spmd.commWorld whose communicator is
/// used has to be dominated by an spmd.init of NCCL.
///
/// And there may be only one of those.  The dialect has one world
/// communicator, so the spmd.commWorld after a second initialisation is the
/// same operation as the one after the first -- the raiser gives both the same
/// attributes, and -cse makes them one -- and a lowering cannot tell which
/// communicator either was meant to read.
LogicalResult checkWorldCommunicator(ModuleOp module) {
  SmallVector<InitOp> inits;
  module.walk([&](InitOp init) {
    if (isTaggedFor(init, ProgrammingModel::NCCL))
      inits.push_back(init);
  });
  bool ok = true;
  for (InitOp init :
       ArrayRef<InitOp>(inits).drop_front(inits.empty() ? 0 : 1)) {
    InFlightDiagnostic diag =
        init.emitOpError()
        << "initialises NCCL a second time; the dialect has one world "
           "communicator, so the spmd.commWorld after either initialisation "
           "reads the same one";
    diag.attachNote(inits.front().getLoc()) << "NCCL is first initialised here";
    ok = false;
  }

  DominanceInfo dominance(module);
  module.walk(
      [&](CommWorldOp world) {
        if (!isTaggedFor(world, ProgrammingModel::NCCL) ||
            llvm::none_of(world->getUses(), passesCommunicatorOn))
          return;
        if (llvm::any_of(inits, [&](InitOp init) {
              return dominance.properlyDominates(init, world);
            }))
          return;
        world.emitOpError()
            << "reads NCCL's world communicator where no spmd.init of NCCL has "
               "run before it; NCCL has no world communicator until "
               "ncclCommInitRank makes one, which spmd.init does";
        ok = false;
      });
  return success(ok);
}

/// A communicator the program keeps in a variable of its own.  The raiser
/// leaves the store and the load of such a variable as casts between the
/// handle and the variable's type; a program written against MPICH declares
/// that variable an int, and an NCCL communicator is a pointer, which does not
/// fit.
LogicalResult checkHandleWidth(UnrealizedConversionCastOp cast) {
  if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
    return success();
  Type from = cast->getOperand(0).getType();
  Type to = cast->getResult(0).getType();

  Type stored;
  bool touchesNCCL = false;
  if (isa<CommType>(from) && to.isIntOrIndex()) {
    stored = to;
    if (Operation *def = cast->getOperand(0).getDefiningOp())
      touchesNCCL = isTaggedFor(def, ProgrammingModel::NCCL);
  } else if (from.isIntOrIndex() && isa<CommType>(to)) {
    stored = from;
    touchesNCCL =
        llvm::any_of(cast->getResult(0).getUsers(), [](Operation *user) {
          return isTaggedFor(user, ProgrammingModel::NCCL);
        });
  }
  // An index is as wide as a pointer, and has no bit width of its own.
  if (!touchesNCCL || isa<IndexType>(stored) ||
      stored.getIntOrFloatBitWidth() >= 64)
    return success();
  return cast.emitError()
         << "the program keeps an NCCL communicator in " << stored
         << ", but a communicator is a pointer in this ABI; a variable "
            "declared for an MPI handle is too narrow for it";
}

/// An operand the dialect has and no NCCL call has a parameter for, and one
/// whose value a call fixes -- a tag, a missing stream.  Both are what the
/// CACTS entry says read backwards, so the table finds them.  Before them, a
/// reduction whose result depends on a signedness the datatype does not
/// record: ncclInt8 and ncclUint8 compare differently.
LogicalResult checkCallable(Operation *op) {
  OpBuilder builder(op->getContext());
  SmallVector<std::string> reasons;
  std::string signedness;
  if (reducesOverUnknownSignedness(op, &signedness))
    reasons.push_back(std::move(signedness));
  selectNCCLCall(op, builder, reasons);
  for (const std::string &reason : reasons)
    op->emitOpError() << reason;
  return failure(!reasons.empty());
}

} // namespace

LogicalResult checkSPMDToNCCL(ModuleOp module) {
  if (failed(checkUsedModels(module, "lower-spmd-to-nccl")))
    return failure();
  // The calls are derived from the same API description the raiser reads, so
  // the lowering needs it too, but only for a module that has something to
  // lower: -lower-spmd runs this check for every model of its table.
  if (getNCCLLoweringTable().empty() && hasOperationTaggedForNCCL(module))
    return emitError(module.getLoc())
           << "lower-spmd-to-nccl: no NCCL API description; set "
              "NCCL_JSON_PATH to includesProgramModels/nccl/api.json";

  bool hasBootstrapModel = findBootstrapModel(module).has_value();
  bool inexpressible = false;
  module.walk([&](Operation *op) {
    if (auto cast = dyn_cast<UnrealizedConversionCastOp>(op)) {
      if (failed(checkHandleWidth(cast)))
        inexpressible = true;
      return;
    }
    if (!isTaggedFor(op, ProgrammingModel::NCCL))
      return;
    // All of them run, so an operation wrong on more than one count reports
    // every reason at once.
    bool handle = succeeded(checkPredefinedHandle(op));
    bool expressible = succeeded(checkExpressible(op));
    bool callable = succeeded(checkCallable(op));
    bool wait = succeeded(checkWait(op));
    bool init = true;
    if (auto initOp = dyn_cast<InitOp>(op))
      init = succeeded(checkInit(initOp, hasBootstrapModel));
    if (!handle || !expressible || !callable || !wait || !init)
      inexpressible = true;
  });

  module.walk([&](Block *block) {
    if (failed(checkGroups(*block)))
      inexpressible = true;
  });
  if (failed(checkWorldCommunicator(module)))
    inexpressible = true;
  return failure(inexpressible);
}

LogicalResult lowerSPMDToNCCL(ModuleOp module, StructureLowering structure) {
  MLIRContext *ctx = module.getContext();

  if (failed(checkSPMDToNCCL(module)))
    return failure();

  // What the patterns read and call has to exist before the conversion starts:
  // the global spmd.commWorld loads, and the bootstrap spmd.init calls.
  Operation *firstInit = nullptr, *firstWorld = nullptr;
  module.walk([&](Operation *op) {
    if (!isTaggedFor(op, ProgrammingModel::NCCL))
      return;
    if (isa<InitOp>(op) && !firstInit)
      firstInit = op;
    if (isa<CommWorldOp>(op) && !firstWorld)
      firstWorld = op;
  });
  LLVM::LLVMFuncOp bootstrap;
  if (firstInit) {
    FailureOr<LLVM::LLVMFuncOp> created =
        getOrCreateBootstrap(module, firstInit, *findBootstrapModel(module));
    if (failed(created))
      return failure();
    bootstrap = *created;
  } else if (firstWorld) {
    OpBuilder builder(ctx);
    if (failed(getOrCreateGlobal(builder, firstWorld, kCommWorldGlobal,
                                 LLVM::LLVMPointerType::get(ctx))))
      return failure();
  }

  NCCLTypeConverter converter(ctx);
  RewritePatternSet patterns(ctx);
  patterns.add<LowerInit>(converter, ctx, bootstrap);
  patterns
      .add<LowerCommWorld, LowerEnumerator<ConstDatatypeOp>,
           LowerEnumerator<ConstOpTypeOp>, LowerEnumerator<ConstThreadLevelOp>,
           LowerRequestNull, LowerCastStream, LowerWaitToGroupEnd<WaitOp>,
           LowerWaitToGroupEnd<WaitAllOp>>(converter, ctx);
  populateNCCLCallPatterns(converter, patterns);
  populateSPMDStructureLoweringPatterns(converter, patterns,
                                        ProgrammingModel::NCCL, structure);

  return applySPMDLowering(module, ProgrammingModel::NCCL, converter,
                           std::move(patterns));
}

namespace {

struct LowerSPMDtoNCCLPass
    : public impl::LowerSPMDtoNCCLBase<LowerSPMDtoNCCLPass> {
  using LowerSPMDtoNCCLBase::LowerSPMDtoNCCLBase;

  void runOnOperation() override {
    if (failed(lowerSPMDToNCCL(getOperation())))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::spmd
