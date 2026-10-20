// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- LowerSPMDCommon.h - Shared pieces of the SPMD lowerings --*- C++ -*-===//
//
// What the passes that lower the dialect to one programming model's calls
// share: the model ids, the conversion driver and its legality rules, the base
// type converter, the process-level spmd.parallel the raiser adds, runtime
// function declarations, and stack storage for out-parameters.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_CONVERSION_LOWERSPMDCOMMON_H
#define SPMD_CONVERSION_LOWERSPMDCOMMON_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/STLFunctionalExtras.h"

#include <cstdint>
#include <string>

namespace mlir::spmd {

/// The usedModel ids -convert-apiCalls-to-spmd assigns (registerApi in
/// ConvertApiCallsToSPMD.cpp).
enum class ProgrammingModel : int32_t {
  MPI = 0,
  SHMEM = 1,
  NVSHMEM = 2,
  NCCL = 3,
  CUDA = 4,
  OMP = 5,
};

/// What a lowering does with the structure -convert-apiCalls-to-spmd puts
/// around the program.
enum class StructureLowering {
  /// The lowering splices the process-level spmd.parallel of its own model
  /// back into the function, as the single-model passes (-lower-spmd-to-mpi)
  /// do: a module they leave half lowered keeps every other model's.
  PerModel,
  /// The lowering leaves the process-level parallels alone.  A driver lowering
  /// several models splices those of every model it lowers in one step
  /// (lowerProcessParallels) before it lowers any of them: the process team
  /// belongs to the launcher, not to a model.  Undoing that structure first
  /// also keeps the calls out of it -- a parallel still standing during the
  /// conversion would need its communicator materialised back into an
  /// !spmd.comm, and that cast would outlive it.
  Deferred,
};

/// Whether \p op carries the usedModel of \p model.
bool isTaggedFor(Operation *op, ProgrammingModel model);

/// Whether \p op is GPU device code: inside a gpu.module or a gpu.launch, or in
/// a function compiled for an NVIDIA GPU.  The frontend links a CUDA program's
/// device module into the host module, so a kernel reaches the passes as an
/// ordinary llvm.func whose target_cpu is the GPU's (sm_89), launched from a
/// gpu.launch in host code.
bool isInDeviceCode(Operation *op);

/// Reports every spmd operation whose usedModel names no programming model.
LogicalResult checkUsedModels(ModuleOp module, StringRef passName);

/// Whether \p op reduces with an operation whose result on integers depends on
/// their signedness -- MAX, MIN, their LOC forms, and NCCL's AVG, which
/// divides -- in any model's spelling.
bool reducesBySignedness(Operation *op);

/// Whether \p op reduces as reducesBySignedness says over a datatype that does
/// not record its signedness: a signless integer, such as the i32 of
/// shmem_put32.  No call can be picked for it without guessing, and a wrong
/// guess changes the result -- unless the target has a datatype that leaves
/// the signedness open too, which \p isOwnDatatype says of a type: MPI_CHAR
/// is i8 exactly.  \p reason, when given, gets the tail of a diagnostic on
/// \p op.
bool reducesOverUnknownSignedness(
    Operation *op, std::string *reason = nullptr,
    llvm::function_ref<bool(Type)> isOwnDatatype = nullptr);

/// The type converter a lowering starts from.  A type without a rule stays as
/// it is, and a value passing between an operation the lowering converts and
/// one it does not is bridged by an unrealized cast, which
/// -spmd-reconcile-unrealized-cast folds once both sides agree -- except
/// between a memref and a pointer, which enzymexla.memref2pointer and
/// enzymexla.pointer2memref bridge.
class SPMDLoweringTypeConverter : public TypeConverter {
public:
  explicit SPMDLoweringTypeConverter(MLIRContext *ctx);
};

/// Converts every spmd operation tagged for \p model with \p patterns, plus
/// the spmd.castSPMDErrorToI32 and spmd.castThreadLevelToI32 of their results.
/// Operations of other models, and casts of their results, are left alone.  An
/// operation tagged for \p model that no pattern converts fails the
/// conversion -- including a process-level spmd.parallel that \p patterns do
/// not lower because lowerProcessParallels was expected to have spliced it.
///
/// \p alsoLowers names the spmd operations without a usedModel of their own
/// that belong to \p model all the same, such as the terminators of its
/// regions; they have to be converted too.  \p targetDialects names the
/// dialects besides LLVM whose operations the patterns build: an operation a
/// pattern creates has to be legal, and the conversion knows nothing of any
/// other dialect.
LogicalResult
applySPMDLowering(ModuleOp module, ProgrammingModel model,
                  const TypeConverter &converter, RewritePatternSet &&patterns,
                  function_ref<bool(Operation *)> alsoLowers = nullptr,
                  ArrayRef<StringRef> targetDialects = {});

/// Lowers spmd.castSPMDErrorToI32 and spmd.castThreadLevelToI32 to their
/// operand once it has become an i32, and the process-level spmd.parallel of
/// \p model to its region, spliced into the block around it without the
/// self-matching barrier the raiser puts before each spmd.parallel_yield.
/// With \p structure deferred, only the casts are lowered.
void populateSPMDStructureLoweringPatterns(
    const TypeConverter &converter, RewritePatternSet &patterns,
    ProgrammingModel model,
    StructureLowering structure = StructureLowering::PerModel);

/// Splices the process-level spmd.parallel tagged for any of \p models into
/// the block around it, as the per-model structure patterns do for one model.
/// A parallel of a model that is not lowered, and one with no usedModel at
/// all, stays: its operations still need the team.
LogicalResult lowerProcessParallels(ModuleOp module,
                                    ArrayRef<ProgrammingModel> models);

/// The declaration of the runtime function \p name in the module around
/// \p anchor, created with \p type when the module has none.  A symbol of that
/// name that is not an LLVM function of type \p type is reported on \p anchor:
/// calling it anyway would pass the arguments under the wrong ABI.
FailureOr<LLVM::LLVMFuncOp>
getOrDeclareRuntimeFunction(OpBuilder &builder, Operation *anchor,
                            StringRef name, LLVM::LLVMFunctionType type);

/// A call to the runtime function \p callee, declared in the module around
/// \p anchor with the types of \p args if it is not there yet.  Failure is
/// what getOrDeclareRuntimeFunction reports on \p anchor: a symbol of that
/// name that is not an LLVM function, or one declared with other types.
FailureOr<LLVM::CallOp> emitRuntimeCall(OpBuilder &builder, Operation *anchor,
                                        Location loc, StringRef callee,
                                        Type returnType, ValueRange args);

Value createI32Constant(OpBuilder &builder, Location loc, int32_t value);
Value createI64Constant(OpBuilder &builder, Location loc, int64_t value);

/// Stack storage for one value of \p type, such as an out-parameter.  The
/// alloca goes into the entry block of the nearest region around the builder's
/// insertion point whose op owns stack allocations (AutomaticAllocationScope):
/// usually the function, but a thread team's region keeps its own, per-thread
/// storage.  An alloca next to a call inside a loop would grow the stack on
/// every iteration.
Value createStackSlot(OpBuilder &builder, Location loc, Type type);

} // namespace mlir::spmd

#endif // SPMD_CONVERSION_LOWERSPMDCOMMON_H
