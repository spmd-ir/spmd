// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "spmd/SPMDOps.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "spmd/SPMDDialect.h"

namespace mlir::spmd {

struct RuntimeResource : public SideEffects::Resource::Base<RuntimeResource> {
  StringRef getName() const final { return "SPMDRuntime"; }
  bool isAddressable() const final { return false; }
};

struct CommResource : public SideEffects::Resource::Base<CommResource> {
  StringRef getName() const final { return "SPMDComm"; }
  // bool isAddressable() const final { return false; }
};

struct WinResource : public SideEffects::Resource::Base<WinResource> {
  StringRef getName() const final { return "SPMDWin"; }
  // bool isAddressable() const final { return false; }
};

struct ReqResource : public SideEffects::Resource::Base<ReqResource> {
  StringRef getName() const final { return "SPMDReq"; }
  // bool isAddressable() const final { return false; }
};

struct StreamResource : public SideEffects::Resource::Base<StreamResource> {
  StringRef getName() const final { return "SPMDStream"; }
  // bool isAddressable() const final { return false; }
};

struct MsgResource : public SideEffects::Resource::Base<MsgResource> {
  StringRef getName() const final { return "SPMDMsg"; }
  // bool isAddressable() const final { return false; }
};
// struct ErrorResource : public SideEffects::Resource::Base<ErrorResource> {
//   StringRef getName() const final { return "SPMDError"; }
// };

// struct CommConfigResource
//     : public SideEffects::Resource::Base<CommConfigResource> {
//   StringRef getName() const final { return "SPMDCommConfig"; }
// };

// struct StatusResource : public SideEffects::Resource::Base<StatusResource> {
//   StringRef getName() const final { return "SPMDStatus"; }
// };

// struct DatatypeResource : public
// SideEffects::Resource::Base<DatatypeResource> {
//   StringRef getName() const final { return "SPMDDatatype"; }
// };

// struct OpResource : public SideEffects::Resource::Base<OpResource> {
//   StringRef getName() const final { return "SPMDOp"; }
// };

// struct SignalOpResource : public
// SideEffects::Resource::Base<SignalOpResource> {
//   StringRef getName() const final { return "SPMDSignalOp"; }
// };

// struct LockResource : public SideEffects::Resource::Base<LockResource> {
//   StringRef getName() const final { return "SPMDLock"; }
// };

} // namespace mlir::spmd

#define GET_OP_CLASSES
#include "spmd/SPMDOps.cpp.inc"

using namespace mlir;
using namespace spmd;

LogicalResult spmd::DeclareReductionOp::verify() {
  auto kindAttr = getKind();
  if (!kindAttr)
    return success();

  Region &combiner = getReductionRegion();
  if (combiner.empty() || combiner.front().getOperations().size() != 2)
    return success(); // non-trivial body — skip check

  Operation &combineOp = combiner.front().front();
  ReductionKind k = *kindAttr;

  auto expect = [&](bool ok, StringRef msg) -> LogicalResult {
    if (!ok)
      return emitOpError() << "$kind=" << stringifyReductionKind(k)
                           << " inconsistent with combiner: " << msg;
    return success();
  };

  if (k == ReductionKind::Add)
    return expect(isa<arith::AddIOp, arith::AddFOp>(combineOp),
                  "expected arith.addi or arith.addf");
  if (k == ReductionKind::Mul)
    return expect(isa<arith::MulIOp, arith::MulFOp>(combineOp),
                  "expected arith.muli or arith.mulf");
  if (k == ReductionKind::MinS)
    return expect(isa<arith::MinSIOp>(combineOp), "expected arith.minsi");
  if (k == ReductionKind::MinU)
    return expect(isa<arith::MinUIOp>(combineOp), "expected arith.minui");
  if (k == ReductionKind::MinF)
    return expect(isa<arith::MinimumFOp>(combineOp), "expected arith.minimumf");
  if (k == ReductionKind::MaxS)
    return expect(isa<arith::MaxSIOp>(combineOp), "expected arith.maxsi");
  if (k == ReductionKind::MaxU)
    return expect(isa<arith::MaxUIOp>(combineOp), "expected arith.maxui");
  if (k == ReductionKind::MaxF)
    return expect(isa<arith::MaximumFOp>(combineOp), "expected arith.maximumf");
  if (k == ReductionKind::And)
    return expect(isa<arith::AndIOp>(combineOp), "expected arith.andi");
  if (k == ReductionKind::Or)
    return expect(isa<arith::OrIOp>(combineOp), "expected arith.ori");
  if (k == ReductionKind::Xor)
    return expect(isa<arith::XOrIOp>(combineOp), "expected arith.xori");

  return success();
}

LogicalResult spmd::ParallelOp::verify() {
  Region &r = getRegion();
  if (r.empty())
    return emitOpError("parallel region must not be empty");

  bool hasExit = static_cast<bool>(getExitCode());

  for (Block &block : r) {
    Operation *term = block.getTerminator();
    if (!term)
      return emitOpError("block in parallel region has no terminator");

    // Interior blocks (cf.switch, cf.br, cf.cond_br, …) branch to other
    // blocks within the same region — nothing to check here.
    if (term->getNumSuccessors() > 0)
      continue;

    // Leaf block: must exit via parallel_yield or be a noreturn path.
    if (isa<LLVM::UnreachableOp>(term))
      continue;

    auto yieldOp = dyn_cast<spmd::ParallelYieldOp>(term);
    if (!yieldOp)
      return emitOpError("leaf block in parallel region must terminate with "
                         "spmd.parallel_yield or llvm.unreachable, got: ")
             << term->getName();

    if (hasExit && yieldOp->getNumOperands() == 0)
      return emitOpError("parallel_yield must carry an exit code when "
                         "spmd.parallel has an exit_code result");
    if (!hasExit && yieldOp->getNumOperands() != 0)
      return emitOpError("parallel_yield must not carry operands when "
                         "spmd.parallel has no exit_code result");
  }
  return success();
}
// OpFoldResult SPMD_CommWorldOp::fold(FoldAdaptor adaptor) {
//   return getUsedModelAttr();
// }
// OpFoldResult SPMD_CommNodeOp::fold(FoldAdaptor adaptor) {
//   return getUsedModelAttr();
// }

// OpFoldResult SPMD_ConstDatatypeOp::fold(FoldAdaptor adaptor) {
//   auto context = getContext();
//   return DictionaryAttr::get(context, {
//     NamedAttribute(StringAttr::get(context, "type"), getTypeAttr()),
//     NamedAttribute(StringAttr::get(context, "usedModel"), getUsedModelAttr())
//   });
// }

// OpFoldResult SPMD_ConstOpTypeOp::fold(FoldAdaptor adaptor) {
//   auto context = getContext();
//   return DictionaryAttr::get(context, {
//     NamedAttribute(StringAttr::get(context, "string"), getStringAttr()),
//     NamedAttribute(StringAttr::get(context, "usedModel"), getUsedModelAttr())
//   });
// }

// OpFoldResult SPMD_ConstSignalOpTypeOp::fold(FoldAdaptor adaptor) {
//   auto context = getContext();
//   return DictionaryAttr::get(context, {
//     NamedAttribute(StringAttr::get(context, "string"), getStringAttr()),
//     NamedAttribute(StringAttr::get(context, "usedModel"), getUsedModelAttr())
//   });
// }
// OpFoldResult SPMD_CommSplitOp::fold(FoldAdaptor adaptor) {
//   return getUsedModelAttr();
// }

OpFoldResult GetParallelCommOp::fold(FoldAdaptor) {
  // Walk up the region stack looking for an enclosing spmd.parallel
  Operation *parent = (*this)->getParentOp();
  while (parent) {
    if (auto parallelOp = dyn_cast<spmd::ParallelOp>(parent))
      return parallelOp.getComm();
    // Stop at function boundaries — no enclosing parallel exists
    if (isa<FunctionOpInterface>(parent))
      break;
    parent = parent->getParentOp();
  }
  return {}; // leave in place — top-level orphaned directive uses commWorld
}