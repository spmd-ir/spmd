// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/Utils/PassUtils.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

#include "llvm/ADT/TypeSwitch.h"

namespace mlir::spmd {

/// Recursively unwrap location wrappers to find a FileLineColLoc.
/// copied over from Diagnostics.cpp
static std::optional<Location> findLocToShow(Location loc) {
  // if (!shouldShowLocFn)
  //   return loc;
  // if (!shouldShowLocFn(loc))
  //   return std::nullopt;

  // Recurse into the child locations of some of location types.
  return TypeSwitch<LocationAttr, std::optional<Location>>(loc)
      .Case([&](CallSiteLoc callLoc) -> std::optional<Location> {
        // We recurse into the callee of a call site, as the caller will be
        // emitted in a different note on the main diagnostic.
        return findLocToShow(callLoc.getCallee());
      })
      .Case([&](FileLineColLoc) -> std::optional<Location> { return loc; })
      .Case([&](FusedLoc fusedLoc) -> std::optional<Location> {
        // Fused location is unique in that we try to find a sub-location to
        // show, rather than the top-level location itself.
        for (Location childLoc : fusedLoc.getLocations())
          if (std::optional<Location> showableLoc = findLocToShow(childLoc))
            return showableLoc;
        return std::nullopt;
      })
      .Case([&](NameLoc nameLoc) -> std::optional<Location> {
        return findLocToShow(nameLoc.getChildLoc());
      })
      .Case([&](OpaqueLoc opaqueLoc) -> std::optional<Location> {
        // OpaqueLoc always falls back to a different source location.
        return findLocToShow(opaqueLoc.getFallbackLocation());
      })
      .Case([](UnknownLoc) -> std::optional<Location> {
        // Prefer not to show unknown locations.
        return std::nullopt;
      });
}

Location findLocToShowWrapper(Location loc) {
  return findLocToShow(loc).value_or(loc);
}

SmallVector<Value> getConditionOperands(Operation *op) {
  return TypeSwitch<Operation *, SmallVector<Value>>(op)
      // probably selectOps and cf::CondBranchOp is never fired due to callsites
      .Case<arith::SelectOp, LLVM::SelectOp, cf::CondBranchOp, LLVM::CondBrOp,
            scf::IfOp>([](auto branchOp) {
        return SmallVector<Value>{branchOp.getCondition()};
      })
      .Case<affine::AffineIfOp>([](affine::AffineIfOp ifOp) {
        return SmallVector<Value>(ifOp.getOperands().begin(),
                                  ifOp.getOperands().end());
      })
      .Case<scf::WhileOp>([](scf::WhileOp whileOp) {
        return SmallVector<Value>{whileOp.getConditionOp().getCondition()};
      })
      .Case<scf::IndexSwitchOp>([](scf::IndexSwitchOp switchOp) {
        return SmallVector<Value>{switchOp.getArg()};
      })
      .Default([](Operation *) { return SmallVector<Value>{}; });
}

// TODO this need to work also for function calls etc
SmallVector<Operation *> getEnclosingRegionBranches(Operation *op) {
  SmallVector<Operation *> branches;
  for (Operation *parent = op ? op->getParentOp() : nullptr; parent;
       parent = parent->getParentOp()) {
    if (isa<RegionBranchOpInterface>(parent))
      branches.push_back(parent);
  }
  return branches;
}

} // namespace mlir::spmd