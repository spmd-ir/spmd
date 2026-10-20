// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir::spmd {

// Define Base Class and Default Constructor for main Class.
#define GEN_PASS_DEF_PRIVATIZESYMBOLSPASS
#include "spmd/SPMDPasses.h.inc"

namespace {

struct PrivatizeSymbolsPass
    : public impl::PrivatizeSymbolsPassBase<PrivatizeSymbolsPass> {
  using Base = impl::PrivatizeSymbolsPassBase<PrivatizeSymbolsPass>;
  using Base::Base;

  LogicalResult initialize(MLIRContext *context) override {
    for (const std::string &symbol : exclude)
      excludedSymbols.insert(StringAttr::get(context, symbol));
    return success();
  }

  void runOnOperation() override {
    Operation *root = getOperation();

    for (Region &region : root->getRegions()) {
      for (Block &block : region) {
        for (Operation &op : block) {
          auto symbol = dyn_cast<SymbolOpInterface>(op);
          if (!symbol)
            continue;

          StringAttr name = symbol.getNameAttr();
          if (!name)
            continue;

          if (excludedSymbols.contains(name))
            continue;

          if (auto func = dyn_cast<func::FuncOp>(op)) {
            if (!func.isExternal())
              symbol.setVisibility(SymbolTable::Visibility::Private);
            continue;
          }

          if (auto llvmFunc = dyn_cast<LLVM::LLVMFuncOp>(op)) {
            if (!llvmFunc.isExternal())
              symbol.setVisibility(SymbolTable::Visibility::Private);
            continue;
          }
        }
      }
    }
  }

private:
  DenseSet<StringAttr> excludedSymbols;
};

} // namespace
} // namespace mlir::spmd