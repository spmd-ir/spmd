/*
 * SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

#ifndef SPMD_ANALYSIS_HANDLEFACTS_H
#define SPMD_ANALYSIS_HANDLEFACTS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/raw_ostream.h"

namespace mlir::spmd {

using HandleAtom = Value;
using HandleAtomSet = llvm::SmallDenseSet<HandleAtom>;

/// Tracks which handle atoms (concrete Values) a handle may refer to.
/// `unknown` means the set is unbounded / pessimistic.
struct HandleFactsValue {
  HandleAtomSet atoms;
  bool unknown = false;

  static HandleFactsValue getUnknown() {
    HandleFactsValue v;
    v.unknown = true;
    return v;
  }

  static HandleFactsValue getAtom(Value value) {
    HandleFactsValue v;
    v.atoms.insert(value);
    return v;
  }

  static HandleFactsValue join(const HandleFactsValue &lhs,
                               const HandleFactsValue &rhs) {
    HandleFactsValue out = lhs;
    (void)out.joinInPlace(rhs);
    return out;
  }

  ChangeResult joinInPlace(const HandleFactsValue &rhs) {
    bool changed = false;
    if (rhs.unknown && !unknown) {
      unknown = true;
      changed = true;
    }
    const size_t oldSize = atoms.size();
    atoms.insert(rhs.atoms.begin(), rhs.atoms.end());
    changed |= atoms.size() != oldSize;
    return changed ? ChangeResult::Change : ChangeResult::NoChange;
  }

  bool operator==(const HandleFactsValue &rhs) const {
    return unknown == rhs.unknown && atoms == rhs.atoms;
  }

  bool operator!=(const HandleFactsValue &rhs) const { return !(*this == rhs); }

  bool emptyKnown() const { return !unknown && atoms.empty(); }

  void print(llvm::raw_ostream &os) const {
    os << "{unknown=" << (unknown ? "true" : "false") << ", atoms=[";
    llvm::interleaveComma(
        atoms, os, [&](Value v) { v.printAsOperand(os, OpPrintingFlags()); });
    os << "]}";
  }
};

inline llvm::raw_ostream &operator<<(llvm::raw_ostream &os,
                                     const HandleFactsValue &v) {
  v.print(os);
  return os;
}

class HandleFactsLattice : public dataflow::Lattice<HandleFactsValue> {
public:
  using dataflow::Lattice<HandleFactsValue>::Lattice;
};

} // namespace mlir::spmd

#endif // SPMD_ANALYSIS_HANDLEFACTS_H