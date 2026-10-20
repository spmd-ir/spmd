// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// -test-spmd-buffer-slices: for every op implementing CommBufferOpInterface,
// writes what the buffer queries answer for its send and receive buffer.
//
//   test.send / test.recv    "<base> + <offset>, <n> bytes, <facts>"
//       <offset>  resolveBufferSlice's displacement: a constant plus
//                 <scale>*<value> terms, or "?" when it was not recoverable
//       <n>       getDatatypeByteSize of the side's datatype, or "?"
//       <facts>   computeBufferFacts: "escapes", or
//                 "private, readers=<r>, writers=<w>"
//   test.root                  the root operand, for an op with one
//   test.send_count_per_rank   present when CommBufferOperands says so
//   test.recv_count_per_rank   likewise
//   test.recv_not_on_all_ranks present when writesRecvBufferOnAllRanks is false

#include "TestPasses.h"

#include "spmd/Analysis/CommBuffers.h"
#include "spmd/SPMDOps.h"
#include "spmd/Utils/MemoryOffsets.h"
#include "spmd/Utils/PassUtils.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir::spmd::test {
namespace {

std::string describeBuffer(OpOperand *buffer, OpOperand *datatype,
                           const DataLayout &dl) {
  std::string rendered;
  llvm::raw_string_ostream os(rendered);

  BufferSlice slice = resolveBufferSlice(buffer->get(), dl);
  os << valAsString(slice.base) << " + ";
  if (slice.offset)
    slice.offset->print(os);
  else
    os << "?";

  os << ", ";
  if (std::optional<int64_t> bytes =
          getDatatypeByteSize(CommBufferOperands::valueOf(datatype), dl))
    os << *bytes;
  else
    os << "?";
  os << " bytes, ";

  BufferFacts facts;
  if (failed(computeBufferFacts(buffer->get(), dl, facts)))
    os << "escapes";
  else
    os << "private, readers=" << facts.readers.size()
       << ", writers=" << facts.writers.size();
  return rendered;
}

struct TestBufferSlicesPass
    : public PassWrapper<TestBufferSlicesPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TestBufferSlicesPass)

  StringRef getArgument() const final { return "test-spmd-buffer-slices"; }
  StringRef getDescription() const final {
    return "Annotate the buffers of communication ops with their slice, "
           "element size and readers and writers";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    DataLayout dl(module);
    Builder builder(module.getContext());

    module.walk([&](CommBufferOpInterface op) {
      CommBufferOperands operands = op.getCommBufferOperands();
      if (operands.sendBuffer)
        op->setAttr("test.send", builder.getStringAttr(describeBuffer(
                                     operands.sendBuffer,
                                     operands.sendDatatype, dl)));
      if (operands.recvBuffer)
        op->setAttr("test.recv", builder.getStringAttr(describeBuffer(
                                     operands.recvBuffer,
                                     operands.recvDatatype, dl)));
      if (operands.root)
        op->setAttr("test.root",
                    builder.getStringAttr(valAsString(operands.root->get())));
      if (operands.sendCountIsPerRank)
        op->setAttr("test.send_count_per_rank", builder.getUnitAttr());
      if (operands.recvCountIsPerRank)
        op->setAttr("test.recv_count_per_rank", builder.getUnitAttr());
      if (operands.recvBuffer && !operands.writesRecvBufferOnAllRanks)
        op->setAttr("test.recv_not_on_all_ranks", builder.getUnitAttr());
    });
  }
};

} // namespace

void registerTestBufferSlicesPass() { PassRegistration<TestBufferSlicesPass>(); }

} // namespace mlir::spmd::test
