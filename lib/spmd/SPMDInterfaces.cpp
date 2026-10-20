// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "spmd/SPMDInterfaces.h"

#include "spmd/SPMDOps.h"

#include "spmd/SPMDInterfaces.cpp.inc"

namespace mlir::spmd {

static void setSend(CommBufferOperands &operands, OpOperand &buffer,
                    OpOperand &count, OpOperand &datatype) {
  operands.sendBuffer = &buffer;
  operands.sendCount = &count;
  operands.sendDatatype = &datatype;
}

static void setRecv(CommBufferOperands &operands, OpOperand &buffer,
                    OpOperand &count, OpOperand &datatype) {
  operands.recvBuffer = &buffer;
  operands.recvCount = &count;
  operands.recvDatatype = &datatype;
}

//===----------------------------------------------------------------------===//
// Point-to-point
//===----------------------------------------------------------------------===//

CommBufferOperands SendOp::getCommBufferOperands() {
  CommBufferOperands operands;
  setSend(operands, getSendBufMutable(), getDataCountMutable(),
          getDatatypeMutable());
  return operands;
}

CommBufferOperands RecvOp::getCommBufferOperands() {
  CommBufferOperands operands;
  setRecv(operands, getRecvBufMutable(), getDataCountMutable(),
          getDatatypeMutable());
  return operands;
}

CommBufferOperands SendrecvOp::getCommBufferOperands() {
  CommBufferOperands operands;
  setSend(operands, getSendBufMutable(), getSendCountMutable(),
          getSendDatatypeMutable());
  operands.sendStride = &getSendStrideMutable();
  setRecv(operands, getRecvBufMutable(), getRecvCountMutable(),
          getRecvDatatypeMutable());
  operands.recvStride = &getRecvStrideMutable();
  return operands;
}

CommBufferOperands MRecvOp::getCommBufferOperands() {
  CommBufferOperands operands;
  setRecv(operands, getRecvBufMutable(), getDataCountMutable(),
          getDatatypeMutable());
  return operands;
}

CommBufferOperands IMRecvOp::getCommBufferOperands() {
  CommBufferOperands operands;
  setRecv(operands, getRecvBufMutable(), getDataCountMutable(),
          getDatatypeMutable());
  return operands;
}

//===----------------------------------------------------------------------===//
// Collectives with one count and one datatype for both buffers
//===----------------------------------------------------------------------===//

/// The shape shared by bcast and the reductions: one count and one datatype
/// describe both buffers.
template <typename OpT> static CommBufferOperands getSymmetricOperands(OpT op) {
  CommBufferOperands operands;
  setSend(operands, op.getSendBufMutable(), op.getDataCountMutable(),
          op.getDatatypeMutable());
  setRecv(operands, op.getRecvBufMutable(), op.getDataCountMutable(),
          op.getDatatypeMutable());
  return operands;
}

CommBufferOperands BcastOp::getCommBufferOperands() {
  CommBufferOperands operands = getSymmetricOperands(*this);
  operands.root = &getRootRankMutable();
  // The root's buffer is the one being broadcast; nothing overwrites it.
  operands.writesRecvBufferOnAllRanks = false;
  return operands;
}

CommBufferOperands ReduceOp::getCommBufferOperands() {
  CommBufferOperands operands = getSymmetricOperands(*this);
  operands.root = &getRootRankMutable();
  operands.writesRecvBufferOnAllRanks = false;
  return operands;
}

CommBufferOperands AllreduceOp::getCommBufferOperands() {
  return getSymmetricOperands(*this);
}

CommBufferOperands ReduceScatterOp::getCommBufferOperands() {
  // MPI_Reduce_scatter_block and ncclReduceScatter: the count is what each
  // rank receives, and every rank contributes that many elements per rank.
  CommBufferOperands operands = getSymmetricOperands(*this);
  operands.sendCountIsPerRank = true;
  return operands;
}

CommBufferOperands ScanOp::getCommBufferOperands() {
  return getSymmetricOperands(*this);
}

CommBufferOperands ExscanOp::getCommBufferOperands() {
  // MPI_Exscan leaves the receive buffer of rank 0 undefined.
  CommBufferOperands operands = getSymmetricOperands(*this);
  operands.writesRecvBufferOnAllRanks = false;
  return operands;
}

//===----------------------------------------------------------------------===//
// Collectives with separate send and receive counts and datatypes
//===----------------------------------------------------------------------===//

template <typename OpT>
static CommBufferOperands getAsymmetricOperands(OpT op) {
  CommBufferOperands operands;
  setSend(operands, op.getSendBufMutable(), op.getSendCountMutable(),
          op.getSendTypeMutable());
  setRecv(operands, op.getRecvBufMutable(), op.getRecvCountMutable(),
          op.getRecvTypeMutable());
  return operands;
}

CommBufferOperands ScatterOp::getCommBufferOperands() {
  CommBufferOperands operands = getAsymmetricOperands(*this);
  operands.root = &getRootRankMutable();
  operands.sendCountIsPerRank = true;
  return operands;
}

CommBufferOperands GatherOp::getCommBufferOperands() {
  CommBufferOperands operands = getAsymmetricOperands(*this);
  operands.root = &getRootRankMutable();
  operands.recvCountIsPerRank = true;
  operands.writesRecvBufferOnAllRanks = false;
  return operands;
}

CommBufferOperands AllgatherOp::getCommBufferOperands() {
  CommBufferOperands operands = getAsymmetricOperands(*this);
  operands.recvCountIsPerRank = true;
  return operands;
}

CommBufferOperands AlltoallOp::getCommBufferOperands() {
  CommBufferOperands operands = getAsymmetricOperands(*this);
  operands.sendCountIsPerRank = true;
  operands.recvCountIsPerRank = true;
  return operands;
}

} // namespace mlir::spmd
