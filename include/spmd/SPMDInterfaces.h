/*
 * SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 */

//===- SPMDInterfaces.h - SPMD op interfaces --------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_INTERFACES_H
#define SPMD_INTERFACES_H

#include "mlir/IR/OpDefinition.h"

namespace mlir::spmd {

/// The operands of a CommBufferOpInterface op, by role. A role the op does not
/// have is null. Operands rather than values are handed out so that a rewrite
/// can change one in place without knowing the op's operand order.
struct CommBufferOperands {
  OpOperand *sendBuffer = nullptr;
  OpOperand *sendCount = nullptr;
  OpOperand *sendDatatype = nullptr;
  /// Stride between the elements sent, for the ops that carry one.
  OpOperand *sendStride = nullptr;

  OpOperand *recvBuffer = nullptr;
  OpOperand *recvCount = nullptr;
  OpOperand *recvDatatype = nullptr;
  /// Stride between the elements received, for the ops that carry one.
  OpOperand *recvStride = nullptr;

  OpOperand *root = nullptr;

  /// The send count is one rank's share: the send buffer holds that many
  /// elements for every rank of the communicator (spmd.scatter,
  /// spmd.alltoall, and the send side of spmd.reduceScatter).
  bool sendCountIsPerRank = false;
  /// The receive count is one rank's share: the receive buffer holds that many
  /// elements for every rank of the communicator (spmd.gather,
  /// spmd.allgather, spmd.alltoall).
  bool recvCountIsPerRank = false;
  /// Every rank that executes the op overwrites its receive buffer. False when
  /// some rank may keep what the buffer held before: the non-roots of
  /// spmd.reduce and spmd.gather, the root of spmd.bcast, rank 0 of
  /// spmd.exscan.
  bool writesRecvBufferOnAllRanks = true;

  static Value valueOf(OpOperand *operand) {
    return operand ? operand->get() : Value();
  }
};

} // namespace mlir::spmd

#include "spmd/SPMDInterfaces.h.inc"

#endif // SPMD_INTERFACES_H
