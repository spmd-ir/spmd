// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===-- SPMDInterfaces.td - SPMD op interfaces -------------*- tablegen -*-===//
//
// Op interfaces of the SPMD dialect.
//
//===----------------------------------------------------------------------===//

#ifndef SPMD_INTERFACES
#define SPMD_INTERFACES

include "mlir/IR/OpBase.td"

def SPMD_CommBufferOpInterface : OpInterface<"CommBufferOpInterface"> {
  let cppNamespace = "::mlir::spmd";
  let description = [{
    A communication op that moves data between a local buffer and other ranks:
    the point-to-point sends and receives, and the collectives that move data.
    The interface names the operands describing each buffer (address, element
    count, datatype and, where the op has one, stride) and the root, however
    the op spells them. It also states two facts an optimization cannot read
    off the operand list:

      * whether a count describes the whole buffer or one rank's share of it.
        MPI_Gather's recvcount is per process, so the root's receive buffer
        holds recvcount * size elements;
      * whether every rank that executes the op overwrites its receive buffer.
        spmd.reduce and spmd.gather write it on the root only, spmd.bcast
        leaves the root's buffer as it was, and MPI_Exscan leaves rank 0's
        undefined.

    Transfers that do not happen at the op itself are not covered: the
    persistent spmd.sendInit/spmd.recvInit move data at spmd.start, and the RMA
    ops address a window rather than a local buffer. For a non-blocking op the
    buffers are read and written by the time it completes, not at the op; that
    is for the caller to account for.
  }];
  let methods = [
    InterfaceMethod<[{
        Returns the operands describing the op's send and receive buffers and
        its root, with the roles the op does not have left null.
      }],
      "::mlir::spmd::CommBufferOperands", "getCommBufferOperands", (ins)>,
  ];
  let extraSharedClassDeclaration = [{
    /// Address of the send buffer, or null when the op sends nothing.
    ::mlir::Value getSendBuffer() {
      return ::mlir::spmd::CommBufferOperands::valueOf(
          $_op.getCommBufferOperands().sendBuffer);
    }
    /// Address of the receive buffer, or null when the op receives nothing.
    ::mlir::Value getRecvBuffer() {
      return ::mlir::spmd::CommBufferOperands::valueOf(
          $_op.getCommBufferOperands().recvBuffer);
    }
    /// Root rank, or null for an op without one.
    ::mlir::Value getRoot() {
      return ::mlir::spmd::CommBufferOperands::valueOf(
          $_op.getCommBufferOperands().root);
    }
  }];
}

#endif // SPMD_INTERFACES
