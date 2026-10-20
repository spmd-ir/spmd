// SPDX-FileCopyrightText: 2025 Semih Burak, RWTH Aachen University
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SPMD_TRAITS
#define SPMD_TRAITS

include "mlir/IR/Traits.td"

// Op results are not the same for each thread.
def ResultsNonUniform : NativeOpTrait<"ResultsNonUniform">{
  let cppNamespace = "::mlir::spmd";
}

def Collective : NativeOpTrait<"Collective">{
  let cppNamespace = "::mlir::spmd";
}

def P2P : NativeOpTrait<"P2P">{
  let cppNamespace = "::mlir::spmd";
}

def Persistent : NativeOpTrait<"Persistent">{
  let cppNamespace = "::mlir::spmd";
}

def ProcessManagement : NativeOpTrait<"ProcessManagement">{
  let cppNamespace = "::mlir::spmd";
}

def CommManagement : NativeOpTrait<"CommManagement">{
  let cppNamespace = "::mlir::spmd";
}

def ReqManagement : NativeOpTrait<"ReqManagement">{
  let cppNamespace = "::mlir::spmd";
}

def DataManagement : NativeOpTrait<"DataManagement">{
  let cppNamespace = "::mlir::spmd";
}

def NonBlockingSync : NativeOpTrait<"NonBlockingSync">{
  let cppNamespace = "::mlir::spmd";
}

def RMA : NativeOpTrait<"RMA">{
  let cppNamespace = "::mlir::spmd";
}

def Sync : NativeOpTrait<"Sync">{
  let cppNamespace = "::mlir::spmd";
}
def Communication : NativeOpTrait<"Communication">{
  let cppNamespace = "::mlir::spmd";
}

// Marks ops where multiple execution units (iterations, sections, tasks)
// may execute concurrently on different threads.
// Used by CheckThreadSafety to detect SERIALIZED violations structurally,
// without relying on ECA lattice modeling of each construct.
def GlobalConcurrency : NativeOpTrait<"GlobalConcurrency"> {
  let cppNamespace = "::mlir::spmd";
}

#endif // SPMD_TRAITS