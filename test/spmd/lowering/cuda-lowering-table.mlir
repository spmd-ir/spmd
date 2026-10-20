// RUN: spmd-opt %s -test-cuda-lowering-table | FileCheck %s
//
// Every CUDA call -lower-spmd-to-cuda can emit, as it derives it from the
// CACTS map the raiser reads (lib/spmd/passes/CACTS/cudaSpmdEntries.cpp) and
// the USIS API description (includesProgramModels/cuda/api.json).
//
// This dump is where the scope of the CUDA lowering is visible: CUDA's runtime
// API has some 330 routines and the raiser maps two of them, so two is what
// there is to lower.  Every other CUDA call a program makes is still the
// llvm.call the frontend produced and needs no lowering.  A routine added to
// the map appears here on its own, which is what makes this test the place the
// scope changes show up.

// CHECK:      cudaMemcpy <- spmd.memcpy
// CHECK-NEXT:     dst: in $0
// CHECK-NEXT:     src: in $1
// CHECK-NEXT:     count: in $2
// CHECK-NEXT:     kind: in $3
// CHECK-NEXT: cudaMemset <- spmd.memset
// CHECK-NEXT:     devPtr: in $0
// CHECK-NEXT:     value: in $1
// CHECK-NEXT:     count: in $2
// CHECK-NOT:  <-
module {
}
