// RUN: env SPMD_IR_EXITCODE=0 spmd-opt %s -split-input-file -omp-to-spmd 2>&1 | FileCheck %s
//
// OpenMP input -omp-to-spmd cannot raise fails the pass, so the module gets no
// verification rather than an unsound one. An op on the list of known
// unsupported ops, or a supported one with a clause it cannot model, says what
// it is before the conversion gives up.

// CHECK: warning: [OMPToSPMD] unsupported OpenMP op 'omp.flush'
// CHECK: error: [OMPToSPMD] conversion failed
llvm.func @flush(%x: !llvm.ptr) {
  omp.parallel {
    omp.flush(%x : !llvm.ptr)
    omp.terminator
  }
  llvm.return
}

// -----

// CHECK: warning: [OMPToSPMD] omp.single copyprivate clause not yet supported
// CHECK: error: [OMPToSPMD] conversion failed
omp.declare_reduction @copy : !llvm.ptr init {
^bb0(%arg0: !llvm.ptr):
  omp.yield(%arg0 : !llvm.ptr)
} combiner {
^bb0(%a: !llvm.ptr, %b: !llvm.ptr):
  omp.yield(%a : !llvm.ptr)
}
llvm.func @copy_func(%dst: !llvm.ptr, %src: !llvm.ptr)
llvm.func @copyprivate(%x: !llvm.ptr) {
  omp.parallel {
    omp.single copyprivate(%x -> @copy_func : !llvm.ptr) {
      omp.terminator
    }
    omp.terminator
  }
  llvm.return
}
