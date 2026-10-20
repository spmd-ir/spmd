// RUN: env SPMD_CHECK_API_COVERAGE=1 spmd-opt %s -convert-apiCalls-to-spmd -o /dev/null 2>&1 | FileCheck %s --implicit-check-not=UNRESOLVED
//
// Every call name a CACTS map raises must have an entry in the model's USIS
// API description: the raiser looks a call up there first, and a call it
// cannot find is left alone without a word. SPMD_CHECK_API_COVERAGE=1 makes
// the pass check every name of every map against the api.json files the unit
// tests use, fail on a miss not on its list of known ones, and print a
// summary. This test is that check, so a refreshed api.json or a new map
// entry that do not agree fail here rather than in a benchmark.

// CHECK: [SPMD] coverage mpi: [[MPI:[0-9]+]]/[[MPI]] mapped names resolve
// CHECK: [SPMD] coverage shmem: {{[0-9]+}}/{{[0-9]+}} mapped names resolve
// CHECK: [SPMD] coverage nvshmem: {{[0-9]+}}/{{[0-9]+}} mapped names resolve
// CHECK: [SPMD] coverage nccl: [[NCCL:[0-9]+]]/[[NCCL]] mapped names resolve
// CHECK: [SPMD] coverage cuda: [[CUDA:[0-9]+]]/[[CUDA]] mapped names resolve
// CHECK: [SPMD] coverage omp: [[OMP:[0-9]+]]/[[OMP]] mapped names resolve
module {}
