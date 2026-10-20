// RUN: spmd-opt %s -test-shmem-lowering-table="canonical-only=true filter=shmem_int_" | FileCheck %s
// RUN: spmd-opt %s -test-shmem-lowering-table="canonical-only=true filter=team" | FileCheck %s --check-prefix=TEAM
// RUN: spmd-opt %s -test-shmem-lowering-table | FileCheck %s --check-prefix=ALIAS
//
// The OpenSHMEM calls -lower-spmd-to-shmem can emit, as it derives them from
// the CACTS map the raiser reads (lib/spmd/passes/CACTS/shmemSpmdEntries.cpp)
// and the USIS API description (includesProgramModels/shmem/api.json).  The
// format is -test-mpi-lowering-table's; "return" says which result the call's
// return value becomes, or which operand's storage it is written to.
//
// The whole table is one entry per spelling of every element type, which is
// not something to read.  The first dump is the calls the lowering picks for
// i32, the second the team calls; the third shows that the calls the lowering
// never picks are in the table, and why each of them is not picked.

module {}

// CHECK:      shmem_int_alltoall <- spmd.alltoall
// CHECK-NEXT:     team: in $0
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     nelems: in $2
// CHECK-NEXT:     $3 sendType: injected
// CHECK-NEXT:     $6 recvType: injected
// CHECK-NEXT:     $2 sendCount == $5 recvCount
// CHECK-NEXT: shmem_int_atomic_compare_swap <- spmd.compareAndSwap [isAtomic=true]
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     cond: in $1
// CHECK-NEXT:     value: in $0
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     return: stored to $2
// CHECK-NEXT:     $3 type: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT: shmem_int_atomic_compare_swap_nbi <- spmd.compareAndSwap [isAtomic=true]
// CHECK-NEXT:     fetch: in $2
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     cond: in $1
// CHECK-NEXT:     value: in $0
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $3 type: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT: shmem_int_atomic_fetch <- spmd.get [isAtomic=true]
// CHECK-NEXT:     source: in $5
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     return: stored to $0
// CHECK-NEXT:     $1 originCount: injected
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $7 targetCount: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: shmem_int_atomic_fetch_inc <- spmd.getAccumulate [isAtomic=true]
// CHECK-NEXT:     dest: in $9
// CHECK-NEXT:     pe: in $8
// CHECK-NEXT:     return: stored to $4
// CHECK-NEXT:     $0 originAddr: injected
// CHECK-NEXT:     $1 originCount: injected
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $5 resultCount: injected
// CHECK-NEXT:     $6 resultDatatype: injected
// CHECK-NEXT:     $7 resultStride: injected
// CHECK-NEXT:     $10 targetDisp: injected
// CHECK-NEXT:     $11 targetCount: injected
// CHECK-NEXT:     $12 targetDatatype: injected
// CHECK-NEXT:     $13 targetStride: injected
// CHECK-NEXT:     $14 accOp: injected
// CHECK-NEXT: shmem_int_atomic_fetch_inc_nbi <- spmd.getAccumulate [isAtomic=true]
// CHECK-NEXT:     fetch: in $4
// CHECK-NEXT:     dest: in $9
// CHECK-NEXT:     pe: in $8
// CHECK-NEXT:     $0 originAddr: injected
// CHECK-NEXT:     $1 originCount: injected
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $5 resultCount: injected
// CHECK-NEXT:     $6 resultDatatype: injected
// CHECK-NEXT:     $7 resultStride: injected
// CHECK-NEXT:     $10 targetDisp: injected
// CHECK-NEXT:     $11 targetCount: injected
// CHECK-NEXT:     $12 targetDatatype: injected
// CHECK-NEXT:     $13 targetStride: injected
// CHECK-NEXT:     $14 accOp: injected
// CHECK-NEXT: shmem_int_atomic_fetch_nbi <- spmd.get [isAtomic=true]
// CHECK-NEXT:     fetch: in $0
// CHECK-NEXT:     source: in $5
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $1 originCount: injected
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $7 targetCount: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: shmem_int_atomic_swap <- spmd.swap [isAtomic=true]
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     value: in $0
// CHECK-NEXT:     pe: in $3
// CHECK-NEXT:     return: stored to $1
// CHECK-NEXT:     $2 datatype: injected
// CHECK-NEXT:     $5 targetDisp: injected
// CHECK-NEXT: shmem_int_broadcast <- spmd.bcast
// CHECK-NEXT:     team: in $0
// CHECK-NEXT:     dest: in $2
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     nelems: in $3
// CHECK-NEXT:     PE_root: in $5
// CHECK-NEXT:     $4 dataType: injected
// CHECK-NEXT: shmem_int_fcollect <- spmd.allgather
// CHECK-NEXT:     team: in $0
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     nelems: in $2
// CHECK-NEXT:     $3 sendType: injected
// CHECK-NEXT:     $6 recvType: injected
// CHECK-NEXT:     $2 sendCount == $5 recvCount
// CHECK-NEXT: shmem_int_get <- spmd.get [isAtomic=false]
// CHECK-NEXT:     dest: in $0
// CHECK-NEXT:     source: in $5
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_get_nbi <- spmd.get [isAtomic=false]
// CHECK-NEXT:     dest: in $0
// CHECK-NEXT:     source: in $5
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_iget <- spmd.get [isAtomic=false]
// CHECK-NEXT:     dest: in $0
// CHECK-NEXT:     source: in $5
// CHECK-NEXT:     dst: in $9
// CHECK-NEXT:     sst: in $3
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_iput <- spmd.put [isAtomic=false]
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     source: in $0
// CHECK-NEXT:     dst: in $9
// CHECK-NEXT:     sst: in $3
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_max_reduce <- spmd.allreduce
// CHECK-NEXT:     team: in $0
// CHECK-NEXT:     dest: in $2
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     nreduce: in $3
// CHECK-NEXT:     $4 datatype: injected
// CHECK-NEXT:     $5 reduceOp: injected
// CHECK-NEXT: shmem_int_min_reduce <- spmd.allreduce
// CHECK-NEXT:     team: in $0
// CHECK-NEXT:     dest: in $2
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     nreduce: in $3
// CHECK-NEXT:     $4 datatype: injected
// CHECK-NEXT:     $5 reduceOp: injected
// CHECK-NEXT: shmem_int_prod_reduce <- spmd.allreduce
// CHECK-NEXT:     team: in $0
// CHECK-NEXT:     dest: in $2
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     nreduce: in $3
// CHECK-NEXT:     $4 datatype: injected
// CHECK-NEXT:     $5 reduceOp: injected
// CHECK-NEXT: shmem_int_put <- spmd.put [isAtomic=false]
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     source: in $0
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_put_nbi <- spmd.put [isAtomic=false]
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     source: in $0
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_put_signal <- spmd.putSignal [isAtomic=false]
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     source: in $0
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     sig_addr: in $10
// CHECK-NEXT:     signal: in $11
// CHECK-NEXT:     sig_op: in $12
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_put_signal_nbi <- spmd.putSignal [isAtomic=false]
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     source: in $0
// CHECK-NEXT:     nelems: in $1
// CHECK-NEXT:     sig_addr: in $10
// CHECK-NEXT:     signal: in $11
// CHECK-NEXT:     sig_op: in $12
// CHECK-NEXT:     pe: in $4
// CHECK-NEXT:     $2 originDatatype: injected
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $6 targetDisp: injected
// CHECK-NEXT:     $8 targetDatatype: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT:     $1 originCount == $7 targetCount
// CHECK-NEXT: shmem_int_sum_reduce <- spmd.allreduce
// CHECK-NEXT:     team: in $0
// CHECK-NEXT:     dest: in $2
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     nreduce: in $3
// CHECK-NEXT:     $4 datatype: injected
// CHECK-NEXT:     $5 reduceOp: injected

// TEAM:      shmem_team_destroy <- spmd.commDestroy
// TEAM-NEXT:     team: in $0
// TEAM-NEXT: shmem_team_my_pe <- spmd.getRankInComm
// TEAM-NEXT:     team: in $0
// TEAM-NEXT:     return: result #0
// TEAM-NEXT: shmem_team_n_pes <- spmd.getSizeOfComm
// TEAM-NEXT:     team: in $0
// TEAM-NEXT:     return: result #0
// TEAM-NEXT: shmem_team_split_strided <- spmd.commSplitStrided
// TEAM-NEXT:     parent_team: in $0
// TEAM-NEXT:     start: in $1
// TEAM-NEXT:     stride: in $2
// TEAM-NEXT:     size: in $3
// TEAM-NEXT:     config: NULL
// TEAM-NEXT:     config_mask: 0L
// TEAM-NEXT:     new_team: outTeam
// TEAM-NEXT: shmem_team_sync <- spmd.barrier
// TEAM-NEXT:     team: in $0

// Other spellings of one transfer of si32: the lowering picks shmem_int_put.
// ALIAS-DAG: {{^}}shmem_int_put <- spmd.put [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_int32_put <- spmd.put [alias] [isAtomic=false]
// ui32 is a type of its own, and so is the i32 of unknown signedness, which only
// the sized form moves.  8 bits of unknown signedness are char's.
// ALIAS-DAG: {{^}}shmem_uint_put <- spmd.put [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_uint32_put <- spmd.put [alias] [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_put32 <- spmd.put [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_put32_nbi <- spmd.put [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_char_put <- spmd.put [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_put8 <- spmd.put [alias] [isAtomic=false]
// 128 bits has no type name, so its sized form is the one.
// ALIAS-DAG: {{^}}shmem_put128 <- spmd.put [isAtomic=false]
// The element by value, which the dialect keeps as an address.
// ALIAS-DAG: {{^}}shmem_int_p <- spmd.put [alias] [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_int_atomic_set <- spmd.put [alias] [isAtomic=true]
// A single element of shmem_int_get, the variable-length allgather, and the
// non-blocking swap spmd.swap cannot be told from the blocking one.
// ALIAS-DAG: {{^}}shmem_int_g <- spmd.get [alias] [isAtomic=false]
// ALIAS-DAG: {{^}}shmem_int_collect <- spmd.allgather [alias]
// ALIAS-DAG: {{^}}shmem_int_atomic_swap_nbi <- spmd.swap [alias] [isAtomic=true]
// No complex type in the dialect.
// ALIAS-DAG: {{^}}shmem_complexd_sum_reduce <- spmd.allreduce [alias]
// Emitted by patterns of their own, which ignore the window.
// ALIAS-DAG: {{^}}shmem_barrier_all <- spmd.fence [alias]
// ALIAS-DAG: {{^}}shmem_quiet <- spmd.flushAll [alias] [isLocal=false]
