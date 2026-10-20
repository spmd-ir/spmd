// RUN: spmd-opt %s -test-nvshmem-lowering-table="canonical-only=true filter=team" | FileCheck %s --check-prefix=TEAM
// RUN: spmd-opt %s -test-nvshmem-lowering-table="canonical-only=true filter=init" | FileCheck %s --check-prefix=INIT
// RUN: spmd-opt %s -test-nvshmem-lowering-table="canonical-only=true filter=nvshmem_int_" | FileCheck %s --check-prefix=INT
// RUN: spmd-opt %s -test-nvshmem-lowering-table="filter=_and_reduce" | FileCheck %s --check-prefix=AND
//
// The NVSHMEM calls -lower-spmd-to-nvshmem can emit, as it derives them from
// the CACTS map the raiser reads (lib/spmd/passes/CACTS/nvshmemSpmdEntries.cpp)
// and the USIS API description (includesProgramModels/mynvshmem/api.json), in
// -test-shmem-lowering-table's format.  The table is OpenSHMEM's with an
// nvshmem_ prefix; the dumps show what differs.

module {}

// A team is an int32_t, passed in and written back (outTeam) as one.  No team
// configuration is passed: a null pointer and an empty mask.
// TEAM:      nvshmem_team_destroy <- spmd.commDestroy
// TEAM-NEXT:     team: in $0
// TEAM-NEXT: nvshmem_team_my_pe <- spmd.getRankInComm
// TEAM-NEXT:     team: in $0
// TEAM-NEXT:     return: result #0
// TEAM-NEXT: nvshmem_team_n_pes <- spmd.getSizeOfComm
// TEAM-NEXT:     team: in $0
// TEAM-NEXT:     return: result #0
// TEAM-NEXT: nvshmem_team_split_strided <- spmd.commSplitStrided
// TEAM-NEXT:     parent_team: in $0
// TEAM-NEXT:     PE_start: in $1
// TEAM-NEXT:     PE_stride: in $2
// TEAM-NEXT:     PE_size: in $3
// TEAM-NEXT:     config: NULL
// TEAM-NEXT:     config_mask: 0L
// TEAM-NEXT:     new_team: outTeam
// TEAM-NEXT: nvshmem_team_sync <- spmd.barrier
// TEAM-NEXT:     team: in $0

// The calls the raiser reads nvshmem_init and nvshmem_init_thread as.  They
// are not symbols of NVIDIA's library; -nvshmem-to-host-abi maps them.
// INIT:      nvshmem_init <- spmd.init
// INIT-NEXT:     $0 threadRequired: injected
// INIT-NEXT: nvshmem_init_thread <- spmd.init
// INIT-NEXT:     requested: in $0
// INIT-NEXT:     provided: outInt (optional)

// The calls picked for si32.  nvshmem_int_p and nvshmem_int_g are not: _p takes
// the element by value, and _g is _get of one element.  The atomic fetch-add,
// fetch-and and the other fetching atomics that take an operand by value are
// not in the table at all, since the raiser keeps no operand for it.
// INT:      nvshmem_int_alltoall <- spmd.alltoall
// INT-NEXT:     team: in $0
// INT:      nvshmem_int_atomic_compare_swap <- spmd.compareAndSwap [isAtomic=true]
// INT:      nvshmem_int_atomic_fetch <- spmd.get [isAtomic=true]
// INT:      nvshmem_int_atomic_fetch_inc <- spmd.getAccumulate [isAtomic=true]
// INT:      nvshmem_int_broadcast <- spmd.bcast
// INT:      nvshmem_int_fcollect <- spmd.allgather
// INT:      nvshmem_int_get <- spmd.get [isAtomic=false]
// INT:      nvshmem_int_get_nbi <- spmd.get [isAtomic=false]
// INT:      nvshmem_int_iget <- spmd.get [isAtomic=false]
// INT:      nvshmem_int_iput <- spmd.put [isAtomic=false]
// INT:      nvshmem_int_max_reduce <- spmd.allreduce
// INT:      nvshmem_int_min_reduce <- spmd.allreduce
// INT:      nvshmem_int_prod_reduce <- spmd.allreduce
// INT:      nvshmem_int_put <- spmd.put [isAtomic=false]
// INT-NEXT:     dest: in $5
// INT-NEXT:     source: in $0
// INT-NEXT:     nelems: in $1
// INT-NEXT:     pe: in $4
// INT:      nvshmem_int_put_nbi <- spmd.put [isAtomic=false]
// INT:      nvshmem_int_sum_reduce <- spmd.allreduce
// INT-NOT:  nvshmem_int_p <-
// INT-NOT:  nvshmem_int_g <-

// NVSHMEM has the bitwise reductions for the unsigned and the fixed-width
// names only, so si32's is nvshmem_int32_and_reduce; of the unsigned names,
// the lowering prefers uint to uint32 and the rest are aliases.
// AND:      nvshmem_int16_and_reduce <- spmd.allreduce
// AND:      nvshmem_int32_and_reduce <- spmd.allreduce
// AND:      nvshmem_int64_and_reduce <- spmd.allreduce
// AND:      nvshmem_int8_and_reduce <- spmd.allreduce
// AND:      nvshmem_size_and_reduce <- spmd.allreduce [alias]
// AND:      nvshmem_uchar_and_reduce <- spmd.allreduce
// AND:      nvshmem_uint16_and_reduce <- spmd.allreduce [alias]
// AND:      nvshmem_uint32_and_reduce <- spmd.allreduce [alias]
// AND:      nvshmem_uint64_and_reduce <- spmd.allreduce [alias]
// AND:      nvshmem_uint8_and_reduce <- spmd.allreduce [alias]
// AND:      nvshmem_uint_and_reduce <- spmd.allreduce
// AND:      nvshmem_ulong_and_reduce <- spmd.allreduce
// AND:      nvshmem_ulonglong_and_reduce <- spmd.allreduce [alias]
// AND:      nvshmem_ushort_and_reduce <- spmd.allreduce
