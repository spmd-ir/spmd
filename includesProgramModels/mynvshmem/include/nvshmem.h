#ifndef _NVSHMEM_H_
#define _NVSHMEM_H_

#include <cuda_runtime.h>
#include "nvshmem_common.cuh"

#ifdef __cplusplus
extern "C" {
#endif

#define NOINLINE_NOOPT __attribute__((noinline, optnone))

// The empty bodies below stand in for NVSHMEM's device routines, for a GPU
// kernel compiled against these headers.  Only a CUDA compile may see them: a
// host compile would take each for the definition of the host routine of the
// same name -- a strong one in C, a linkonce_odr one in C++ -- and a call of
// that routine would get the empty body instead of the library's.
#if defined(__CUDACC__)
#define NVSHMEMI_DEVICE_STUB(...) __VA_ARGS__
#else
#define NVSHMEMI_DEVICE_STUB(...)
#endif

// ============================================================================
// 1. HOST-ONLY LIBRARY MANAGEMENT
// ============================================================================

__host__ void NOINLINE_NOOPT nvshmem_init(void);
__host__ int NOINLINE_NOOPT nvshmem_init_thread(int requested, int *provided);
__host__ void NOINLINE_NOOPT nvshmem_finalize(void);
__host__ void NOINLINE_NOOPT nvshmem_query_thread(int *provided);

__host__ void NOINLINE_NOOPT nvshmem_team_get_config(nvshmem_team_t team, nvshmem_team_config_t *config);
__host__ int NOINLINE_NOOPT nvshmem_team_split_strided(nvshmem_team_t parent_team, int PE_start, int PE_stride, int PE_size,
                                                       const nvshmem_team_config_t *config, long config_mask,
                                                       nvshmem_team_t *new_team);
__host__ int NOINLINE_NOOPT nvshmem_team_split_2d(nvshmem_team_t parent_team, int xrange,
                                                  const nvshmem_team_config_t *xaxis_config, long xaxis_mask,
                                                  nvshmem_team_t *xaxis_team, const nvshmem_team_config_t *yaxis_config,
                                                  long yaxis_mask, nvshmem_team_t *yaxis_team);
__host__ void NOINLINE_NOOPT nvshmem_team_destroy(nvshmem_team_t team);

__host__ void * NOINLINE_NOOPT nvshmem_malloc(size_t size);
__host__ void * NOINLINE_NOOPT nvshmem_calloc(size_t, size_t); 
__host__ void * NOINLINE_NOOPT nvshmem_align(size_t, size_t);
__host__ void NOINLINE_NOOPT nvshmem_free(void *ptr);
__host__ void * NOINLINE_NOOPT nvshmem_realloc(void *ptr, size_t size);


// ============================================================================
// 2. EXPLICIT PAIRED PAIRS (HOST DECLARATION & DEVICE DEFINITION)
// ============================================================================

__host__ void NOINLINE_NOOPT nvshmem_global_exit(int status);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_global_exit(int status) { return; })

//////////////////// Teams API ////////////////////

__host__ int NOINLINE_NOOPT nvshmem_team_my_pe(nvshmem_team_t team);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_team_my_pe(nvshmem_team_t team) { return 0; })

__host__ int NOINLINE_NOOPT nvshmem_team_n_pes(nvshmem_team_t team);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_team_n_pes(nvshmem_team_t team) { return 0; })

__host__ int NOINLINE_NOOPT nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_team_translate_pe(nvshmem_team_t src_team, int src_pe, nvshmem_team_t dest_team) { return 0; })

//////////////////// PE Info Queries ////////////////////

__host__ int NOINLINE_NOOPT nvshmem_my_pe(void);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_my_pe(void) { return 0; })

__host__ int NOINLINE_NOOPT nvshmem_n_pes(void);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_n_pes(void) { return 0; })

__host__ void NOINLINE_NOOPT nvshmem_info_get_version(int *major, int *minor);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_info_get_version(int *major, int *minor) { return; })

__host__ void NOINLINE_NOOPT nvshmem_info_get_name(char *name);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_info_get_name(char *name) { return; })

__host__ void NOINLINE_NOOPT nvshmemx_vendor_get_version_info(int *major, int *minor, int *patch);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmemx_vendor_get_version_info(int *major, int *minor, int *patch) { return; })

//////////////////// Point-to-Point Synchronization ////////////////////

__host__ void NOINLINE_NOOPT nvshmem_quiet(void);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_quiet(void) { return; })

__host__ void NOINLINE_NOOPT nvshmem_fence(void);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_fence(void) { return; })

//////////////////// Collective Operations ////////////////////

__host__ int NOINLINE_NOOPT nvshmem_barrier(nvshmem_team_t team);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_barrier(nvshmem_team_t team) { return 0; })

__host__ void NOINLINE_NOOPT nvshmem_barrier_all(void);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_barrier_all(void) { return; })

__host__ int NOINLINE_NOOPT nvshmem_team_sync(nvshmem_team_t team);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_team_sync(nvshmem_team_t team) { return 0; })

__host__ void NOINLINE_NOOPT nvshmem_sync_all(void);
NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_sync_all(void) { return; })

__host__ int NOINLINE_NOOPT nvshmem_broadcastmem(nvshmem_team_t team, void *dest, const void *src, size_t nelems, int PE_root);
NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_broadcastmem(nvshmem_team_t team, void *dest, const void *src, size_t nelems, int PE_root) { return 0; })


// ============================================================================
// 3. MACRO-EXPANDED PAIRED PAIRS (DECLARATION & INLINE DEFINITION CONSECUTIVE)
// ============================================================================

// Broadcast Collectives
#define DECL_NVSHMEM_TYPENAME_BROADCAST(TYPENAME, TYPE)            \
    __host__ int NOINLINE_NOOPT nvshmem_##TYPENAME##_broadcast(   \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nelem, int PE_root); \
    NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_##TYPENAME##_broadcast( \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nelem, int PE_root) { return 0; })
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(DECL_NVSHMEM_TYPENAME_BROADCAST)
#undef DECL_NVSHMEM_TYPENAME_BROADCAST

// Reduction Collectives
#define NVSHMEMI_DECL_TEAM_REDUCE(NAME, TYPE, OP)                  \
    __host__ int NOINLINE_NOOPT nvshmem_##NAME##_##OP##_reduce(   \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce); \
    NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_##NAME##_##OP##_reduce( \
        nvshmem_team_t team, TYPE *dest, const TYPE *src, size_t nreduce) { return 0; })

NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, and)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, or)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, xor)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, max)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, min)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, sum)
NVSHMEMI_REPT_FOR_STANDARD_REDUCE_TYPES(NVSHMEMI_DECL_TEAM_REDUCE, prod)
#undef NVSHMEMI_DECL_TEAM_REDUCE

//////////////////// Put API ////////////////////

#define NVSHMEMI_DECL_TYPE_PUT(NAME, TYPE)                                \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_put(TYPE *dest, const TYPE *source, size_t nelems, int pe) { return; })
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_PUT)
#undef NVSHMEMI_DECL_TYPE_PUT

#define NVSHMEMI_DECL_TYPE_PUT_NBI(NAME, TYPE)                                               \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_put_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) { return; })
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_PUT_NBI)
#undef NVSHMEMI_DECL_TYPE_PUT_NBI

#define NVSHMEMI_DECL_TYPE_PUT_SIGNAL(NAME, TYPE)                                               \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_put_signal(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_put_signal(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { return; })
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_PUT_SIGNAL)
#undef NVSHMEMI_DECL_TYPE_PUT_SIGNAL

#define NVSHMEMI_DECL_TYPE_PUT_SIGNAL_NBI(NAME, TYPE)                                               \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_put_signal_nbi(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_put_signal_nbi(TYPE *dest, const TYPE *source, size_t nelems, uint64_t *sig_addr, uint64_t signal, int sig_op, int pe) { return; })
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_PUT_SIGNAL_NBI)
#undef NVSHMEMI_DECL_TYPE_PUT_SIGNAL_NBI

//////////////////// Get API ////////////////////

#define NVSHMEMI_DECL_TYPE_GET(NAME, TYPE)                                \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_get(TYPE *dest, const TYPE *source, size_t nelems, int pe) { return; })
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_GET)
#undef NVSHMEMI_DECL_TYPE_GET

#define NVSHMEMI_DECL_TYPE_GET_NBI(NAME, TYPE)                                               \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_get_nbi(TYPE *dest, const TYPE *source, size_t nelems, int pe) { return; })
NVSHMEMI_REPT_FOR_STANDARD_RMA_TYPES(NVSHMEMI_DECL_TYPE_GET_NBI)
#undef NVSHMEMI_DECL_TYPE_GET_NBI

//////////////////// Atomics (AMO) ////////////////////

#define NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(OPGRPNAME, opname)                  \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint, unsigned int, opname)                    \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulong, unsigned long, opname)              \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ulonglong, unsigned long long, opname) \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int32, int32_t, opname)            \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint32, uint32_t, opname)      \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int64, int64_t, opname)    \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(uint64, uint64_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(OPGRPNAME, opname)       \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(int, int, opname)                    \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(long, long, opname)              \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(longlong, long long, opname) \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(size, size_t, opname)    \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(ptrdiff, ptrdiff_t, opname)

#define NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(OPGRPNAME, opname) \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(float, float, opname)          \
    NVSHMEMI_DECL_TYPE_##OPGRPNAME(double, double, opname)

/* inc */
#define NVSHMEMI_DECL_TYPE_INC(type, TYPE, opname) \
    __host__ void NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, int pe) { return; })

/* finc */
#define NVSHMEMI_DECL_TYPE_FINC(type, TYPE, opname) \
    __host__ TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, int pe) { return (TYPE)0; })

/* fetch */
#define NVSHMEMI_DECL_TYPE_FETCH(type, TYPE, opname) \
    __host__ TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(const TYPE *dest, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(const TYPE *dest, int pe) { return (TYPE)0; })

/* add, set */
#define NVSHMEMI_DECL_TYPE_ADD_SET(type, TYPE, opname)                                       \
    __host__ void NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe) { return; })

/* fadd, swap */
#define NVSHMEMI_DECL_TYPE_FADD_SWAP(type, TYPE, opname)                                     \
    __host__ TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe) { return (TYPE)0; })

/* cswap */
#define NVSHMEMI_DECL_TYPE_CSWAP(type, TYPE, opname)                                        \
    __host__ TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE cond, TYPE value, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE cond, TYPE value, int pe) { return (TYPE)0; })

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(INC, inc)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(INC, inc)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FINC, fetch_inc)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FINC, fetch_inc)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FETCH, fetch)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FETCH, fetch)
NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(FETCH, fetch)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(ADD_SET, add)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(ADD_SET, add)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(ADD_SET, set)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(ADD_SET, set)
NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(ADD_SET, set)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FADD_SWAP, fetch_add)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FADD_SWAP, fetch_add)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FADD_SWAP, swap)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(FADD_SWAP, swap)
NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO(FADD_SWAP, swap)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(CSWAP, compare_swap)
NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO(CSWAP, compare_swap)

#undef NVSHMEMI_DECL_TYPE_INC
#undef NVSHMEMI_DECL_TYPE_FINC
#undef NVSHMEMI_DECL_TYPE_FETCH
#undef NVSHMEMI_DECL_TYPE_ADD_SET
#undef NVSHMEMI_DECL_TYPE_FADD_SWAP
#undef NVSHMEMI_DECL_TYPE_CSWAP

/* OpenSHMEM 1.4 Atomics */
#define NVSHMEMI_DECL_TYPE_AND_OR_XOR(type, TYPE, opname)                                    \
    __host__ void NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##type##_atomic_##opname(TYPE *dest, TYPE value, int pe) { return; })

#define NVSHMEMI_DECL_TYPE_FAND_FOR_FXOR(type, TYPE, opname)                                       \
    __host__ TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_fetch_##opname(TYPE *dest, TYPE value, int pe); \
    NVSHMEMI_DEVICE_STUB(__device__ inline TYPE NOINLINE_NOOPT nvshmem_##type##_atomic_fetch_##opname(TYPE *dest, TYPE value, int pe) { return (TYPE)0; })

NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, and)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, or)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(AND_OR_XOR, xor)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, and)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, or)
NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO(FAND_FOR_FXOR, xor)

#undef NVSHMEMI_DECL_TYPE_AND_OR_XOR
#undef NVSHMEMI_DECL_TYPE_FAND_FOR_FXOR
#undef NVSHMEMI_REPT_OPGROUP_FOR_BITWISE_AMO
#undef NVSHMEMI_REPT_OPGROUP_FOR_STANDARD_AMO
#undef NVSHMEMI_REPT_OPGROUP_FOR_EXTENDED_AMO

//////////////////// Wait and Test API ////////////////////

#define NVSHMEMI_DECL_WAIT_UNTIL(NAME, TYPE) \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_wait_until(TYPE *ivar, int cmp, TYPE cmp_value) { return; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL)
#undef NVSHMEMI_DECL_WAIT_UNTIL

#define NVSHMEMI_DECL_WAIT_UNTIL_ALL(NAME, TYPE)                                                  \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_wait_until_all(TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_wait_until_all(TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value) { return; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ALL)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ALL

#define NVSHMEMI_DECL_WAIT_UNTIL_ANY(NAME, TYPE)                                 \
    __host__ size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_any(TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_any(TYPE *ivar, size_t nelems, const int *status, int cmp, TYPE cmp_value) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ANY)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ANY

#define NVSHMEMI_DECL_WAIT_UNTIL_SOME(NAME, TYPE)       \
    __host__ size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_some(TYPE *ivar, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_some(TYPE *ivar, size_t nelems, size_t *indices, const int *status, int cmp, TYPE cmp_value) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_SOME)
#undef NVSHMEMI_DECL_WAIT_UNTIL_SOME

#define NVSHMEMI_DECL_WAIT_UNTIL_ALL_VECTOR(NAME, TYPE)     \
    __host__ void NOINLINE_NOOPT nvshmem_##NAME##_wait_until_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values); \
    NVSHMEMI_DEVICE_STUB(__device__ inline void NOINLINE_NOOPT nvshmem_##NAME##_wait_until_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values) { return; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ALL_VECTOR)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ALL_VECTOR

#define NVSHMEMI_DECL_WAIT_UNTIL_ANY_VECTOR(NAME, TYPE)       \
    __host__ size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_ANY_VECTOR)
#undef NVSHMEMI_DECL_WAIT_UNTIL_ANY_VECTOR

#define NVSHMEMI_DECL_WAIT_UNTIL_SOME_VECTOR(NAME, TYPE)                                          \
    __host__ size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##NAME##_wait_until_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_WAIT_UNTIL_SOME_VECTOR)
#undef NVSHMEMI_DECL_WAIT_UNTIL_SOME_VECTOR

#define NVSHMEMI_DECL_TEST(NAME, TYPE) \
    __host__ int NOINLINE_NOOPT nvshmem_##NAME##_test(TYPE *ivar, int cmp, TYPE cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_##NAME##_test(TYPE *ivar, int cmp, TYPE cmp_value) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST)
#undef NVSHMEMI_DECL_TEST

#define NVSHMEMI_DECL_TEST_ALL(Name, Type)                                                  \
    __host__ int NOINLINE_NOOPT nvshmem_##Name##_test_all(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_##Name##_test_all(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ALL)
#undef NVSHMEMI_DECL_TEST_ALL

#define NVSHMEMI_DECL_TEST_ANY(Name, Type)                                                     \
    __host__ size_t NOINLINE_NOOPT nvshmem_##Name##_test_any(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##Name##_test_any(Type *ivars, size_t nelems, const int *status, int cmp, Type cmp_value) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ANY)
#undef NVSHMEMI_DECL_TEST_ANY

#define NVSHMEMI_DECL_TEST_SOME(Name, Type)                                                 \
    __host__ size_t NOINLINE_NOOPT nvshmem_##Name##_test_some(Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##Name##_test_some(Type *ivars, size_t nelems, size_t *indices, const int *status, int cmp, Type cmp_value) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_SOME)
#undef NVSHMEMI_DECL_TEST_SOME

#define NVSHMEMI_DECL_TEST_ALL_VECTOR(NAME, TYPE)                                                  \
    __host__ int NOINLINE_NOOPT nvshmem_##NAME##_test_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values); \
    NVSHMEMI_DEVICE_STUB(__device__ inline int NOINLINE_NOOPT nvshmem_##NAME##_test_all_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ALL_VECTOR)
#undef NVSHMEMI_DECL_TEST_ALL_VECTOR

#define NVSHMEMI_DECL_TEST_ANY_VECTOR(NAME, TYPE)       \
    __host__ size_t NOINLINE_NOOPT nvshmem_##NAME##_test_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##NAME##_test_any_vector(TYPE *ivars, size_t nelems, const int *status, int cmp, TYPE *cmp_values) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_ANY_VECTOR)
#undef NVSHMEMI_DECL_TEST_ANY_VECTOR

#define NVSHMEMI_DECL_TEST_SOME_VECTOR(NAME, TYPE)                                          \
    __host__ size_t NOINLINE_NOOPT nvshmem_##NAME##_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values); \
    NVSHMEMI_DEVICE_STUB(__device__ inline size_t NOINLINE_NOOPT nvshmem_##NAME##_test_some_vector(TYPE *ivars, size_t nelems, size_t *indices, const int *status, int cmp, TYPE *cmp_values) { return 0; })
NVSHMEMI_REPT_FOR_WAIT_TYPES(NVSHMEMI_DECL_TEST_SOME_VECTOR)
#undef NVSHMEMI_DECL_TEST_SOME_VECTOR

#ifdef __cplusplus
}
#endif

#endif // _NVSHMEM_H_