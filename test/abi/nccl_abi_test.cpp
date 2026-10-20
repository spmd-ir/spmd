// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- nccl_abi_test.cpp - NCCL.h against the vendored nccl.h -------------===//
//
// include/spmd/TargetABI/NCCL.h copies NCCL's enumerators by hand, so that the
// passes never see nccl.h.  This test compiles the table against
// includesProgramModels/nccl/include/nccl.h and checks every entry against the
// enumerator it copies, so a value that drifts fails the build.  It also pins
// the ABI facts the lowering rests on: the handles are pointers, the enums are
// ints, and ncclCommInitRank takes the unique id by value.
//
// nccl.h includes CUDA's runtime header for cudaStream_t.  The test compiles
// it against nccl-cuda-stub, which declares that one type as CUDA 13.1 does,
// rather than against a CUDA install; that the stub still matches CUDA is the
// one thing this test cannot check.
//
//===----------------------------------------------------------------------===//

// The table first, as in mpich_abi_test.cpp: MLIR's headers before the vendor
// header, so that nothing of the latter can rewrite them.
#include "spmd/TargetABI/NCCL.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <type_traits>

#include "nccl.h"

namespace nccl = mlir::spmd::abi::nccl;

namespace {

// Not a value NCCL uses, so a name missing from the table never compares
// equal to an enumerator.
constexpr int64_t kMissing = INT64_MIN;

constexpr bool sameName(llvm::StringLiteral name, std::string_view expected) {
  return std::string_view(name.data(), name.size()) == expected;
}

template <std::size_t N>
constexpr int64_t handleNamed(const nccl::NamedHandle (&table)[N],
                              std::string_view name) {
  for (const nccl::NamedHandle &entry : table)
    if (sameName(entry.name, name))
      return entry.handle;
  return kMissing;
}

constexpr int64_t datatypeNamed(std::string_view name) {
  for (const nccl::DatatypeHandle &entry : nccl::kDatatypes)
    if (sameName(entry.name, name))
      return entry.handle;
  return kMissing;
}

#define CHECK_REDUCE_OP(NAME, ENUMERATOR)                                      \
  static_assert(handleNamed(nccl::kReduceOps, NAME) == (ENUMERATOR), #ENUMERATOR)
#define CHECK_DATATYPE(ENUMERATOR)                                             \
  static_assert(datatypeNamed(#ENUMERATOR) == (ENUMERATOR), #ENUMERATOR)

// Adding a table entry without checking it here fails too.
static_assert(std::size(nccl::kReduceOps) == 5);
static_assert(std::size(nccl::kDatatypes) == 17);

// Every built-in reduction and every datatype NCCL has is in the table.
static_assert(std::size(nccl::kReduceOps) == ncclNumOps);
static_assert(ncclFloat8e5m2 + 1 == ncclNumTypes);

CHECK_REDUCE_OP("SUM", ncclSum);
CHECK_REDUCE_OP("PROD", ncclProd);
CHECK_REDUCE_OP("MAX", ncclMax);
CHECK_REDUCE_OP("MIN", ncclMin);
CHECK_REDUCE_OP("AVG", ncclAvg);

CHECK_DATATYPE(ncclInt8);
CHECK_DATATYPE(ncclChar);
CHECK_DATATYPE(ncclUint8);
CHECK_DATATYPE(ncclInt32);
CHECK_DATATYPE(ncclInt);
CHECK_DATATYPE(ncclUint32);
CHECK_DATATYPE(ncclInt64);
CHECK_DATATYPE(ncclUint64);
CHECK_DATATYPE(ncclFloat16);
CHECK_DATATYPE(ncclHalf);
CHECK_DATATYPE(ncclFloat32);
CHECK_DATATYPE(ncclFloat);
CHECK_DATATYPE(ncclFloat64);
CHECK_DATATYPE(ncclDouble);
CHECK_DATATYPE(ncclBfloat16);
CHECK_DATATYPE(ncclFloat8e4m3);
CHECK_DATATYPE(ncclFloat8e5m2);

static_assert(nccl::kSuccess == ncclSuccess);

// The enums are ints, which is why the lowering passes a datatype, a reduction
// and a status as i32.
static_assert(sizeof(ncclDataType_t) == sizeof(int32_t));
static_assert(sizeof(ncclRedOp_t) == sizeof(int32_t));
static_assert(sizeof(ncclResult_t) == sizeof(int32_t));

// A communicator and a stream are pointers, which is why the lowering converts
// !spmd.comm and !spmd.stream to !llvm.ptr rather than to an integer.
static_assert(std::is_pointer_v<ncclComm_t>);
static_assert(std::is_pointer_v<cudaStream_t>);

// The unique id is a struct of chars, which the bootstrap stores in a global
// of that many bytes and passes by value.
static_assert(nccl::kUniqueIdBytes == NCCL_UNIQUE_ID_BYTES);
static_assert(sizeof(ncclUniqueId) == nccl::kUniqueIdBytes);
static_assert(alignof(ncclUniqueId) == 1);

// The calls the bootstrap emits itself, rather than deriving them from the
// CACTS map: ncclCommInitRank takes the id by value, not by address.
static_assert(std::is_same_v<decltype(&ncclGetUniqueId),
                             ncclResult_t (*)(ncclUniqueId *)>);
static_assert(std::is_same_v<decltype(&ncclCommInitRank),
                             ncclResult_t (*)(ncclComm_t *, int, ncclUniqueId,
                                              int)>);
static_assert(std::is_same_v<decltype(&ncclGroupStart), ncclResult_t (*)()>);
static_assert(std::is_same_v<decltype(&ncclGroupEnd), ncclResult_t (*)()>);

} // namespace

// Nothing here needs the runtime: every check is a static_assert, so the test
// passing is the test having compiled.
int main() { return 0; }
