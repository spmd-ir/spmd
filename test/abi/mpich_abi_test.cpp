// SPDX-FileCopyrightText: 2026 Semih Burak, RWTH Aachen University
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- mpich_abi_test.cpp - MPICH.h against the vendored mpi.h ------------===//
//
// include/spmd/TargetABI/MPICH.h copies MPICH's predefined handles by hand, so
// that the passes never see mpi.h's macros.  This test compiles the table
// against includesProgramModels/mpi/include/mpi.h and checks every entry
// against the macro it copies, so a value that drifts fails the build.  The
// status-ignore pointers, which no constant expression can hold, are checked
// when the test runs.
//
//===----------------------------------------------------------------------===//

// The table first: once mpi.h is in, its macros would rewrite MLIR's headers.
#include "spmd/TargetABI/MPICH.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string_view>

#define MPICH_SKIP_MPICXX
#include "mpi.h"

namespace mpich = mlir::spmd::abi::mpich;

namespace {

// Not a handle MPICH uses, so a name missing from the table never compares
// equal to a macro.
constexpr int64_t kMissing = INT64_MIN;

constexpr bool sameName(llvm::StringLiteral name, std::string_view expected) {
  return std::string_view(name.data(), name.size()) == expected;
}

template <std::size_t N>
constexpr int64_t handleNamed(const mpich::NamedHandle (&table)[N],
                              std::string_view name) {
  for (const mpich::NamedHandle &entry : table)
    if (sameName(entry.name, name))
      return entry.handle;
  return kMissing;
}

constexpr int64_t datatypeNamed(std::string_view name) {
  for (const mpich::DatatypeHandle &entry : mpich::kDatatypes)
    if (sameName(entry.name, name))
      return entry.handle;
  return kMissing;
}

// Every handle is an int in this ABI; the two above INT_MAX wrap, as they do
// in the table.
#define CHECK_HANDLE(VALUE, MACRO)                                             \
  static_assert((VALUE) == static_cast<int32_t>(MACRO), #MACRO)
#define CHECK_NAMED(TABLE, NAME, MACRO)                                        \
  CHECK_HANDLE(handleNamed(mpich::TABLE, NAME), MACRO)
#define CHECK_DATATYPE(MACRO) CHECK_HANDLE(datatypeNamed(#MACRO), MACRO)

// Adding a table entry without checking it here fails too.
static_assert(std::size(mpich::kReduceOps) == 15);
static_assert(std::size(mpich::kLockTypes) == 2);
static_assert(std::size(mpich::kThreadLevels) == 4);
static_assert(std::size(mpich::kDatatypes) == 20);

CHECK_HANDLE(mpich::kCommWorld, MPI_COMM_WORLD);
CHECK_HANDLE(mpich::kCommSelf, MPI_COMM_SELF);
CHECK_HANDLE(mpich::kRequestNull, MPI_REQUEST_NULL);
CHECK_HANDLE(mpich::kInfoNull, MPI_INFO_NULL);

static_assert(sizeof(MPI_Status) == mpich::kStatusNumFields * sizeof(int));

CHECK_HANDLE(mpich::kLockExclusive, MPI_LOCK_EXCLUSIVE);
CHECK_HANDLE(mpich::kLockShared, MPI_LOCK_SHARED);
CHECK_NAMED(kLockTypes, "EXCLUSIVE", MPI_LOCK_EXCLUSIVE);
CHECK_NAMED(kLockTypes, "SHARED", MPI_LOCK_SHARED);

CHECK_NAMED(kThreadLevels, "SINGLE", MPI_THREAD_SINGLE);
CHECK_NAMED(kThreadLevels, "FUNNELED", MPI_THREAD_FUNNELED);
CHECK_NAMED(kThreadLevels, "SERIALIZED", MPI_THREAD_SERIALIZED);
CHECK_NAMED(kThreadLevels, "MULTIPLE", MPI_THREAD_MULTIPLE);

CHECK_NAMED(kReduceOps, "MAX", MPI_MAX);
CHECK_NAMED(kReduceOps, "MIN", MPI_MIN);
CHECK_NAMED(kReduceOps, "SUM", MPI_SUM);
CHECK_NAMED(kReduceOps, "PROD", MPI_PROD);
CHECK_NAMED(kReduceOps, "LAND", MPI_LAND);
CHECK_NAMED(kReduceOps, "BAND", MPI_BAND);
CHECK_NAMED(kReduceOps, "LOR", MPI_LOR);
CHECK_NAMED(kReduceOps, "BOR", MPI_BOR);
CHECK_NAMED(kReduceOps, "LXOR", MPI_LXOR);
CHECK_NAMED(kReduceOps, "BXOR", MPI_BXOR);
CHECK_NAMED(kReduceOps, "MINLOC", MPI_MINLOC);
CHECK_NAMED(kReduceOps, "MAXLOC", MPI_MAXLOC);
CHECK_NAMED(kReduceOps, "REPLACE", MPI_REPLACE);
CHECK_NAMED(kReduceOps, "NO_OP", MPI_NO_OP);
// MPICH 4.2.2 has EQUAL only as an extension.
CHECK_NAMED(kReduceOps, "EQUAL", MPIX_EQUAL);

CHECK_DATATYPE(MPI_C_BOOL);
CHECK_DATATYPE(MPI_CHAR);
CHECK_DATATYPE(MPI_SIGNED_CHAR);
CHECK_DATATYPE(MPI_UNSIGNED_CHAR);
CHECK_DATATYPE(MPI_BYTE);
CHECK_DATATYPE(MPI_SHORT);
CHECK_DATATYPE(MPI_UNSIGNED_SHORT);
CHECK_DATATYPE(MPI_INT);
CHECK_DATATYPE(MPI_UNSIGNED);
CHECK_DATATYPE(MPI_LONG);
CHECK_DATATYPE(MPI_UNSIGNED_LONG);
CHECK_DATATYPE(MPI_LONG_LONG_INT);
CHECK_DATATYPE(MPI_UNSIGNED_LONG_LONG);
CHECK_DATATYPE(MPI_UINT64_T);
CHECK_DATATYPE(MPI_FLOAT);
CHECK_DATATYPE(MPI_DOUBLE);
CHECK_DATATYPE(MPI_LONG_DOUBLE);
CHECK_DATATYPE(MPI_FLOAT_INT);
CHECK_DATATYPE(MPI_DOUBLE_INT);
CHECK_DATATYPE(MPI_2INT);

} // namespace

// MPI_STATUS_IGNORE and MPI_STATUSES_IGNORE are pointer casts, which a
// constant expression cannot hold.
int main() {
  int failures = 0;
  if (reinterpret_cast<intptr_t>(MPI_STATUS_IGNORE) != mpich::kStatusIgnore) {
    std::fprintf(stderr, "kStatusIgnore does not match MPI_STATUS_IGNORE\n");
    ++failures;
  }
  if (reinterpret_cast<intptr_t>(MPI_STATUSES_IGNORE) != mpich::kStatusIgnore) {
    std::fprintf(stderr, "kStatusIgnore does not match MPI_STATUSES_IGNORE\n");
    ++failures;
  }
  return failures == 0 ? 0 : 1;
}
