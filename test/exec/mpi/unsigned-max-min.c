// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS unsigned-max-min
//
// MAX and MIN over unsigned datatypes, with one value that has its top bit
// set.  Compared as signed, 1 is the maximum and the large value the minimum,
// so a round trip that loses the datatype's signedness gets both of them
// wrong.

#include <limits.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

static void check_mpi(int err, int line) {
  if (err == MPI_SUCCESS)
    return;
  char errstr[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(err, errstr, &len);
  fprintf(stderr, "MPI error at line %d: %.*s\n", line, len, errstr);
  MPI_Abort(MPI_COMM_WORLD, err);
}

#define CHECK_MPI(call) check_mpi((call), __LINE__)

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));

  int rank = -1, size = 0;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (size < 2) {
    if (rank == 0)
      fprintf(stderr, "Run with at least 2 ranks.\n");
    MPI_Finalize();
    return 1;
  }

  // Rank 0 contributes the large value, every other rank 1.
  unsigned big = (unsigned)INT_MAX + 1u;
  unsigned value = rank == 0 ? big : 1u;
  unsigned long bigLong = (unsigned long)LONG_MAX + 1ul;
  unsigned long valueLong = rank == 0 ? bigLong : 1ul;
  unsigned char bigChar = (unsigned char)SCHAR_MAX + 1u;
  unsigned char valueChar = rank == 0 ? bigChar : 1u;

  unsigned max = 0, min = 0;
  unsigned long maxLong = 0, minLong = 0;
  unsigned char maxChar = 0, minChar = 0;

  CHECK_MPI(MPI_Allreduce(&value, &max, 1, MPI_UNSIGNED, MPI_MAX,
                          MPI_COMM_WORLD));
  CHECK_MPI(MPI_Allreduce(&value, &min, 1, MPI_UNSIGNED, MPI_MIN,
                          MPI_COMM_WORLD));
  CHECK_MPI(MPI_Allreduce(&valueLong, &maxLong, 1, MPI_UNSIGNED_LONG, MPI_MAX,
                          MPI_COMM_WORLD));
  CHECK_MPI(MPI_Allreduce(&valueLong, &minLong, 1, MPI_UNSIGNED_LONG, MPI_MIN,
                          MPI_COMM_WORLD));
  CHECK_MPI(MPI_Allreduce(&valueChar, &maxChar, 1, MPI_UNSIGNED_CHAR, MPI_MAX,
                          MPI_COMM_WORLD));
  CHECK_MPI(MPI_Allreduce(&valueChar, &minChar, 1, MPI_UNSIGNED_CHAR, MPI_MIN,
                          MPI_COMM_WORLD));

  if (max != big || min != 1u || maxLong != bigLong || minLong != 1ul ||
      maxChar != bigChar || minChar != 1u) {
    fprintf(stderr,
            "Rank %d: unexpected result unsigned max=%u min=%u, "
            "unsigned long max=%lu min=%lu, unsigned char max=%u min=%u\n",
            rank, max, min, maxLong, minLong, (unsigned)maxChar,
            (unsigned)minChar);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  if (rank == 0)
    printf("PASS unsigned-max-min\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
