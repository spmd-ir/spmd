// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS nb-collectives-mix

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

  int bcastData = (rank == 0) ? 21 : 0;
  int send = rank + 1;
  int allreduceOut = 0;
  int reduceOut = 0;

  CHECK_MPI(MPI_Bcast(&bcastData, 1, MPI_INT, 0, MPI_COMM_WORLD));
  CHECK_MPI(
      MPI_Allreduce(&send, &allreduceOut, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD));
  CHECK_MPI(
      MPI_Reduce(&send, &reduceOut, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  int expectedSum = size * (size + 1) / 2;
  int expectedMax = size;
  if (bcastData != 21 || allreduceOut != expectedSum ||
      (rank == 0 && reduceOut != expectedMax)) {
    fprintf(stderr,
            "Rank %d: unexpected result bcast=%d allreduce=%d reduce=%d "
            "expectedSum=%d expectedMax=%d\n",
            rank, bcastData, allreduceOut, reduceOut, expectedSum, expectedMax);
    MPI_Abort(MPI_COMM_WORLD, 2);
  }

  if (rank == 0)
    printf("PASS nb-collectives-mix\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
