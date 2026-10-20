// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS isend-irecv-wait

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

  int sendVal = rank + 42;
  int recvVal = -1;
  int peer = 1 - rank;

  MPI_Request recvReq, sendReq;
  CHECK_MPI(
      MPI_Irecv(&recvVal, 1, MPI_INT, peer, 17, MPI_COMM_WORLD, &recvReq));
  CHECK_MPI(
      MPI_Isend(&sendVal, 1, MPI_INT, peer, 17, MPI_COMM_WORLD, &sendReq));

  CHECK_MPI(MPI_Wait(&recvReq, MPI_STATUS_IGNORE));
  CHECK_MPI(MPI_Wait(&sendReq, MPI_STATUS_IGNORE));
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  if (recvVal != peer + 42) {
    fprintf(stderr, "Rank %d: expected %d, got %d\n", rank, peer + 42,
            recvVal);
    MPI_Abort(MPI_COMM_WORLD, 3);
  }

  if (rank == 0)
    printf("PASS isend-irecv-wait\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
