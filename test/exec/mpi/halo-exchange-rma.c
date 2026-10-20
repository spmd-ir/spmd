// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS halo-exchange-rma

#include <mpi.h>
#include <stdio.h>

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

  int rank = -1;
  int size = 0;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (size != 2) {
    if (rank == 0)
      fprintf(stderr, "Run with exactly 2 ranks for this halo exchange test.\n");
    MPI_Finalize();
    return 1;
  }

  int halo[2] = {-1, -1};
  int sendLeft = 10 + rank;
  int sendRight = 20 + rank;
  int peer = 1 - rank;

  MPI_Win win;
  CHECK_MPI(MPI_Win_create(halo, sizeof(halo), sizeof(int), MPI_INFO_NULL,
                           MPI_COMM_WORLD, &win));

  CHECK_MPI(MPI_Win_fence(0, win));
  CHECK_MPI(MPI_Put(&sendLeft, 1, MPI_INT, peer, 1, 1, MPI_INT, win));
  CHECK_MPI(MPI_Put(&sendRight, 1, MPI_INT, peer, 0, 1, MPI_INT, win));
  CHECK_MPI(MPI_Win_fence(0, win));

  if (halo[0] != 20 + peer || halo[1] != 10 + peer) {
    fprintf(stderr, "Rank %d: halo mismatch [%d, %d]\n", rank, halo[0],
            halo[1]);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }

  if (rank == 0)
    printf("PASS halo-exchange-rma\n");

  CHECK_MPI(MPI_Win_free(&win));
  CHECK_MPI(MPI_Finalize());
  return 0;
}