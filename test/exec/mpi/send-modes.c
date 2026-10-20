// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS send-modes-ssend-bsend-rsend

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

  if (size != 2) {
    if (rank == 0)
      fprintf(stderr, "Run with exactly 2 ranks for this test.\n");
    MPI_Finalize();
    return 1;
  }

  int recv0 = -1, recv1 = -1, recv2 = -1;
  int peer = 1 - rank;

  int bufferSize = 0;
  CHECK_MPI(MPI_Pack_size(1, MPI_INT, MPI_COMM_WORLD, &bufferSize));
  bufferSize += MPI_BSEND_OVERHEAD;
  char *buffer = (char *)malloc(bufferSize);
  CHECK_MPI(MPI_Buffer_attach(buffer, bufferSize));

  int sendA = 100 + rank;
  int sendB = 200 + rank;
  int sendC = 300 + rank;

  MPI_Request r0;
  CHECK_MPI(MPI_Irecv(&recv0, 1, MPI_INT, peer, 31, MPI_COMM_WORLD, &r0));
  CHECK_MPI(MPI_Ssend(&sendA, 1, MPI_INT, peer, 31, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Wait(&r0, MPI_STATUS_IGNORE));

  MPI_Request r1;
  CHECK_MPI(MPI_Irecv(&recv1, 1, MPI_INT, peer, 32, MPI_COMM_WORLD, &r1));
  CHECK_MPI(MPI_Bsend(&sendB, 1, MPI_INT, peer, 32, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Wait(&r1, MPI_STATUS_IGNORE));

  MPI_Request r2;
  CHECK_MPI(MPI_Irecv(&recv2, 1, MPI_INT, peer, 33, MPI_COMM_WORLD, &r2));
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  CHECK_MPI(MPI_Rsend(&sendC, 1, MPI_INT, peer, 33, MPI_COMM_WORLD));
  CHECK_MPI(MPI_Wait(&r2, MPI_STATUS_IGNORE));

  CHECK_MPI(MPI_Buffer_detach(&buffer, &bufferSize));
  free(buffer);

  if (recv0 != 100 + peer || recv1 != 200 + peer || recv2 != 300 + peer) {
    fprintf(stderr, "Rank %d: send-mode mismatch r0=%d r1=%d r2=%d\n", rank,
            recv0, recv1, recv2);
    MPI_Abort(MPI_COMM_WORLD, 4);
  }

  if (rank == 0)
    printf("PASS send-modes-ssend-bsend-rsend\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
