// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS request-arrays

/* Test: completion calls over an array of requests — 2 ranks.
 *
 * Covers: MPI_Waitall, MPI_Waitany in a loop until MPI_UNDEFINED, MPI_Testall
 *
 * MPI writes MPI_REQUEST_NULL back into every request it completes, and a
 * loop over MPI_Waitany depends on that to terminate.  Each section checks the
 * array afterwards, so a round trip that completes copies of the requests
 * instead of the program's own array shows up as a wrong value or a hang.
 */

#include <mpi.h>
#include <stdio.h>

static void check_mpi(int err, int line) {
  if (err == MPI_SUCCESS)
    return;
  char msg[MPI_MAX_ERROR_STRING];
  int len = 0;
  MPI_Error_string(err, msg, &len);
  fprintf(stderr, "MPI error at line %d: %.*s\n", line, len, msg);
  MPI_Abort(MPI_COMM_WORLD, err);
}

#define CHECK_MPI(call) check_mpi((call), __LINE__)

#define FAIL(fmt, ...)                                                         \
  do {                                                                         \
    fprintf(stderr, "FAIL rank %d: " fmt "\n", rank, ##__VA_ARGS__);          \
    MPI_Abort(MPI_COMM_WORLD, 1);                                              \
  } while (0)

int main(int argc, char **argv) {
  CHECK_MPI(MPI_Init(&argc, &argv));

  int rank = -1, size = 0;
  CHECK_MPI(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  CHECK_MPI(MPI_Comm_size(MPI_COMM_WORLD, &size));

  if (size != 2) {
    if (rank == 0)
      fprintf(stderr, "Run with exactly 2 ranks.\n");
    MPI_Finalize();
    return 1;
  }

  int peer = 1 - rank;
  MPI_Request reqs[2];

  /* --- MPI_Waitall --- */
  int send = 100 + rank, recv = -1;
  CHECK_MPI(MPI_Irecv(&recv, 1, MPI_INT, peer, 1, MPI_COMM_WORLD, &reqs[0]));
  CHECK_MPI(MPI_Isend(&send, 1, MPI_INT, peer, 1, MPI_COMM_WORLD, &reqs[1]));
  CHECK_MPI(MPI_Waitall(2, reqs, MPI_STATUSES_IGNORE));
  if (recv != 100 + peer)
    FAIL("Waitall: expected %d, got %d", 100 + peer, recv);
  if (reqs[0] != MPI_REQUEST_NULL || reqs[1] != MPI_REQUEST_NULL)
    FAIL("Waitall left a request that is not MPI_REQUEST_NULL");

  /* --- MPI_Waitany until MPI_UNDEFINED --- */
  int send2 = 200 + rank, recv2 = -1;
  CHECK_MPI(MPI_Irecv(&recv2, 1, MPI_INT, peer, 2, MPI_COMM_WORLD, &reqs[0]));
  CHECK_MPI(MPI_Isend(&send2, 1, MPI_INT, peer, 2, MPI_COMM_WORLD, &reqs[1]));
  int completed = 0;
  for (;;) {
    int index = -1;
    CHECK_MPI(MPI_Waitany(2, reqs, &index, MPI_STATUS_IGNORE));
    if (index == MPI_UNDEFINED)
      break;
    if (index < 0 || index > 1)
      FAIL("Waitany: index %d", index);
    if (++completed > 2)
      FAIL("Waitany completed more requests than were posted");
  }
  if (completed != 2)
    FAIL("Waitany: completed %d of 2", completed);
  if (recv2 != 200 + peer)
    FAIL("Waitany: expected %d, got %d", 200 + peer, recv2);

  /* --- MPI_Testall, polled until both are done --- */
  int send3 = 300 + rank, recv3 = -1;
  CHECK_MPI(MPI_Irecv(&recv3, 1, MPI_INT, peer, 3, MPI_COMM_WORLD, &reqs[0]));
  CHECK_MPI(MPI_Isend(&send3, 1, MPI_INT, peer, 3, MPI_COMM_WORLD, &reqs[1]));
  int done = 0;
  while (!done)
    CHECK_MPI(MPI_Testall(2, reqs, &done, MPI_STATUSES_IGNORE));
  if (recv3 != 300 + peer)
    FAIL("Testall: expected %d, got %d", 300 + peer, recv3);
  if (reqs[0] != MPI_REQUEST_NULL || reqs[1] != MPI_REQUEST_NULL)
    FAIL("Testall left a request that is not MPI_REQUEST_NULL");

  if (rank == 0)
    printf("PASS request-arrays\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
