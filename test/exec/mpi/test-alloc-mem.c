// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS alloc-mem

/* Test: MPI_Alloc_mem and MPI_Free_mem — 2 ranks.
 *
 * Allocates MPI-managed memory on each rank, uses it as a communication
 * buffer (Isend/Recv), verifies the received data, then frees it.
 *
 * Covers: MPI_Alloc_mem, MPI_Free_mem
 * Also exercises: MPI_Isend, MPI_Wait, MPI_Request_free of an active request
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

#define N 4

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

  /* Allocate N ints of MPI-managed memory on each rank. */
  int *mem = NULL;
  CHECK_MPI(MPI_Alloc_mem((MPI_Aint)(N * sizeof(int)), MPI_INFO_NULL, &mem));
  if (!mem)
    FAIL("Alloc_mem returned NULL");

  if (rank == 0) {
    /* Fill and send. */
    for (int i = 0; i < N; i++)
      mem[i] = (i + 1) * 10; /* [10, 20, 30, 40] */

    MPI_Request req;
    CHECK_MPI(MPI_Isend(mem, N, MPI_INT, 1, 0, MPI_COMM_WORLD, &req));
    CHECK_MPI(MPI_Wait(&req, MPI_STATUS_IGNORE));

    /* MPI_Request_free on an active request lets the send complete on its
     * own and writes MPI_REQUEST_NULL back into the variable.  Freeing
     * MPI_REQUEST_NULL itself is an error, and MPICH crashes on it. */
    CHECK_MPI(MPI_Isend(mem, N, MPI_INT, 1, 1, MPI_COMM_WORLD, &req));
    CHECK_MPI(MPI_Request_free(&req));
    if (req != MPI_REQUEST_NULL)
      FAIL("Request_free did not set the request to MPI_REQUEST_NULL");
  } else {
    /* Receive into MPI-managed buffer and verify. */
    for (int i = 0; i < N; i++)
      mem[i] = -1;

    CHECK_MPI(MPI_Recv(mem, N, MPI_INT, 0, 0, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE));

    for (int i = 0; i < N; i++) {
      int expected = (i + 1) * 10;
      if (mem[i] != expected)
        FAIL("mem[%d]: expected %d, got %d", i, expected, mem[i]);
    }

    /* The send whose request rank 0 freed. */
    for (int i = 0; i < N; i++)
      mem[i] = -1;
    CHECK_MPI(MPI_Recv(mem, N, MPI_INT, 0, 1, MPI_COMM_WORLD,
                       MPI_STATUS_IGNORE));
    for (int i = 0; i < N; i++) {
      int expected = (i + 1) * 10;
      if (mem[i] != expected)
        FAIL("mem[%d] after Request_free: expected %d, got %d", i, expected,
             mem[i]);
    }
  }

  /* Rank 0's freed send has been received once rank 1 gets here, so its
   * buffer may go. */
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));
  CHECK_MPI(MPI_Free_mem(mem));

  if (rank == 0)
    printf("PASS alloc-mem\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
