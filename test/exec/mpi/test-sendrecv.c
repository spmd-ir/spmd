// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS sendrecv

/* Test: MPI_Sendrecv and MPI_Sendrecv_replace — 2 ranks required.
 *
 * sendRecv_replace.c in this directory covers Sendrecv_replace structurally
 * but does not verify the received value.  This file tests both operations
 * with correctness checks.
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

  /* --- MPI_Sendrecv ---
   * Rank 0 sends 42, rank 1 sends 99.  Each receives the peer's value. */
  int sr_send = (rank == 0) ? 42 : 99;
  int sr_recv = -1;
  CHECK_MPI(MPI_Sendrecv(&sr_send, 1, MPI_INT, peer, 10,
                         &sr_recv, 1, MPI_INT, peer, 10,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE));
  int sr_expected = (rank == 0) ? 99 : 42;
  if (sr_recv != sr_expected)
    FAIL("Sendrecv: expected %d, got %d", sr_expected, sr_recv);

  /* --- MPI_Sendrecv_replace ---
   * Rank 0 starts with 10, rank 1 with 20.
   * After in-place exchange: rank 0 has 20, rank 1 has 10. */
  int srr_buf = (rank == 0) ? 10 : 20;
  CHECK_MPI(MPI_Sendrecv_replace(&srr_buf, 1, MPI_INT,
                                 peer, 11, peer, 11,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE));
  int srr_expected = (rank == 0) ? 20 : 10;
  if (srr_buf != srr_expected)
    FAIL("Sendrecv_replace: expected %d, got %d", srr_expected, srr_buf);

  if (rank == 0)
    printf("PASS sendrecv\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
