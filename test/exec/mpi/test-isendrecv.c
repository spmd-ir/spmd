// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS isendrecv

/* Test: MPI_Isendrecv and MPI_Isendrecv_replace — 2 ranks required.
 *
 * The non-blocking counterpart of test-sendrecv.c.  IsendRecv2.c and
 * IsendRecv_replace.c in this directory cover the two calls structurally but
 * never check what arrived; this file waits on the request and verifies the
 * received value, so a round-trip that drops or mis-wires the request result
 * shows up as a wrong answer rather than as IR that merely parses.
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
  MPI_Request req = MPI_REQUEST_NULL;

  /* --- MPI_Isendrecv ---
   * Rank 0 sends 42, rank 1 sends 99.  Each receives the peer's value. */
  int sr_send = (rank == 0) ? 42 : 99;
  int sr_recv = -1;
  CHECK_MPI(MPI_Isendrecv(&sr_send, 1, MPI_INT, peer, 10,
                          &sr_recv, 1, MPI_INT, peer, 10,
                          MPI_COMM_WORLD, &req));
  CHECK_MPI(MPI_Wait(&req, MPI_STATUS_IGNORE));
  int sr_expected = (rank == 0) ? 99 : 42;
  if (sr_recv != sr_expected)
    FAIL("Isendrecv: expected %d, got %d", sr_expected, sr_recv);

  /* --- MPI_Isendrecv_replace ---
   * Rank 0 starts with 10, rank 1 with 20.
   * After in-place exchange: rank 0 has 20, rank 1 has 10. */
  int srr_buf = (rank == 0) ? 10 : 20;
  CHECK_MPI(MPI_Isendrecv_replace(&srr_buf, 1, MPI_INT,
                                  peer, 11, peer, 11,
                                  MPI_COMM_WORLD, &req));
  CHECK_MPI(MPI_Wait(&req, MPI_STATUS_IGNORE));
  int srr_expected = (rank == 0) ? 20 : 10;
  if (srr_buf != srr_expected)
    FAIL("Isendrecv_replace: expected %d, got %d", srr_expected, srr_buf);

  if (rank == 0)
    printf("PASS isendrecv\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
