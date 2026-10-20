// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 4 %t | FileCheck %s
//
// CHECK: PASS collectives

/* Test: blocking collective operations — 4 ranks required.
 *
 * Covers: MPI_Barrier, MPI_Bcast, MPI_Reduce, MPI_Allreduce,
 *         MPI_Gather, MPI_Scatter, MPI_Allgather, MPI_Alltoall,
 *         MPI_Scan, MPI_Exscan, MPI_Reduce_scatter_block
 *
 * Each section computes a result whose expected value is analytically
 * derived from the rank, then checks correctness.  Aborts on any mismatch.
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

  if (size != 4) {
    if (rank == 0)
      fprintf(stderr, "Run with exactly 4 ranks.\n");
    MPI_Finalize();
    return 1;
  }

  /* send_val for each rank = rank+1: [1, 2, 3, 4] */
  int send_val = rank + 1;

  /* --- MPI_Barrier --- */
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  /* --- MPI_Bcast ---
   * Root=0 broadcasts 77; all ranks verify. */
  int bcast_val = (rank == 0) ? 77 : -1;
  CHECK_MPI(MPI_Bcast(&bcast_val, 1, MPI_INT, 0, MPI_COMM_WORLD));
  if (bcast_val != 77)
    FAIL("Bcast: expected 77, got %d", bcast_val);

  /* --- MPI_Reduce ---
   * Sum = 1+2+3+4 = 10, delivered to root=0. */
  int reduce_result = -1;
  CHECK_MPI(
      MPI_Reduce(&send_val, &reduce_result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD));
  if (rank == 0 && reduce_result != 10)
    FAIL("Reduce: expected 10, got %d", reduce_result);

  /* --- MPI_Allreduce ---
   * Same sum = 10, delivered to every rank. */
  int allreduce_result = -1;
  CHECK_MPI(MPI_Allreduce(&send_val, &allreduce_result, 1, MPI_INT, MPI_SUM,
                          MPI_COMM_WORLD));
  if (allreduce_result != 10)
    FAIL("Allreduce: expected 10, got %d", allreduce_result);

  /* --- MPI_Gather ---
   * Each rank sends rank+1; root collects [1, 2, 3, 4]. */
  int gather_recv[4];
  memset(gather_recv, -1, sizeof(gather_recv));
  CHECK_MPI(MPI_Gather(&send_val, 1, MPI_INT, gather_recv, 1, MPI_INT, 0,
                       MPI_COMM_WORLD));
  if (rank == 0) {
    for (int i = 0; i < size; i++) {
      if (gather_recv[i] != i + 1)
        FAIL("Gather[%d]: expected %d, got %d", i, i + 1, gather_recv[i]);
    }
  }

  /* --- MPI_Scatter ---
   * Root distributes [10, 20, 30, 40]; rank j expects 10*(j+1). */
  int scatter_send[4] = {0};
  if (rank == 0) {
    for (int i = 0; i < size; i++)
      scatter_send[i] = 10 * (i + 1);
  }
  int scatter_recv = -1;
  CHECK_MPI(MPI_Scatter(scatter_send, 1, MPI_INT, &scatter_recv, 1, MPI_INT,
                        0, MPI_COMM_WORLD));
  if (scatter_recv != 10 * (rank + 1))
    FAIL("Scatter: expected %d, got %d", 10 * (rank + 1), scatter_recv);

  /* --- MPI_Allgather ---
   * Each rank sends rank+1; all verify recvbuf = [1, 2, 3, 4]. */
  int allgather_recv[4];
  CHECK_MPI(MPI_Allgather(&send_val, 1, MPI_INT, allgather_recv, 1, MPI_INT,
                          MPI_COMM_WORLD));
  for (int i = 0; i < size; i++) {
    if (allgather_recv[i] != i + 1)
      FAIL("Allgather[%d]: expected %d, got %d", i, i + 1, allgather_recv[i]);
  }

  /* --- MPI_Alltoall ---
   * sendBuf[j] = rank*size + j.
   * After alltoall: recvBuf[i] = i*size + rank. */
  int a2a_send[4], a2a_recv[4];
  for (int j = 0; j < size; j++)
    a2a_send[j] = rank * size + j;
  CHECK_MPI(MPI_Alltoall(a2a_send, 1, MPI_INT, a2a_recv, 1, MPI_INT,
                         MPI_COMM_WORLD));
  for (int i = 0; i < size; i++) {
    int expected = i * size + rank;
    if (a2a_recv[i] != expected)
      FAIL("Alltoall recv[%d]: expected %d, got %d", i, expected, a2a_recv[i]);
  }

  /* --- MPI_Scan ---
   * Inclusive prefix sum of send_val = rank+1.
   * Rank i gets (i+1)*(i+2)/2. */
  int scan_result = -1;
  CHECK_MPI(MPI_Scan(&send_val, &scan_result, 1, MPI_INT, MPI_SUM,
                     MPI_COMM_WORLD));
  int scan_expected = (rank + 1) * (rank + 2) / 2;
  if (scan_result != scan_expected)
    FAIL("Scan: expected %d, got %d", scan_expected, scan_result);

  /* --- MPI_Exscan ---
   * Exclusive prefix sum of send_val = rank+1.
   * Rank i (i>0) gets i*(i+1)/2.  Rank 0 result is undefined by standard. */
  int exscan_result = 0;
  CHECK_MPI(MPI_Exscan(&send_val, &exscan_result, 1, MPI_INT, MPI_SUM,
                       MPI_COMM_WORLD));
  if (rank > 0) {
    int exscan_expected = rank * (rank + 1) / 2;
    if (exscan_result != exscan_expected)
      FAIL("Exscan: expected %d, got %d", exscan_expected, exscan_result);
  }

  /* --- MPI_Reduce_scatter_block ---
   * sendbuf[j] = (rank+1)*(j+1).
   * recvbuf at rank j = sum_i(sendbuf_i[j]) = (j+1)*10. */
  int rsb_send[4];
  for (int j = 0; j < size; j++)
    rsb_send[j] = (rank + 1) * (j + 1);
  int rsb_recv = -1;
  CHECK_MPI(MPI_Reduce_scatter_block(rsb_send, &rsb_recv, 1, MPI_INT, MPI_SUM,
                                     MPI_COMM_WORLD));
  int rsb_expected = (rank + 1) * 10;
  if (rsb_recv != rsb_expected)
    FAIL("Reduce_scatter_block: expected %d, got %d", rsb_expected, rsb_recv);

  if (rank == 0)
    printf("PASS collectives\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
