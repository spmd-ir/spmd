// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS rma-get

/* Test: MPI_Win_allocate and MPI_Get with fence synchronization — 2 ranks.
 *
 * halo-exchange-rma.c covers MPI_Win_create + MPI_Put + MPI_Win_fence.
 * This file exercises the Win_allocate path and MPI_Get.
 *
 * Covers: MPI_Win_allocate, MPI_Get, MPI_Win_fence, MPI_Win_free
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

  /* MPI_Win_allocate: each rank gets a one-int window. */
  int *win_buf = NULL;
  MPI_Win win;
  CHECK_MPI(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL,
                             MPI_COMM_WORLD, &win_buf, &win));

  /* Rank 0 exposes value 42; rank 1 will fetch it. */
  *win_buf = (rank == 0) ? 42 : -1;

  /* Opening fence: make local stores visible before RMA. */
  CHECK_MPI(MPI_Win_fence(0, win));

  /* Rank 1 issues MPI_Get to fetch rank 0's value. */
  int got = -1;
  if (rank == 1) {
    CHECK_MPI(MPI_Get(&got, 1, MPI_INT, /*target_rank=*/0,
                      /*target_disp=*/0, 1, MPI_INT, win));
  }

  /* Closing fence: waits for all RMA ops to complete. */
  CHECK_MPI(MPI_Win_fence(0, win));

  if (rank == 1 && got != 42)
    FAIL("Get: expected 42, got %d", got);

  /* Test a second round: rank 0 gets rank 1's window value. */
  *win_buf = (rank == 0) ? -1 : 99;
  CHECK_MPI(MPI_Win_fence(0, win));

  int got2 = -1;
  if (rank == 0) {
    CHECK_MPI(MPI_Get(&got2, 1, MPI_INT, /*target_rank=*/1,
                      /*target_disp=*/0, 1, MPI_INT, win));
  }
  CHECK_MPI(MPI_Win_fence(0, win));

  if (rank == 0 && got2 != 99)
    FAIL("Get (reverse): expected 99, got %d", got2);

  CHECK_MPI(MPI_Win_free(&win));

  if (rank == 0)
    printf("PASS rma-get\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
