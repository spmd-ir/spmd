// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS rma-atomics

/* Test: atomic RMA operations with fence synchronization — 2 ranks.
 *
 * Covers: MPI_Accumulate, MPI_Get_accumulate, MPI_Fetch_and_op,
 *         MPI_Compare_and_swap
 *
 * Each section resets the window value, runs one atomic op from rank 1
 * targeting rank 0, then verifies both the result buffer and the updated
 * window contents after the closing fence.
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

  int *win_buf = NULL;
  MPI_Win win;
  CHECK_MPI(MPI_Win_allocate(sizeof(int), sizeof(int), MPI_INFO_NULL,
                             MPI_COMM_WORLD, &win_buf, &win));

  /* ------------------------------------------------------------------ */
  /* MPI_Accumulate (MPI_SUM)
   * Rank 0 window = 10.  Rank 1 accumulates +5.  Expected: 15. */
  *win_buf = (rank == 0) ? 10 : 0;
  CHECK_MPI(MPI_Win_fence(0, win));

  if (rank == 1) {
    int operand = 5;
    CHECK_MPI(MPI_Accumulate(&operand, 1, MPI_INT, /*target_rank=*/0,
                             /*target_disp=*/0, 1, MPI_INT, MPI_SUM, win));
  }
  CHECK_MPI(MPI_Win_fence(0, win));

  if (rank == 0 && *win_buf != 15)
    FAIL("Accumulate: expected 15, got %d", *win_buf);

  /* ------------------------------------------------------------------ */
  /* MPI_Get_accumulate (MPI_SUM)
   * Rank 0 window = 20.  Rank 1: result gets old value (20), window += 7. */
  *win_buf = (rank == 0) ? 20 : 0;
  CHECK_MPI(MPI_Win_fence(0, win));

  int ga_result = -1;
  if (rank == 1) {
    int origin = 7;
    CHECK_MPI(MPI_Get_accumulate(&origin, 1, MPI_INT,
                                 &ga_result, 1, MPI_INT,
                                 /*target_rank=*/0, /*target_disp=*/0,
                                 1, MPI_INT, MPI_SUM, win));
  }
  CHECK_MPI(MPI_Win_fence(0, win));

  if (rank == 1 && ga_result != 20)
    FAIL("Get_accumulate result: expected 20 (old), got %d", ga_result);
  if (rank == 0 && *win_buf != 27)
    FAIL("Get_accumulate target: expected 27, got %d", *win_buf);

  /* ------------------------------------------------------------------ */
  /* MPI_Fetch_and_op (MPI_SUM)
   * Rank 0 window = 100.  Rank 1 adds 7; gets back 100.  Window → 107. */
  *win_buf = (rank == 0) ? 100 : 0;
  CHECK_MPI(MPI_Win_fence(0, win));

  int fop_result = -1;
  if (rank == 1) {
    int operand = 7;
    CHECK_MPI(MPI_Fetch_and_op(&operand, &fop_result, MPI_INT,
                               /*target_rank=*/0, /*target_disp=*/0,
                               MPI_SUM, win));
  }
  CHECK_MPI(MPI_Win_fence(0, win));

  if (rank == 1 && fop_result != 100)
    FAIL("Fetch_and_op result: expected 100 (old), got %d", fop_result);
  if (rank == 0 && *win_buf != 107)
    FAIL("Fetch_and_op target: expected 107, got %d", *win_buf);

  /* ------------------------------------------------------------------ */
  /* MPI_Compare_and_swap
   * Rank 0 window = 42.  Rank 1 swaps 99 in if window == 42.
   * Result (old value) = 42; window → 99. */
  *win_buf = (rank == 0) ? 42 : 0;
  CHECK_MPI(MPI_Win_fence(0, win));

  int cas_result = -1;
  if (rank == 1) {
    int origin = 99, compare = 42;
    CHECK_MPI(MPI_Compare_and_swap(&origin, &compare, &cas_result, MPI_INT,
                                   /*target_rank=*/0, /*target_disp=*/0, win));
  }
  CHECK_MPI(MPI_Win_fence(0, win));

  if (rank == 1 && cas_result != 42)
    FAIL("Compare_and_swap result: expected 42 (old), got %d", cas_result);
  if (rank == 0 && *win_buf != 99)
    FAIL("Compare_and_swap target: expected 99, got %d", *win_buf);

  /* Second CAS where compare does not match: window should not change. */
  CHECK_MPI(MPI_Win_fence(0, win)); /* rank 0 window is already 99 */
  if (rank == 1) {
    int origin = 0, compare = 42, result2 = -1; /* compare=42 won't match 99 */
    CHECK_MPI(MPI_Compare_and_swap(&origin, &compare, &result2, MPI_INT,
                                   /*target_rank=*/0, /*target_disp=*/0, win));
  }
  CHECK_MPI(MPI_Win_fence(0, win));
  if (rank == 0 && *win_buf != 99)
    FAIL("Compare_and_swap (no-match): window changed unexpectedly to %d", *win_buf);

  CHECK_MPI(MPI_Win_free(&win));

  if (rank == 0)
    printf("PASS rma-atomics\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
