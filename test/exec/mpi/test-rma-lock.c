// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// CHECK: PASS rma-lock

/* Test: RMA passive-target synchronization — 2 ranks.
 *
 * Covers: MPI_Win_lock, MPI_Win_lock_all, MPI_Win_unlock, MPI_Win_unlock_all,
 *         MPI_Win_flush, MPI_Win_flush_local, MPI_Win_flush_all,
 *         MPI_Win_flush_local_all
 *
 * Two ranks share a one-int window per rank.  The origin rank (0) exercises
 * each synchronization primitive while the target rank (1) waits at barriers
 * and then verifies its own window memory.
 *
 * In passive-target mode only the origin calls lock/unlock; the target does
 * not participate in window synchronization.  A barrier after each unlock
 * epoch coordinates the verification step.
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
  /* Win_lock_all / Win_flush* / Win_unlock_all
   *
   * Rank 0 puts 42 into rank 1's window, then exercises all four flush
   * variants before releasing the lock.
   */
  *win_buf = rank;
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD)); /* ensure init is visible */

  CHECK_MPI(MPI_Win_lock_all(0, win));

  if (rank == 0) {
    int put_val = 42;
    CHECK_MPI(MPI_Put(&put_val, 1, MPI_INT, /*target_rank=*/1,
                      /*target_disp=*/0, 1, MPI_INT, win));

    /* flush: guarantees put has completed at target before continuing */
    CHECK_MPI(MPI_Win_flush(/*rank=*/1, win));

    /* flush_local: local (origin-side) completion — put buffer reusable */
    CHECK_MPI(MPI_Win_flush_local(/*rank=*/1, win));

    /* flush_all: completes all pending ops against every target */
    CHECK_MPI(MPI_Win_flush_all(win));

    /* flush_local_all: local completion for all targets */
    CHECK_MPI(MPI_Win_flush_local_all(win));
  }

  CHECK_MPI(MPI_Win_unlock_all(win));
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  if (rank == 1 && *win_buf != 42)
    FAIL("lock_all+flush: expected 42, got %d", *win_buf);

  /* ------------------------------------------------------------------ */
  /* Win_lock (exclusive) / Win_unlock
   *
   * Rank 0 acquires an exclusive lock on rank 1's window, puts 77, unlocks.
   */
  *win_buf = -1;
  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  if (rank == 0) {
    int put_val = 77;
    CHECK_MPI(MPI_Win_lock(MPI_LOCK_EXCLUSIVE, /*target_rank=*/1, 0, win));
    CHECK_MPI(MPI_Put(&put_val, 1, MPI_INT, /*target_rank=*/1,
                      /*target_disp=*/0, 1, MPI_INT, win));
    CHECK_MPI(MPI_Win_unlock(/*target_rank=*/1, win));
  }

  CHECK_MPI(MPI_Barrier(MPI_COMM_WORLD));

  if (rank == 1 && *win_buf != 77)
    FAIL("lock exclusive: expected 77, got %d", *win_buf);

  CHECK_MPI(MPI_Win_free(&win));

  if (rank == 0)
    printf("PASS rma-lock\n");

  CHECK_MPI(MPI_Finalize());
  return 0;
}
