// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// After -coalesce-comm each rank sends one message and receives one.
// RUN: %spmd-verify compile opt=coalesce-comm intermediates=%t.coalesced -o %t.coalesced.exe %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.coalesced/02-lowered.mlir
// RUN: %mpiexec -n 2 %t.coalesced.exe | FileCheck %s
//
// CHECK: PASS coalesce-contiguous
//
// IR-COUNT-2: llvm.call @MPI_Send(
// IR-NOT:     llvm.call @MPI_Send(
//
// Two messages carved out of one array, adjacent, to the same neighbour.
//
// The zero-copy tier of -coalesce-comm: the pieces are already contiguous, so
// merging them rewrites the count and emits no data movement at all.  Round-trip
// this file with and without --optimize and compare the MPI_Send call counts.
//
// Note the exchange is ordered even-sends-first rather than send-then-receive on
// every rank, so it does not depend on the transport buffering the message.

#include <mpi.h>
#include <stdio.h>

#define HALF 8

int main(int argc, char *argv[]) {
  int myRank, size;
  int out[2 * HALF];
  int in[2 * HALF];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size % 2 != 0) {
    MPI_Finalize();
    return 1;
  }

  int peer = myRank ^ 1;

  for (int i = 0; i < 2 * HALF; ++i) {
    out[i] = myRank * 1000 + i;
    in[i] = -1;
  }

  if (myRank % 2 == 0) {
    MPI_Send(&out[0], HALF, MPI_INT, peer, 0, MPI_COMM_WORLD);
    MPI_Send(&out[HALF], HALF, MPI_INT, peer, 1, MPI_COMM_WORLD);
    MPI_Recv(&in[0], HALF, MPI_INT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&in[HALF], HALF, MPI_INT, peer, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(&in[0], HALF, MPI_INT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(&in[HALF], HALF, MPI_INT, peer, 1, MPI_COMM_WORLD,
             MPI_STATUS_IGNORE);
    MPI_Send(&out[0], HALF, MPI_INT, peer, 0, MPI_COMM_WORLD);
    MPI_Send(&out[HALF], HALF, MPI_INT, peer, 1, MPI_COMM_WORLD);
  }

  int ok = 1;
  for (int i = 0; i < 2 * HALF; ++i)
    if (in[i] != peer * 1000 + i)
      ok = 0;

  if (ok && myRank == 0)
    printf("PASS coalesce-contiguous\n");

  MPI_Finalize();
  return ok ? 0 : 1;
}
