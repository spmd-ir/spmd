// RUN: %spmd-verify compile -o %t %s
// RUN: %mpiexec -n 2 %t | FileCheck %s
//
// After -coalesce-comm each rank packs its two fields into one message and
// unpacks the one it receives.
// RUN: %spmd-verify compile opt=coalesce-comm intermediates=%t.coalesced -o %t.coalesced.exe %s
// RUN: FileCheck %s --check-prefix=IR --input-file=%t.coalesced/02-lowered.mlir
// RUN: %mpiexec -n 2 %t.coalesced.exe | FileCheck %s
//
// CHECK: PASS coalesce-two-field
//
// IR-COUNT-2: llvm.call @MPI_Send(
// IR-NOT:     llvm.call @MPI_Send(
//
// Two separately allocated fields sent to the same neighbour.
//
// The staging tier of -coalesce-comm: the pieces cannot be merged in place, so
// the pass packs them through a scratch buffer, sends once, and unpacks on the
// far side.  The layout is a concatenation in ascending-tag order, which both
// endpoints derive without exchanging anything.
//
// It also exercises the point that uniform *execution* is not required: the
// whole exchange sits under `if (myRank % 2 == 0)`, so no rank executes it
// uniformly, and it merges anyway.

#include <mpi.h>
#include <stdio.h>

#define N 8

int main(int argc, char *argv[]) {
  int myRank, size;
  int fieldA[N], fieldB[N];
  int inA[N], inB[N];

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size % 2 != 0) {
    MPI_Finalize();
    return 1;
  }

  int peer = myRank ^ 1;

  for (int i = 0; i < N; ++i) {
    fieldA[i] = myRank * 1000 + i;
    fieldB[i] = myRank * 2000 + i;
    inA[i] = -1;
    inB[i] = -1;
  }

  if (myRank % 2 == 0) {
    MPI_Send(fieldA, N, MPI_INT, peer, 0, MPI_COMM_WORLD);
    MPI_Send(fieldB, N, MPI_INT, peer, 1, MPI_COMM_WORLD);
    MPI_Recv(inA, N, MPI_INT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(inB, N, MPI_INT, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  } else {
    MPI_Recv(inA, N, MPI_INT, peer, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Recv(inB, N, MPI_INT, peer, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(fieldA, N, MPI_INT, peer, 0, MPI_COMM_WORLD);
    MPI_Send(fieldB, N, MPI_INT, peer, 1, MPI_COMM_WORLD);
  }

  int ok = 1;
  for (int i = 0; i < N; ++i) {
    if (inA[i] != peer * 1000 + i)
      ok = 0;
    if (inB[i] != peer * 2000 + i)
      ok = 0;
  }

  if (ok && myRank == 0)
    printf("PASS coalesce-two-field\n");

  MPI_Finalize();
  return ok ? 0 : 1;
}
