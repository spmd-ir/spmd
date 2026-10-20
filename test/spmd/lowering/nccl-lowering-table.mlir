// RUN: spmd-opt %s -test-nccl-lowering-table | FileCheck %s
//
// The NCCL calls -lower-spmd-to-nccl can emit, as it derives them from the
// CACTS map the raiser reads (lib/spmd/passes/CACTS/ncclSpmdEntries.cpp) and
// the USIS API description (includesProgramModels/nccl/api.json).  The format
// is -test-mpi-lowering-table's.
//
// Two calls of the map are not here as the raiser has them.  ncclBcast is the
// in-place ncclBroadcast and never picked.  ncclCommInitRank is missing
// altogether: the raiser makes it an spmd.commWorld and drops its size, unique
// id and rank, which no operation of the dialect carries back, so the lowering
// bootstraps the world communicator in spmd.init instead of inverting it.

module {}

// CHECK:      ncclAllGather <- spmd.allgather
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $4
// CHECK-NEXT:     sendcount: in $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $7
// CHECK-NEXT:     $2 sendCount == $5 recvCount
// CHECK-NEXT:     $3 sendType == $6 recvType
// CHECK-NEXT: ncclAllReduce <- spmd.allreduce
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $2
// CHECK-NEXT:     count: in $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $6
// CHECK-NEXT: ncclAlltoAll <- spmd.alltoall
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $4
// CHECK-NEXT:     count: in $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $7
// CHECK-NEXT:     $2 sendCount == $5 recvCount
// CHECK-NEXT:     $3 sendType == $6 recvType
// CHECK-NEXT: ncclBcast <- spmd.bcast [alias]
// CHECK-NEXT:     buff: in $1
// CHECK-NEXT:     count: in $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     root: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $6
// CHECK-NEXT:     $1 sendBuf == $2 recvBuf
// CHECK-NEXT: ncclBroadcast <- spmd.bcast
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $2
// CHECK-NEXT:     count: in $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     root: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $6
// CHECK-NEXT: ncclCommCount <- spmd.getSizeOfComm
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     count: outInt
// CHECK-NEXT: ncclCommCuDevice <- spmd.getDeviceInComm
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     device: outInt
// CHECK-NEXT: ncclCommDestroy <- spmd.commDestroy
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: ncclCommSplit <- spmd.commSplit
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     color: in $1
// CHECK-NEXT:     key: in $2
// CHECK-NEXT:     newcomm: outComm
// CHECK-NEXT:     config: NULL
// CHECK-NEXT: ncclCommUserRank <- spmd.getRankInComm
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     rank: outInt
// CHECK-NEXT: ncclGather <- spmd.gather
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $4
// CHECK-NEXT:     count: in $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     root: in $7
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $8
// CHECK-NEXT:     $2 sendCount == $5 recvCount
// CHECK-NEXT:     $3 sendType == $6 recvType
// CHECK-NEXT: ncclGroupEnd <- spmd.groupEnd
// CHECK-NEXT: ncclGroupStart <- spmd.groupStart
// CHECK-NEXT: ncclMemAlloc <- spmd.malloc
// CHECK-NEXT:     ptr: outPtr
// CHECK-NEXT:     size: in $1
// CHECK-NEXT:     $0 comm: injected
// CHECK-NEXT: ncclMemFree <- spmd.free
// CHECK-NEXT:     ptr: in $1
// CHECK-NEXT:     $0 comm: injected
// CHECK-NEXT: ncclRecv <- spmd.recv
// CHECK-NEXT:     recvbuff: in $1
// CHECK-NEXT:     count: in $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     peer: in $4
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $6
// CHECK-NEXT:     $5 tag: injected
// CHECK-NEXT: ncclReduce <- spmd.reduce
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $2
// CHECK-NEXT:     count: in $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     root: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $7
// CHECK-NEXT: ncclReduceScatter <- spmd.reduceScatter
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $2
// CHECK-NEXT:     recvcount: in $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $6
// CHECK-NEXT: ncclScatter <- spmd.scatter
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     recvbuff: in $4
// CHECK-NEXT:     count: in $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     root: in $7
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $8
// CHECK-NEXT:     $2 sendCount == $5 recvCount
// CHECK-NEXT:     $3 sendType == $6 recvType
// CHECK-NEXT: ncclSend <- spmd.send
// CHECK-NEXT:     sendbuff: in $1
// CHECK-NEXT:     count: in $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     peer: in $4
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     stream: in $6
// CHECK-NEXT:     $5 tag: injected
