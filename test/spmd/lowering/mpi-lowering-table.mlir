// RUN: spmd-opt %s -test-mpi-lowering-table | FileCheck %s
//
// Every MPI call -lower-spmd-to-mpi can emit, as it derives it from the CACTS
// map the raiser reads (lib/spmd/passes/CACTS/mpiSpmdEntries.cpp) and the USIS
// API description (includesProgramModels/mpi/api.json).  Nothing here is
// written twice: each line below is what the two sources say together, and a
// change to either shows up as a change to this file.
//
// Per call: which spmd operation it raises to, whether it is an alias the
// lowering does not pick, whether it hands out a request, and the attributes
// the raiser records.  Then one line per C argument, in prototype order,
// saying what fills it -- $n is the operation's n-th operand -- followed by
// the operands the raiser injects (which the lowering checks instead) and the
// operands MPI shares one parameter for (which have to be the same value).

module {}

// CHECK:      MPI_Abort <- spmd.abort
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     errorcode: in $1
// CHECK-NEXT: MPI_Accumulate <- spmd.accumulate [isAtomic=true]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     target_count: count $7
// CHECK-NEXT:     target_datatype: in $8
// CHECK-NEXT:     op: in $10
// CHECK-NEXT:     win: in $5
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: MPI_Allgather <- spmd.allgather
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Alloc_mem <- spmd.malloc
// CHECK-NEXT:     size: in $1
// CHECK-NEXT:     info: MPI_INFO_NULL
// CHECK-NEXT:     baseptr: outPtr
// CHECK-NEXT:     $0 comm: injected
// CHECK-NEXT: MPI_Allreduce <- spmd.allreduce
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Alltoall <- spmd.alltoall
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Barrier <- spmd.barrier
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Bcast <- spmd.bcast
// CHECK-NEXT:     buffer: in $1
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     root: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     $1 sendBuf == $2 recvBuf
// CHECK-NEXT: MPI_Bsend <- spmd.send [isReadyMode=false] [isBuffered=true]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Bsend_init <- spmd.sendInit [request] [isBuffered=true] [isReadyMode=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Cart_create <- spmd.cartCreate
// CHECK-NEXT:     comm_old: in $0
// CHECK-NEXT:     ndims: in $1
// CHECK-NEXT:     dims: in $2
// CHECK-NEXT:     periods: in $3
// CHECK-NEXT:     reorder: in $4
// CHECK-NEXT:     comm_cart: outInt
// CHECK-NEXT: MPI_Comm_dup <- spmd.commDup
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     newcomm: outInt
// CHECK-NEXT: MPI_Comm_free <- spmd.commDestroy
// CHECK-NEXT:     comm: byAddress $0
// CHECK-NEXT: MPI_Comm_rank <- spmd.getRankInComm
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     rank: outInt
// CHECK-NEXT: MPI_Comm_size <- spmd.getSizeOfComm
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     size: outInt
// CHECK-NEXT: MPI_Comm_split <- spmd.commSplit
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     color: in $1
// CHECK-NEXT:     key: in $2
// CHECK-NEXT:     newcomm: outInt
// CHECK-NEXT: MPI_Compare_and_swap <- spmd.compareAndSwap [isAtomic=true]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     compare_addr: in $1
// CHECK-NEXT:     result_addr: in $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     win: in $5
// CHECK-NEXT: MPI_Dims_create <- spmd.dimsCreate
// CHECK-NEXT:     nnodes: in $0
// CHECK-NEXT:     ndims: in $1
// CHECK-NEXT:     dims: in $2
// CHECK-NEXT: MPI_Exscan <- spmd.exscan
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Fetch_and_op <- spmd.fetchAndOp [isAtomic=true]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     result_addr: in $1
// CHECK-NEXT:     datatype: in $2
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     op: in $3
// CHECK-NEXT:     win: in $5
// CHECK-NEXT: MPI_Finalize <- spmd.finalize
// CHECK-NEXT: MPI_Free_mem <- spmd.free
// CHECK-NEXT:     base: in $1
// CHECK-NEXT:     $0 comm: injected
// CHECK-NEXT: MPI_Gather <- spmd.gather
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     root: in $7
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Get <- spmd.get [isAtomic=false]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     target_count: count $7
// CHECK-NEXT:     target_datatype: in $8
// CHECK-NEXT:     win: in $5
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: MPI_Get_accumulate <- spmd.getAccumulate [isAtomic=true]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     result_addr: in $4
// CHECK-NEXT:     result_count: count $5
// CHECK-NEXT:     result_datatype: in $6
// CHECK-NEXT:     target_rank: in $8
// CHECK-NEXT:     target_disp: in $10
// CHECK-NEXT:     target_count: count $11
// CHECK-NEXT:     target_datatype: in $12
// CHECK-NEXT:     op: in $14
// CHECK-NEXT:     win: in $9
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $7 resultStride: injected
// CHECK-NEXT:     $13 targetStride: injected
// CHECK-NEXT: MPI_Iallgather <- spmd.allgather [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Iallreduce <- spmd.allreduce [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Ialltoall <- spmd.alltoall [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Ibarrier <- spmd.barrier [request]
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Ibcast <- spmd.bcast [request]
// CHECK-NEXT:     buffer: in $1
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     root: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT:     $1 sendBuf == $2 recvBuf
// CHECK-NEXT: MPI_Ibsend <- spmd.send [request] [isReadyMode=false] [isBuffered=true]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Iexscan <- spmd.exscan [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Igather <- spmd.gather [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     root: in $7
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Imrecv <- spmd.imRecv [request]
// CHECK-NEXT:     buf: in $0
// CHECK-NEXT:     count: count $1
// CHECK-NEXT:     datatype: in $2
// CHECK-NEXT:     message: byAddress $3
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Init <- spmd.init
// CHECK-NEXT:     argc: NULL
// CHECK-NEXT:     argv: NULL
// CHECK-NEXT:     $0 threadRequired: injected
// CHECK-NEXT: MPI_Init_thread <- spmd.init
// CHECK-NEXT:     argc: NULL
// CHECK-NEXT:     argv: NULL
// CHECK-NEXT:     required: in $0
// CHECK-NEXT:     provided: outInt (optional)
// CHECK-NEXT: MPI_Irecv <- spmd.recv [request]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     source: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Ireduce <- spmd.reduce [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     root: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Ireduce_scatter_block <- spmd.reduceScatter [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     recvcount: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Irsend <- spmd.send [request] [isReadyMode=true] [isBuffered=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Iscan <- spmd.scan [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Iscatter <- spmd.scatter [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     root: in $7
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Isend <- spmd.send [request] [isReadyMode=false] [isBuffered=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Isendrecv <- spmd.sendrecv [request]
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     sendtag: in $6
// CHECK-NEXT:     recvbuf: in $7
// CHECK-NEXT:     recvcount: count $8
// CHECK-NEXT:     recvtype: in $9
// CHECK-NEXT:     source: in $11
// CHECK-NEXT:     recvtag: in $12
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT:     $4 sendStride: injected
// CHECK-NEXT:     $10 recvStride: injected
// CHECK-NEXT: MPI_Isendrecv_replace <- spmd.sendrecv [alias] [request]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     sendtag: in $6
// CHECK-NEXT:     source: in $11
// CHECK-NEXT:     recvtag: in $12
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT:     $4 sendStride: injected
// CHECK-NEXT:     $10 recvStride: injected
// CHECK-NEXT:     $1 sendBuf == $7 recvBuf
// CHECK-NEXT:     $2 sendCount == $8 recvCount
// CHECK-NEXT:     $3 sendDatatype == $9 recvDatatype
// CHECK-NEXT: MPI_Issend <- spmd.send [alias] [request] [isReadyMode=false] [isBuffered=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Mprobe <- spmd.mProbe
// CHECK-NEXT:     source: in $1
// CHECK-NEXT:     tag: in $2
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     message: outInt
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT: MPI_Mrecv <- spmd.mRecv
// CHECK-NEXT:     buf: in $0
// CHECK-NEXT:     count: count $1
// CHECK-NEXT:     datatype: in $2
// CHECK-NEXT:     message: byAddress $3
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT: MPI_Pready <- spmd.pReady
// CHECK-NEXT:     partition: in $0
// CHECK-NEXT:     request: in $1
// CHECK-NEXT: MPI_Precv_init <- spmd.recvInit [request]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     partitions: in $2
// CHECK-NEXT:     count: in $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     source: in $5
// CHECK-NEXT:     tag: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     info: MPI_INFO_NULL
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Psend_init <- spmd.sendInit [request] [isBuffered=false] [isReadyMode=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     partitions: in $2
// CHECK-NEXT:     count: in $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     tag: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     info: MPI_INFO_NULL
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Put <- spmd.put [isAtomic=false]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     target_count: count $7
// CHECK-NEXT:     target_datatype: in $8
// CHECK-NEXT:     win: in $5
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: MPI_Raccumulate <- spmd.accumulate [request] [isAtomic=true]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     target_count: count $7
// CHECK-NEXT:     target_datatype: in $8
// CHECK-NEXT:     op: in $10
// CHECK-NEXT:     win: in $5
// CHECK-NEXT:     request: outInt
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: MPI_Recv <- spmd.recv
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     source: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT: MPI_Recv_init <- spmd.recvInit [request]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     source: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Reduce <- spmd.reduce
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     root: in $6
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Reduce_scatter_block <- spmd.reduceScatter
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     recvcount: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Request_free <- spmd.reqFree
// CHECK-NEXT:     request: byAddress $0
// CHECK-NEXT: MPI_Rget <- spmd.get [request] [isAtomic=false]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     target_count: count $7
// CHECK-NEXT:     target_datatype: in $8
// CHECK-NEXT:     win: in $5
// CHECK-NEXT:     request: outInt
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: MPI_Rget_accumulate <- spmd.getAccumulate [request] [isAtomic=true]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     result_addr: in $4
// CHECK-NEXT:     result_count: count $5
// CHECK-NEXT:     result_datatype: in $6
// CHECK-NEXT:     target_rank: in $8
// CHECK-NEXT:     target_disp: in $10
// CHECK-NEXT:     target_count: count $11
// CHECK-NEXT:     target_datatype: in $12
// CHECK-NEXT:     op: in $14
// CHECK-NEXT:     win: in $9
// CHECK-NEXT:     request: outInt
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $7 resultStride: injected
// CHECK-NEXT:     $13 targetStride: injected
// CHECK-NEXT: MPI_Rput <- spmd.put [request] [isAtomic=false]
// CHECK-NEXT:     origin_addr: in $0
// CHECK-NEXT:     origin_count: count $1
// CHECK-NEXT:     origin_datatype: in $2
// CHECK-NEXT:     target_rank: in $4
// CHECK-NEXT:     target_disp: in $6
// CHECK-NEXT:     target_count: count $7
// CHECK-NEXT:     target_datatype: in $8
// CHECK-NEXT:     win: in $5
// CHECK-NEXT:     request: outInt
// CHECK-NEXT:     $3 originStride: injected
// CHECK-NEXT:     $9 targetStride: injected
// CHECK-NEXT: MPI_Rsend <- spmd.send [isReadyMode=true] [isBuffered=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Rsend_init <- spmd.sendInit [request] [isBuffered=false] [isReadyMode=true]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Scan <- spmd.scan
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     recvbuf: in $2
// CHECK-NEXT:     count: count $3
// CHECK-NEXT:     datatype: in $4
// CHECK-NEXT:     op: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Scatter <- spmd.scatter
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     recvbuf: in $4
// CHECK-NEXT:     recvcount: count $5
// CHECK-NEXT:     recvtype: in $6
// CHECK-NEXT:     root: in $7
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Send <- spmd.send [isReadyMode=false] [isBuffered=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Send_init <- spmd.sendInit [request] [isBuffered=false] [isReadyMode=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Sendrecv <- spmd.sendrecv
// CHECK-NEXT:     sendbuf: in $1
// CHECK-NEXT:     sendcount: count $2
// CHECK-NEXT:     sendtype: in $3
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     sendtag: in $6
// CHECK-NEXT:     recvbuf: in $7
// CHECK-NEXT:     recvcount: count $8
// CHECK-NEXT:     recvtype: in $9
// CHECK-NEXT:     source: in $11
// CHECK-NEXT:     recvtag: in $12
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT:     $4 sendStride: injected
// CHECK-NEXT:     $10 recvStride: injected
// CHECK-NEXT: MPI_Sendrecv_replace <- spmd.sendrecv [alias]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $5
// CHECK-NEXT:     sendtag: in $6
// CHECK-NEXT:     source: in $11
// CHECK-NEXT:     recvtag: in $12
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT:     $4 sendStride: injected
// CHECK-NEXT:     $10 recvStride: injected
// CHECK-NEXT:     $1 sendBuf == $7 recvBuf
// CHECK-NEXT:     $2 sendCount == $8 recvCount
// CHECK-NEXT:     $3 sendDatatype == $9 recvDatatype
// CHECK-NEXT: MPI_Ssend <- spmd.send [alias] [isReadyMode=false] [isBuffered=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT: MPI_Ssend_init <- spmd.sendInit [alias] [request] [isBuffered=false] [isReadyMode=false]
// CHECK-NEXT:     buf: in $1
// CHECK-NEXT:     count: count $2
// CHECK-NEXT:     datatype: in $3
// CHECK-NEXT:     dest: in $4
// CHECK-NEXT:     tag: in $5
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     request: outInt
// CHECK-NEXT: MPI_Start <- spmd.start
// CHECK-NEXT:     request: byAddress $0
// CHECK-NEXT: MPI_Test <- spmd.test
// CHECK-NEXT:     request: byAddress $0
// CHECK-NEXT:     flag: flag
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT: MPI_Testall <- spmd.testAll
// CHECK-NEXT:     count: in $0
// CHECK-NEXT:     array_of_requests: in $1
// CHECK-NEXT:     flag: flag
// CHECK-NEXT:     array_of_statuses: statuses $2
// CHECK-NEXT: MPI_Testany <- spmd.testAny
// CHECK-NEXT:     count: in $0
// CHECK-NEXT:     array_of_requests: in $1
// CHECK-NEXT:     index: outInt
// CHECK-NEXT:     flag: flag
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT: MPI_Testsome <- spmd.testSome
// CHECK-NEXT:     incount: in $0
// CHECK-NEXT:     array_of_requests: in $1
// CHECK-NEXT:     outcount: outInt
// CHECK-NEXT:     array_of_indices: in $2
// CHECK-NEXT:     array_of_statuses: statuses $3
// CHECK-NEXT: MPI_Wait <- spmd.wait
// CHECK-NEXT:     request: byAddress $0
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT: MPI_Waitall <- spmd.waitAll
// CHECK-NEXT:     count: in $0
// CHECK-NEXT:     array_of_requests: in $1
// CHECK-NEXT:     array_of_statuses: statuses $2
// CHECK-NEXT: MPI_Waitany <- spmd.waitAny
// CHECK-NEXT:     count: in $0
// CHECK-NEXT:     array_of_requests: in $1
// CHECK-NEXT:     index: outInt
// CHECK-NEXT:     status: status (optional)
// CHECK-NEXT: MPI_Waitsome <- spmd.waitSome
// CHECK-NEXT:     incount: in $0
// CHECK-NEXT:     array_of_requests: in $1
// CHECK-NEXT:     outcount: outInt
// CHECK-NEXT:     array_of_indices: in $2
// CHECK-NEXT:     array_of_statuses: statuses $3
// CHECK-NEXT: MPI_Win_allocate <- spmd.winAlloc
// CHECK-NEXT:     size: in $1
// CHECK-NEXT:     disp_unit: in $2
// CHECK-NEXT:     info: MPI_INFO_NULL
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     baseptr: outPtr
// CHECK-NEXT:     win: outInt
// CHECK-NEXT: MPI_Win_complete <- spmd.winComplete
// CHECK-NEXT:     win: in $0
// CHECK-NEXT: MPI_Win_create <- spmd.winCreate
// CHECK-NEXT:     base: in $1
// CHECK-NEXT:     size: in $2
// CHECK-NEXT:     disp_unit: in $3
// CHECK-NEXT:     info: MPI_INFO_NULL
// CHECK-NEXT:     comm: in $0
// CHECK-NEXT:     win: outInt
// CHECK-NEXT: MPI_Win_fence <- spmd.fence
// CHECK-NEXT:     assert: in $1
// CHECK-NEXT:     win: in $0
// CHECK-NEXT: MPI_Win_flush <- spmd.flush [isLocal=false]
// CHECK-NEXT:     rank: in $0
// CHECK-NEXT:     win: in $1
// CHECK-NEXT: MPI_Win_flush_all <- spmd.flushAll [isLocal=false]
// CHECK-NEXT:     win: in $0
// CHECK-NEXT: MPI_Win_flush_local <- spmd.flush [isLocal=true]
// CHECK-NEXT:     rank: in $0
// CHECK-NEXT:     win: in $1
// CHECK-NEXT: MPI_Win_flush_local_all <- spmd.flushAll [isLocal=true]
// CHECK-NEXT:     win: in $0
// CHECK-NEXT: MPI_Win_free <- spmd.winFree
// CHECK-NEXT:     win: byAddress $0
// CHECK-NEXT: MPI_Win_lock <- spmd.lock
// CHECK-NEXT:     lock_type: in $2
// CHECK-NEXT:     rank: in $0
// CHECK-NEXT:     assert: in $3
// CHECK-NEXT:     win: in $1
// CHECK-NEXT: MPI_Win_lock_all <- spmd.lockAll
// CHECK-NEXT:     assert: in $1
// CHECK-NEXT:     win: in $0
// CHECK-NEXT: MPI_Win_start <- spmd.winStart
// CHECK-NEXT:     group: in $0
// CHECK-NEXT:     assert: in $1
// CHECK-NEXT:     win: in $2
// CHECK-NEXT: MPI_Win_unlock <- spmd.unlock
// CHECK-NEXT:     rank: in $0
// CHECK-NEXT:     win: in $1
// CHECK-NEXT: MPI_Win_unlock_all <- spmd.unlockAll
// CHECK-NEXT:     win: in $0
// CHECK-NEXT: module {
// CHECK-NEXT: }
