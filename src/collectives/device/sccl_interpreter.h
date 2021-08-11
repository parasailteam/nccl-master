/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"
#include "prims_ll128.h"
#define SCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead 
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
   SCCL_MAX_ITER*SCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * SCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

template<typename T, class FUNC, typename PRIMS_WRAPPER>
class scclFunction {
  public:
    __device__ void run(struct ncclWorkElem* args, int sizeMultiplier) {
      struct ncclDevComm* comm = args->comm;
      int nranks = comm->nRanks;
      // struct scclAlgorithm* scclAlgo = &comm->scclAlgo;
      const int tid = threadIdx.x;
      const int bid = blockIdx.x;
      const int nthreads = args->nThreads;

      // Compute pointers
      T * thisInput = (T*)args->sendbuff;
      // T * thisOutput = (T*)args->recvbuff;
      // T * thisScratch = (T*)args->scratchbuff;

      int nghrs [] = {1,2,3,4,5,6,7,
                      0,2,3,4,5,6,7,
                      0,1,3,4,5,6,7,
                      0,1,2,4,5,6,7,
                      0,1,2,3,5,6,7,
                      0,1,2,3,4,6,7,
                      0,1,2,3,4,5,7,
                      0,1,2,3,4,5,6};

      const int stepLines = comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS);
      ssize_t chunkSize = stepLines * sizeof(uint64_t) / sizeof(T);
      const ssize_t size = args->coll.count;
      // const ssize_t sizePerGpu = size/nranks;
      struct ncclRing* ring = &comm->channels->ring;
      const ssize_t loopSize = nranks*chunkSize;
      const ssize_t minChunkSize = nthreads * (sizeof(uint64_t)) / sizeof(T);

      int myRank = ring->devUserRanks[0];
      int myOutNghr = nghrs[myRank*7+bid];
      ncclLLPrimitives<T, FUNC, 1, 1> LLprims2(tid, nthreads, &myOutNghr, &myOutNghr, stepLines, comm->channels+1, comm);
      
      if (bid == 0){
        int* mynghrs = &nghrs[myRank*7];
        ncclLLPrimitives<T, FUNC, 7, 1> LLprims(tid, nthreads, mynghrs, &myOutNghr, stepLines, comm->channels, comm);

        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize){
          chunkSize = min(DIVUP(size-gridOffset, nranks*minChunkSize)*minChunkSize, chunkSize);
          int nelem = min(chunkSize, (size-gridOffset)/nranks);
          LLprims.send(thisInput+gridOffset+nelem*myOutNghr, nelem);
          LLprims.recvReduceCopy(thisInput+gridOffset+myRank*chunkSize, thisInput+gridOffset+myRank*nelem, nelem);
          LLprims2.send(thisInput+gridOffset+nelem*myRank, nelem);
          LLprims2.recv(thisInput+gridOffset+nelem*myOutNghr, nelem);
        }
      } else {
        int m1 = -1;
        ncclLLPrimitives<T, FUNC, 0, 1> LLprims(tid, nthreads, &m1, &myOutNghr, stepLines, comm->channels, comm);
        for (ssize_t gridOffset = 0; gridOffset < size; gridOffset += loopSize){
          chunkSize = min(DIVUP(size-gridOffset, nranks*minChunkSize)*minChunkSize, chunkSize);
          int nelem = min(chunkSize, (size-gridOffset)/nranks);
          LLprims.send(thisInput+gridOffset+nelem*myOutNghr, nelem);
          LLprims2.send(thisInput+gridOffset+nelem*myRank, nelem);
          LLprims2.recv(thisInput+gridOffset+nelem*myOutNghr, nelem);
        }
      }
    }
};

template<class FUNC, typename T, int UNROLL>
struct SimpleWrapper {
  const int nthreads;
  const int stepSize;
  const int chunkSize;
  int nelem;

  ncclPrimitives<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ SimpleWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : nthreads(args->nThreads-WARP_SIZE),
      stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
      chunkSize(stepSize * SCCL_CHUNKSTEPS),
      prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    int realChunkSize = min(chunkSize, sizePerScclChunk-gridOffset);
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset;
    nelem = min(realChunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.directSend(chunkPointer, dstoffset, nelem*count);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.directRecv(chunkPointer, dstoffset, nelem*count);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.directRecvCopySend(chunkPointer, dstoffset, nelem*count);
  }
  
  __device__ void recvReduceSend(T * chunkPointer, int count) {
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionSimple : public scclFunction<T, FUNC, SimpleWrapper<FUNC, T, UNROLL>> {};


template<class FUNC, typename T>
struct LL128Wrapper {
  const int stepSize;
  ssize_t chunkSize;
  const ssize_t minChunkSize;
  int nelem;

  ncclLL128Primitives<T, FUNC, 1, 1> prims;

  __device__ LL128Wrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : stepSize(args->comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS)),
      chunkSize(stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T))),
      minChunkSize((NCCL_LL128_SHMEM_ELEMS_PER_THREAD*args->nThreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2),
      prims(tid, args->nThreads, recvPeer, sendPeer, stepSize, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    chunkSize = min(chunkSize, DIVUP(sizePerScclChunk-gridOffset,minChunkSize)*minChunkSize);
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.send(chunkPointer, nelem*count);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recv(chunkPointer, nelem*count);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recvCopySend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceSend(T * chunkPointer, int count) {
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL128 : public scclFunction<T, FUNC, LL128Wrapper<FUNC, T>> {};

template<class FUNC, typename T>
struct LLWrapper {
  const int stepLines;
  const ssize_t chunkSize;
  int nelem;
  
  ncclLLPrimitives<T, FUNC, 1, 1> prims;

  __device__ LLWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : stepLines(args->comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS)),
      chunkSize(stepLines * sizeof(uint64_t) / sizeof(T)),
      prims(tid, args->nThreads-WARP_SIZE, recvPeer, sendPeer, stepLines, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ void send(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.send(chunkPointer, nelem*count);
  }

  __device__ void recv(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recv(chunkPointer, nelem*count);
  }

  __device__ void recvCopySend(T * chunkPointer, ssize_t dstoffset, int count) {
    prims.recvCopySend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceSend(T * chunkPointer, int count) {
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
  
  __device__ void reduce(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL : public scclFunction<T, FUNC, LLWrapper<FUNC, T>> {};
