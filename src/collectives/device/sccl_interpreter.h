/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "devcomm.h"
#include "primitives.h"
#include "collectives.h"

#define SCCL_MAX_ITER 65536

// flags are a 3-tuple of (workindex, gridoffset_iter, step) and it follows a lexicographical order. a threadblock is ahead of another iff its flag is ahead 
#define COMPUTE_FLAG(__WORKINDEX__,__GRIDOFFSET_ITER__,__STEP__) \
   SCCL_MAX_ITER*SCCL_MAX_NUM_STEPS*(uint64_t)__WORKINDEX__ + ((uint64_t)__GRIDOFFSET_ITER__ * SCCL_MAX_NUM_STEPS + (uint64_t)__STEP__)

template<typename T, typename PRIMS_WRAPPER>
class scclFunction {
  public:
    __device__ void run(struct ncclWorkElem* args, int sizeMultiplier) {
      struct ncclDevComm* comm = args->comm;
      const int tid = threadIdx.x;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for SCCL cross thread synchronization
      const int bid = blockIdx.x;
      const int bdim = blockDim.x;
      struct ncclChannel* channel = comm->channels;

      // Compute pointers
      T * thisInput = (T*)args->sendbuff;

      T * thisScratch = (T*)args->scratchbuff;
      int myRank = channel->ring.devUserRanks[0];
      int peer = (myRank > bid) ? bid : bid+1;
      PRIMS_WRAPPER prims{args, tid, &peer, &peer, thisInput, channel};

      const ssize_t size = args->coll.count;
      const int sizePerChunk = size/8;
      size_t chunkOffset = prims.initIter(sizePerChunk, 0);

      // sccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scclFlag* scclFlags = comm->scclAlgo.flags;

      if (tid == 0) printf("here0\n");
      prims.send(thisInput+peer*sizePerChunk, peer*sizePerChunk, 1);
      if (tid == 0) printf("here1\n");
      prims.recv(thisScratch+bid*sizePerChunk, bid*sizePerChunk, 1);
      if (tid == 0) printf("here1.5\n");
      if (tid == sync_tid){
        __threadfence();
        uint64_t curFlag = COMPUTE_FLAG(workIndex, 0, 0);
        scclFlags[bid].flag = curFlag;
      }
      if (tid == 0) printf("here1.6\n");
      if (tid < 7){
        uint64_t goalFlag = COMPUTE_FLAG(workIndex, 0, 0);
        while ((scclFlags + tid)->flag < goalFlag){};
      }
      __syncthreads();
      if (tid == 0) printf("here1.7\n");

      const int nthreads = args->nThreads;
      for (int j = bid*bdim+tid; j < sizePerChunk; j += nthreads*7){
        T t = thisInput[myRank*sizePerChunk+j];
        for (int i = 0; i < 7; i++){
          T c = thisScratch[i*sizePerChunk+j];
          t = c + t;
        }
        thisInput[myRank*sizePerChunk+j] = t;
      }
      __syncthreads();
      if (tid == 0) printf("here1.8\n");

      if (bid*bdim < sizePerChunk && tid == sync_tid){
        __threadfence();
        uint64_t curFlag = COMPUTE_FLAG(workIndex, 0, 1);
        scclFlags[bid].flag = curFlag;
      }
      if (tid == 0) printf("here1.9\n");
      if (tid*bdim < sizePerChunk){
        uint64_t goalFlag = COMPUTE_FLAG(workIndex, 0, 1);
        while ((scclFlags + tid)->flag < goalFlag){};
      }
      if (tid == 0) printf("here1.10\n");
      __syncthreads();
      if (tid == 0) printf("here2\n");
      prims.send(thisInput+myRank*sizePerChunk, myRank*sizePerChunk, 1);
      prims.recv(thisInput+peer*sizePerChunk, peer*sizePerChunk, 1);
      if (tid == 0) printf("here3\n");
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
class scclFunctionSimple : public scclFunction<T, SimpleWrapper<FUNC, T, UNROLL>> {};

#include "prims_ll128.h"
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
class scclFunctionLL128 : public scclFunction<T, LL128Wrapper<FUNC, T>> {};

template<class FUNC, typename T>
struct LLWrapper {
  const int stepLines;
  const ssize_t chunkSize;
  int nelem;
  
  ncclLLPrimitives<T, FUNC, 1, 1> prims;

  __device__ LLWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel)
    : stepLines(args->comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS)),
      chunkSize(stepLines * sizeof(uint64_t) / sizeof(T)),
      prims(tid, args->nThreads, recvPeer, sendPeer, stepLines, channel, args->comm) {}

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
class scclFunctionLL : public scclFunction<T, LLWrapper<FUNC, T>> {};
