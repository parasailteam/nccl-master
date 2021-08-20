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

//Input/output type (T), offset type (Offset_t) as size_t or Block2D
//and primitives wrapper for ncclPrimitives, llprimitives, and ll128primitives
template<typename T, typename Offset, typename PRIMS_WRAPPER>
class scclFunction {
  public:
    __device__ void run(struct ncclWorkElem* args, int sizeMultiplier) {
      struct ncclDevComm* comm = args->comm;
      struct scclAlgorithm* scclAlgo = &comm->scclAlgo;
      const int tid = threadIdx.x;
      const int sync_tid = args->nThreads-1; // last thread is most likely not doing anthing and used for SCCL cross thread synchronization
      const int bid = blockIdx.x;
      struct scclThreadBlock* scclTB = &scclAlgo->scclTB[bid];
      const int channelId = scclTB->channelId;
      struct ncclChannel* channel = comm->channels+channelId;

      // Compute pointers
      T * thisInput = (T*)args->sendbuff;
      T * thisOutput = (T*)args->recvbuff;
      T * thisScratch = (T*)args->scratchbuff;
      int recvPeer = scclTB->recvpeer;
      int sendPeer = scclTB->sendpeer;

      const ssize_t size = args->coll.count;
      const ssize_t sizePerScclChunk = (size*sizeMultiplier)/scclAlgo->nchunksPerLoop;

      PRIMS_WRAPPER prims{args, tid, &recvPeer, &sendPeer, size, sizePerScclChunk, thisOutput, channel};

      const ssize_t loopSize = (ssize_t)prims.chunkSize;
      const size_t numScclChunks = prims.numScclChunks;
      uint32_t scclMaxAllowedCount = args->scclMaxAllowedCount;

      // sccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scclFlag* scclFlags = comm->scclAlgo.flags;

      //Loop over all chunks.
      for (ssize_t gridChunk = 0; gridChunk < numScclChunks; gridChunk++) {
        Offset chunkOffset = prims.initIter(sizePerScclChunk, gridChunk*loopSize);
        Offset srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        for (int i = 0; i < scclTB->nsteps; i++){
          struct scclTransfer* sccltran = &scclTB->transfers[i];
          // first wait if there is a dependence
          int8_t dependentBid = sccltran->dependentBid;
          int8_t dependentStep = sccltran->dependentStep;
          if (dependentBid >= 0){
              if (tid == sync_tid){
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, gridChunk, dependentStep);
              while ((scclFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
          }

          srcPointer = (sccltran->srcbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->srcbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          dstPointer = (sccltran->dstbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->dstbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          int count = sccltran->count;
          for (int c = 0; c < count; c += scclMaxAllowedCount) {
            srcoffset = prims.getOffset(chunkOffset, gridChunk, sccltran->srcoffset, c, sizePerScclChunk);
            dstoffset = prims.getOffset(chunkOffset, gridChunk, sccltran->dstoffset, c, sizePerScclChunk);
            int thisCount = min(scclMaxAllowedCount, count-c);
            switch (sccltran->type) {
              case SCCL_SEND:
                prims.send(srcPointer, srcoffset, dstoffset, thisCount);
                break;
              case SCCL_RECV:
                prims.recv(dstPointer, dstoffset, thisCount);
                break;
              case SCCL_RECV_COPY_SEND:
                prims.recvCopySend(dstPointer, dstoffset, thisCount);
                break;
              case SCCL_RECV_REDUCE_SEND:
                prims.recvReduceSend(srcPointer, srcoffset, thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY_SEND:
                prims.recvReduceCopySend(srcPointer, srcoffset, dstPointer, dstoffset, thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY:
                prims.recvReduceCopy(srcPointer, srcoffset, dstPointer, dstoffset, thisCount);
                break;
              case SCCL_REDUCE:
                prims.reduce(srcPointer, srcoffset, dstPointer, dstoffset, thisCount);
                break;
              case SCCL_LOCAL_COPY:
                prims.localCopy(srcPointer, srcoffset, dstPointer, dstoffset, thisCount);
                break;
              case SCCL_NO_OP:
                break;
              default:
                return;
            }
          }
          if (sccltran->has_dependence)
            __syncthreads();
          if (tid == sync_tid && sccltran->has_dependence){
            __threadfence();
            uint64_t curFlag = COMPUTE_FLAG(workIndex, gridChunk, i);
            scclFlags[bid].flag = curFlag;
          }
        }
      }
    }
};

template<class FUNC, typename T, int UNROLL>
struct SimpleWrapper {
  const int nthreads;
  const int stepSize;
  const int chunkSize;
  const size_t numScclChunks;
  int nelem;

  ncclPrimitives<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ SimpleWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, ssize_t size, ssize_t sizePerScclChunk, 
                           T * thisOutput, struct ncclChannel* channel)
    : nthreads(args->nThreads-WARP_SIZE),
      stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
      chunkSize(stepSize * SCCL_CHUNKSTEPS),
      numScclChunks(DIVUP(sizePerScclChunk, chunkSize)),
      prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    int realChunkSize = min(chunkSize, sizePerScclChunk-gridOffset);
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset;
    nelem = min(realChunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ ssize_t getOffset(size_t chunkOffset, ssize_t gridChunk, int sccltranOffset, int count, ssize_t sizePerScclChunk) {
    return chunkOffset + (ssize_t) (sccltranOffset+count) * sizePerScclChunk;
  }

  __device__ void send(T * srcPointer, ssize_t srcoffset, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    prims.directSend(srcChunkPointer, dstoffset, nelem*count);
  }

  __device__ void recv(T * dstPointer, ssize_t dstoffset, int count) {
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.directRecv(dstChunkPointer, dstoffset, nelem*count);
  }

  __device__ void recvCopySend(T * dstPointer, ssize_t dstoffset, int count) {
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.directRecvCopySend(dstChunkPointer, dstoffset, nelem*count);
  }
  
  __device__ void recvReduceSend(T * srcPointer, ssize_t srcoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    prims.recvReduceSend(srcChunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
  
  __device__ void recvReduceCopySend(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* dstChunkPointer = dstPointer + dstoffset;
    T* srcChunkPointer = srcPointer + srcoffset;
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionSimple : public scclFunction<T, ssize_t, SimpleWrapper<FUNC, T, UNROLL>> {};

#include "prims_ll128.h"
template<class FUNC, typename T>
struct LL128Wrapper {
  const int stepSize;
  ssize_t chunkSize;
  const ssize_t minChunkSize;
  const size_t numScclChunks;
  int nelem;

  ncclLL128Primitives<T, FUNC, 1, 1> prims;

  __device__ LL128Wrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, ssize_t size, ssize_t sizePerScclChunk, T * thisOutput, struct ncclChannel* channel)
    : stepSize(args->comm->buffSizes[NCCL_PROTO_LL128] / (sizeof(uint64_t)*NCCL_STEPS)),
      chunkSize(stepSize*NCCL_LL128_DATAELEMS*sizeof(uint64_t) / (NCCL_LL128_LINEELEMS*sizeof(T))),
      minChunkSize((NCCL_LL128_SHMEM_ELEMS_PER_THREAD*args->nThreads*NCCL_LL128_DATAELEMS*sizeof(uint64_t))/(NCCL_LL128_LINEELEMS*sizeof(T))/2),
      numScclChunks(DIVUP(sizePerScclChunk, chunkSize)),
      prims(tid, args->nThreads, recvPeer, sendPeer, stepSize, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    chunkSize = min(chunkSize, DIVUP(sizePerScclChunk-gridOffset,minChunkSize)*minChunkSize);
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ ssize_t getOffset(size_t chunkOffset, ssize_t gridChunk, int sccltranOffset, int count, ssize_t sizePerScclChunk) {
    return chunkOffset + (ssize_t) (sccltranOffset+count) * sizePerScclChunk;
  }

  __device__ void send(T * srcPointer, ssize_t srcoffset, ssize_t dstoffset, int count) {
    T* chunkPointer = srcPointer + srcoffset;
    prims.send(chunkPointer, nelem*count);
  }

  __device__ void recv(T * dstPointer, ssize_t dstoffset, int count) {
    T* chunkPointer = dstPointer + dstoffset;
    prims.recv(chunkPointer, nelem*count);
  }

  __device__ void recvCopySend(T * dstPointer, ssize_t dstoffset, int count) {
    T* chunkPointer = dstPointer + dstoffset;
    prims.recvCopySend(chunkPointer, nelem*count);
  }
  
  __device__ void recvReduceSend(T * srcPointer, ssize_t srcoffset, int count) {
    T* chunkPointer = srcPointer + srcoffset;
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
  
  __device__ void recvReduceCopySend(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* dstChunkPointer = dstPointer + dstoffset;
    T* srcChunkPointer = srcPointer + srcoffset;
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL128 : public scclFunction<T, ssize_t, LL128Wrapper<FUNC, T>> {};

template<class FUNC, typename T>
struct LLWrapper {
  const int stepLines;
  const ssize_t chunkSize;
  const size_t numScclChunks;
  int nelem;
  
  ncclLLPrimitives<T, FUNC, 1, 1> prims;

  __device__ LLWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, ssize_t size, ssize_t sizePerScclChunk, T * thisOutput, struct ncclChannel* channel)
    : stepLines(args->comm->buffSizes[NCCL_PROTO_LL] / (sizeof(union ncclLLFifoLine)*NCCL_STEPS)),
      chunkSize(stepLines * sizeof(uint64_t) / sizeof(T)),
      numScclChunks(DIVUP(sizePerScclChunk, chunkSize)),
      prims(tid, args->nThreads, recvPeer, sendPeer, stepLines, channel, args->comm) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    ssize_t chunkOffset = gridOffset;
    nelem = min(chunkSize, sizePerScclChunk-chunkOffset);
    return chunkOffset;
  }

  __device__ ssize_t getOffset(size_t chunkOffset, ssize_t gridChunk, int sccltranOffset, int count, ssize_t sizePerScclChunk) {
    return chunkOffset + (ssize_t) (sccltranOffset+count) * sizePerScclChunk;
  }

  __device__ void send(T * srcPointer, ssize_t srcoffset, ssize_t dstoffset, int count) {
    T* chunkPointer = srcPointer + srcoffset;
    prims.send(chunkPointer, nelem*count);
  }

  __device__ void recv(T * dstPointer, ssize_t dstoffset, int count) {
    T* chunkPointer = dstPointer + dstoffset;
    prims.recv(chunkPointer, nelem*count);
  }

  __device__ void recvCopySend(T * dstPointer, ssize_t dstoffset, int count) {
    T* chunkPointer = dstPointer + dstoffset;
    prims.recvCopySend(chunkPointer, nelem*count);
  }
  
  __device__ void recvReduceSend(T * srcPointer, ssize_t srcoffset, int count) {
    T* chunkPointer = srcPointer + srcoffset;
    prims.recvReduceSend(chunkPointer, nelem*count);
  }

  __device__ void recvReduceCopy(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.recvReduceCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
  
  __device__ void recvReduceCopySend(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* dstChunkPointer = dstPointer + dstoffset;
    T* srcChunkPointer = srcPointer + srcoffset;
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void reduce(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcPointer, ssize_t srcoffset, T * dstPointer, ssize_t dstoffset, int count) {
    T* srcChunkPointer = srcPointer + srcoffset;
    T* dstChunkPointer = dstPointer + dstoffset;
    prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL : public scclFunction<T, ssize_t, LLWrapper<FUNC, T>> {};
