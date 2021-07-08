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
    __device__ void run(struct ncclWorkElem* args) {
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

      PRIMS_WRAPPER prims{args, tid, &recvPeer, &sendPeer, thisOutput, channel, 1024, scclAlgo->chunkld};

      const int nranks = comm->nRanks;
      const ssize_t loopSize = (ssize_t)prims.chunkSize;
      const ssize_t size = args->coll.count;
      const ssize_t sizePerScclChunk = (size*nranks)/scclAlgo->nchunksPerLoop;
      uint32_t scclMaxAllowedCount = args->scclMaxAllowedCount;

      // sccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scclFlag* scclFlags = comm->scclAlgo.flags;

      for (ssize_t gridOffset = 0, iter = 0; gridOffset < sizePerScclChunk; gridOffset += loopSize, iter++) {
        size_t chunkOffset = prims.initIter(sizePerScclChunk, gridOffset);

        ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;
        for (int i = 0; i < scclTB->nsteps; i++){
          struct scclTransfer* sccltran = &scclTB->transfers[i];
          // first wait if there is a dependence
          int8_t dependentBid = sccltran->dependentBid;
          int8_t dependentStep = sccltran->dependentStep;
          if (sccltran->dependentBid >= 0){
              if (tid == sync_tid){
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((scclFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
          }

          srcPointer = (sccltran->srcbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->srcbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          dstPointer = (sccltran->dstbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->dstbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          int count = sccltran->count;

          for (int c = 0; c < count; c += scclMaxAllowedCount) {
            srcoffset = chunkOffset + (ssize_t) (sccltran->srcoffset+c) * sizePerScclChunk;
            dstoffset = chunkOffset + (ssize_t) (sccltran->dstoffset+c) * sizePerScclChunk;

            int thisCount = min(scclMaxAllowedCount, count-c);
            switch (sccltran->type) {
              case SCCL_SEND:
                prims.send(srcPointer + srcoffset, dstoffset, thisCount);
                break;
              case SCCL_RECV:
                prims.recv(dstPointer + dstoffset, dstoffset, thisCount);
                break;
              case SCCL_RECV_COPY_SEND:
                prims.recvCopySend(dstPointer + dstoffset, dstoffset, thisCount);
                break;
              case SCCL_RECV_REDUCE_SEND:
                prims.recvReduceSend(srcPointer + srcoffset, thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY_SEND:
                prims.recvReduceCopySend(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY:
                prims.recvReduceCopy(srcPointer + srcoffset, dstPointer + dstoffset, thisCount);
                break;
              case SCCL_NO_OP:
                break;
              default:
                return;
            }
          }
          if (tid == sync_tid && sccltran->has_dependence){
            __threadfence();
            uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
            scclFlags[bid].flag = curFlag;
          }
        }
      }
    }
};

//Represents a 2D chunk
template<typename T>
struct Block2D {
  ssize_t chunkStartRow;
  ssize_t chunkStartCol;
  int chunkRows;
  int chunkCols;

  __device__ __forceinline__ Block2D(const ssize_t size, const int chunkIdx, const int realChunkSize, const int numRealChunks, const int realChunkRows, const int realChunkCols, const int rows, const int ld) {
    // ssize_t rows = size/ld;
    chunkStartRow = chunkIdx / numRealChunks * realChunkRows;
    chunkStartCol = chunkIdx % numRealChunks * realChunkCols;
    int nelem = min((ssize_t)realChunkSize, (size - (chunkStartRow * ld + (rows - chunkStartRow) * (ld - (ld - chunkStartCol)))));
    chunkRows = min((ssize_t)min(nelem/realChunkCols, realChunkRows), rows - chunkStartRow);
    chunkCols = realChunkCols;
  }

  __device__ __forceinline__ Block2D() :
    chunkStartRow(-1), chunkStartCol(-1), chunkRows(-1), chunkCols(-1)
  {}
  __device__ __forceinline__
  bool isValid() const {return chunkStartCol >= 0 && chunkStartRow >= 0 && chunkRows > 0 && chunkCols > 0;}
  __device__ __forceinline__ 
  int nelem() const {return chunkRows * chunkCols;}
};

template<typename T, typename PRIMS_WRAPPER>
class scclFunction2D {
  public:
    __device__ void run(struct ncclWorkElem* args) {
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

      const int nranks = comm->nRanks;
      const ssize_t size = args->coll.count;
      const int ld = args->ld;
      const int rows = (size * nranks)/ld;
      int chunkld = scclAlgo->chunkld;
      int nchunksPerLoop = scclAlgo->nchunksPerLoop;
      // if (threadIdx.x == 0) {
      //   printf("ld %ld\n", ld);
      // }
      PRIMS_WRAPPER prims{args, tid, &recvPeer, &sendPeer, thisOutput, channel, ld, rows, chunkld, nchunksPerLoop};
      
      const ssize_t sizePerScclChunk = (ssize_t)prims.chunkSize;
      const ssize_t loopSize = sizePerScclChunk * nchunksPerLoop;
      uint32_t scclMaxAllowedCount = args->scclMaxAllowedCount;
      // sccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scclFlag* scclFlags = comm->scclAlgo.flags;

      auto chunkSize = prims.chunkSize;
      auto numRealChunks = prims.numRealChunks;
      auto chunkRows = prims.chunkRows;
      auto realChunkCols = prims.realChunkCols;

      int srcGridChunkIdx = 0;
      int dstGridChunkIdx = 0;
      const int total2DChunks = DIVUP(rows, chunkRows)*(ld/chunkld);
      // if (threadIdx.x == 0) {
      //   printf("loopSize %ld total2DChunks %ld sizePerScclChunk %ld\n", loopSize, total2DChunks, sizePerScclChunk);
      // }
      int iter;

      for (iter = 0, srcGridChunkIdx = 0, dstGridChunkIdx = 0; srcGridChunkIdx < total2DChunks && dstGridChunkIdx < total2DChunks; 
           srcGridChunkIdx += nchunksPerLoop, dstGridChunkIdx += nchunksPerLoop, iter++) {
        
        // if (threadIdx.x == 0)
        //   printf("realChunkSize %d loopSize %ld gridOffsetStartRow %ld gridOffset %ld\n", prims.realChunkSize, loopSize, gridOffsetStartRow, gridOffset);

        // ssize_t srcoffset, dstoffset;
        T* srcPointer, * dstPointer;

        for (int i = 0; i < scclTB->nsteps; i++){
          struct scclTransfer* sccltran = &scclTB->transfers[i];
          // first wait if there is a dependence
          int8_t dependentBid = sccltran->dependentBid;
          int8_t dependentStep = sccltran->dependentStep;
          if (sccltran->dependentBid >= 0){
              if (tid == sync_tid){
              uint64_t goalFlag = COMPUTE_FLAG(workIndex, iter, dependentStep);
              while ((scclFlags + dependentBid)->flag < goalFlag){};
              }
              __syncthreads();
          }

          srcPointer = (sccltran->srcbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->srcbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          dstPointer = (sccltran->dstbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->dstbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          int count = sccltran->count;

          for (int c = 0; c < count; c += scclMaxAllowedCount) {           
            int dstChunkIdx = dstGridChunkIdx + sccltran->dstoffset + c;
            int srcChunkIdx = srcGridChunkIdx + sccltran->srcoffset + c;
           
            const Block2D<T> srcChunk2D = Block2D<T>(size*nranks, srcChunkIdx, chunkSize, numRealChunks, chunkRows, realChunkCols, rows, ld);
            const Block2D<T> dstChunk2D = Block2D<T>(size*nranks, dstChunkIdx, chunkSize, numRealChunks, chunkRows, realChunkCols, rows, ld);
            
            int thisCount = min(scclMaxAllowedCount, count-c);
            switch (sccltran->type) {
              case SCCL_SEND:
                prims.send(i, srcPointer, srcChunk2D, thisCount);
                break;
              case SCCL_RECV:
                prims.recv(i, dstPointer, dstChunk2D, thisCount);
                break;
              case SCCL_RECV_COPY_SEND:
                prims.recvCopySend(i, dstPointer, dstChunk2D, thisCount);
                break;
              case SCCL_RECV_REDUCE_SEND:
                prims.recvReduceSend(i, srcPointer, srcChunk2D, thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY_SEND:
                prims.recvReduceCopySend(i, srcPointer, dstPointer, srcChunk2D, dstChunk2D, thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY:
                prims.recvReduceCopy(i, srcPointer, dstPointer, srcChunk2D, dstChunk2D, thisCount);
                break;
              case SCCL_NO_OP:
                break;
              default:
                return;
            }
          }
          if (tid == sync_tid && sccltran->has_dependence){
            __threadfence();
            uint64_t curFlag = COMPUTE_FLAG(workIndex, iter, i);
            scclFlags[bid].flag = curFlag;
          }
        }
      }
    }
};

// template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
// class ncclPrimitives2D : public ncclPrimitives<UNROLL, SLICESPERCHUNK, SLICESTEPS, T, NRECV, NSEND, DIRECT, FUNC> {
// protected:
//   const Block2D<T> invalidBlock;

//   template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
//   inline __device__ void
//   GenericOp(const T* srcPtr, T* dstPtr, const Block2D<T>& srcBlock, const Block2D<T>& dstBlock, int nelem, ssize_t directOffset) {
//     int offset = 0;
//     int sliceSize = this->stepSize*SLICESTEPS;
//     int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);
//     #pragma unroll
//     for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
//       int realSize = max(0, min(dataSize, nelem-offset));
//       if (this->tid < this->nworkers) {
//         if (SRC && (this->role & ROLE_SRC)) this->srcs[0] = srcPtr;//+offset;
//         if (RECV && (this->role & ROLE_WAIT_RECV)) this->waitRecv<SRC, DIRECTRECV>(directOffset+offset);
//         if (DST && (this->role & ROLE_DST)) this->dsts[0] = dstPtr;//+offset;
//         if (SEND && (this->role & ROLE_WAIT_SEND)) this->waitSend<DST, DIRECTSEND>(directOffset+offset, realSize*sizeof(T));
//         if (realSize > 0) {
//           this->subBarrier();
//           if (DIRECTRECV && this->srcs[0] == this->dsts[0]) {
//             // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
//             if (SEND) {
//               // (1-SEND) is only there to avoid compilation errors in case NSEND=0 (and SEND=0).
//             //  ReduceOrCopyMulti<UNROLL, FUNC, T, 1, 1, 1, (1-SEND)+NSEND>(this->tid, this->nworkers, 1, this->srcs, this->nsend, this->dsts+1, realSize);
//             }
//           } else {
//             ReduceOrCopyMulti2D<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, SRC, DST, Block2D<T>>(this->tid, this->nworkers, RECV*this->nrecv+SRC, this->srcs, SEND*this->nsend+DST, this->dsts, 
//             offset, srcBlock, dstBlock, matrixRows, matrixCols, realSize);
//           }
//         }
//       }
//       this->barrier();
//       if (SEND && (this->role & ROLE_POST_SEND) && realSize > 0 && this->index == 0) __threadfence_system();
//       __syncwarp();
//       if (SEND && (this->role & ROLE_POST_SEND)) this->postSend();
//       if (RECV && (this->role & ROLE_POST_RECV)) this->postRecv();
//       offset += realSize;
//     }
//   }

// public:
//   size_t matrixRows, matrixCols;
//   __device__ __forceinline__
//   ncclPrimitives2D(const int tid, const int nworkers, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, struct ncclShmemPtrs* ptrs, int group):
//     invalidBlock(), ncclPrimitives<UNROLL, SLICESPERCHUNK, SLICESTEPS, T, NRECV, NSEND, DIRECT, FUNC>(tid, nworkers, recvPeers, sendPeers, directBuff, stepSize, channel, comm, ptrs, group)
//   {}
  
//   __device__ __forceinline__ void
//   send(const T* src, const Block2D<T>& srcBlock, int offset, int nelem) {
//     GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, srcBlock, invalidBlock, nelem, offset);
//   }

//   __device__ __forceinline__ void
//   recv(T* dst, const Block2D<T>& dstBlock, int offset, int nelem) {
//     GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, invalidBlock, dstBlock, nelem, offset);
//   }

//   __device__ __forceinline__ void
//   recvCopySend(T* dst, const Block2D<T>& dstBlock, int offset, int nelem) {
//     GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, invalidBlock, dstBlock, nelem, offset);
//   }

//   __device__ __forceinline__ void
//   recvReduceCopy(const T* src, T* dst, const Block2D<T>& srcBlock, const Block2D<T>& dstBlock, int offset, int nelem) {
//     GenericOp<0, 0, 1, 0, 1, 1>(src, dst, srcBlock, dstBlock, nelem, offset);
//   }

//   __device__ __forceinline__ void
//   recvReduceSend(const T* src, const Block2D<T>& srcBlock, int offset, int nelem) {
//     GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, srcBlock, invalidBlock, nelem, offset);
//   }

//   __device__ __forceinline__ void
//   recvReduceCopySend(const T* src, T* dst, const Block2D<T>& srcBlock, const Block2D<T>& dstBlock, int offset, int nelem) {
//     GenericOp<0, 0, 1, 1, 1, 1>(src, dst, srcBlock, dstBlock, nelem, offset);
//   }
// };

template<class FUNC, typename T, int UNROLL>
struct SimpleWrapper2D {
  const int nthreads;
  const int stepSize;
  int chunkSize;
  int nelem;
  int chunkld;
  int realChunkCols;
  int ld;
  int realChunkRows;
  int numRealChunks;
  int realChunkSize;
  int rows;
  int rank;
  int chunkRows;

  ncclPrimitives<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ __forceinline__ SimpleWrapper2D(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel,
                           int ld, int rows, int chunkld, int nchunksPerLoop)
    : nthreads(args->nThreads-WARP_SIZE),
      stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
      rank(args->comm->rank),
      ld(ld), chunkld(chunkld), realChunkCols(chunkld),
      prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {
        prims.matrixRows = rows;
        prims.matrixCols = ld;
        //Align chunk size to the number of columns.
        int _chunkSize = min((size_t)stepSize * SCCL_CHUNKSTEPS, DIVUP(((size_t)ld)*((size_t)rows),nchunksPerLoop));
        int __chunkSize = _chunkSize;
        ALIGN_SIZE(__chunkSize, ld);
        //chunkSize should not have more than 'matrixRows' rows.
        chunkRows = min((__chunkSize/chunkld), (int)rows);
        chunkSize = chunkRows * chunkld;
        // if (threadIdx.x == 0) {
        //   printf("[%d, %d] chunkSize %d %d %d\n", args->comm->rank, blockIdx.x, _chunkSize, __chunkSize, chunkSize);
        // }
        numRealChunks = ld/realChunkCols;
      }

  __device__ __forceinline__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    realChunkSize = min(chunkSize, sizePerScclChunk-gridOffset);
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset;
    nelem = min(realChunkSize, sizePerScclChunk-chunkOffset);
    realChunkRows = realChunkSize/realChunkCols;
    

    return chunkOffset;
  }

  const bool toPrint = false;
  __device__ __forceinline__ void send(int step, T * src, const Block2D<T>& srcBlock, int count) {
    //assert(srcBlock.isValid());
    // assert(srcBlock.nelem() == 128*1024);
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, srcBlock.nelem(), srcBlock.chunkStartRow, srcBlock.chunkStartCol, srcBlock.chunkRows, srcBlock.chunkCols);
    }
    prims.send(src, srcBlock, 0, srcBlock.nelem()*count);
  }

  __device__ __forceinline__ void recv(int step, T * dst, const Block2D<T>& dstBlock, int count) {
    //assert(dstBlock.isValid());
    // assert(dstBlock.nelem() == 128*1024);
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, dstBlock.nelem(), dstBlock.chunkStartRow, dstBlock.chunkStartCol, dstBlock.chunkRows, dstBlock.chunkCols);
    }
    prims.recv(dst, dstBlock, 0, dstBlock.nelem()*count);
  }

  __device__ __forceinline__ void recvCopySend(int step, T * dst, const Block2D<T>& dstBlock, int count) {
    //assert(dstBlock.isValid());
    // assert(dstBlock.nelem() == 128*1024);
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, dstBlock.nelem(), dstBlock.chunkStartRow, dstBlock.chunkStartCol, dstBlock.chunkRows, dstBlock.chunkCols);
    }
    prims.recvCopySend(dst, dstBlock, 0, dstBlock.nelem()*count);
  }
  
  __device__ __forceinline__ void recvReduceSend(int step, T * src, const Block2D<T>& srcBlock, int count) {
    //assert(srcBlock.isValid());
    // assert(srcBlock.nelem() == 128*1024);
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, srcBlock.nelem(), srcBlock.chunkStartRow, srcBlock.chunkStartCol, srcBlock.chunkRows, srcBlock.chunkCols);
    }
    prims.recvReduceSend(src, srcBlock, 0, srcBlock.nelem()*count);
  }

  __device__ __forceinline__ void recvReduceCopy(int step, T * src, T * dst, const Block2D<T>& srcBlock, const Block2D<T>& dstBlock, int count) {
    //assert(dstBlock.isValid()); assert(srcBlock.isValid());
    // assert(srcBlock.nelem() == 128*1024);
    // assert(dstBlock.nelem() == 128*1024);
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, dstBlock.nelem(), dstBlock.chunkStartRow, dstBlock.chunkStartCol, dstBlock.chunkRows, dstBlock.chunkCols);
    }
    prims.recvReduceCopy(src, dst, srcBlock, dstBlock, 0, dstBlock.nelem()*count);
  }
  
  __device__ __forceinline__ void recvReduceCopySend(int step, T * src, T * dst, const Block2D<T>& srcBlock, const Block2D<T>& dstBlock, int count) {
    //assert(dstBlock.isValid()); assert(srcBlock.isValid());
    // assert(srcBlock.nelem() == 128*1024);
    // assert(dstBlock.nelem() == 128*1024);
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, dstBlock.nelem(), dstBlock.chunkStartRow, dstBlock.chunkStartCol, dstBlock.chunkRows, dstBlock.chunkCols);
    }
    prims.recvReduceCopySend(src, dst, srcBlock, dstBlock, 0, dstBlock.nelem()*count);
  }
};

template<class FUNC, typename T, int UNROLL>
struct SimpleWrapper {
  const int nthreads;
  const int stepSize;
  const int chunkSize;
  int nelem;
  int chunkld;
  int realChunkCols;
  int ld;
  int realChunkRows;
  int numRealChunks;
  int realChunkSize;

  ncclPrimitives<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ SimpleWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel,
                           int ld, int chunkld)
    : nthreads(args->nThreads-WARP_SIZE),
      stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
      chunkSize(stepSize * SCCL_CHUNKSTEPS),
      ld(ld), chunkld(chunkld), realChunkCols(chunkld),
      prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {}

  __device__ size_t initIter(ssize_t sizePerScclChunk, ssize_t gridOffset) {
    realChunkSize = min(chunkSize, sizePerScclChunk-gridOffset);
    ALIGN_SIZE(realChunkSize, nthreads*sizeof(uint64_t)/sizeof(T));
    ssize_t chunkOffset = gridOffset;
    nelem = min(realChunkSize, sizePerScclChunk-chunkOffset);
    realChunkRows = realChunkSize/realChunkCols;
    numRealChunks = ld/realChunkCols;

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
  
  __device__ void recvReduceCopySend(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
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
  int chunkld;
  int ld;
  int realChunkRows;
  int realChunkCols;
  int numRealChunks;
  int realChunkSize;
  ncclLL128Primitives<T, FUNC, 1, 1> prims;

  __device__ LL128Wrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel, int ld, int chunkld)
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

  __device__ void recvReduceCopySend(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL128 : public scclFunction<T, LL128Wrapper<FUNC, T>> {};

template<class FUNC, typename T>
struct LLWrapper {
  const int stepLines;
  const ssize_t chunkSize;
  int nelem;
  int chunkld;
  int ld;
  int realChunkRows;
  int realChunkCols;
  int numRealChunks;
  int realChunkSize;
  ncclLLPrimitives<T, FUNC, 1, 1> prims;

  __device__ LLWrapper(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, T * thisOutput, struct ncclChannel* channel, int ld, int chunkld)
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
  
  __device__ void recvReduceCopySend(T * srcChunkPointer, T * dstChunkPointer, int count) {
    prims.recvReduceCopySend(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunctionLL : public scclFunction<T, LLWrapper<FUNC, T>> {};