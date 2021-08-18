/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sccl_interpreter.h"

//Support for Custom Collectives operations on 2D arrays
 
template<typename T, typename PRIMS_WRAPPER, bool DO_SYNC>
class scclFunction2D {
  private:
    __device__ __forceinline__ void initBlock2D(Block2D& block, const ssize_t size, const int chunkIdx, const int chunkSize, const int numChunks, 
                                      const int chunkRows, const int cols, const int rows, const int ld) {
      block.startRow = chunkIdx / numChunks * chunkRows;
      block.startCol = chunkIdx % numChunks * cols;
      int nelem = MIN(chunkSize, (int)(size - (block.startRow * ld + (rows - block.startRow) * (ld - (ld - block.startCol)))));
      block.rows = MIN(MIN(nelem/cols, chunkRows), rows - block.startRow);
      block.cols = cols;
    }

    __device__ __forceinline__ int nelemBlock2D(Block2D& block) {
      return block.rows * block.cols;
    }
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
      T * __restrict__ thisInput = (T*)args->sendbuff;
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

      const ssize_t sizePerScclChunk = (size*sizeMultiplier)/scclAlgo->nchunksPerLoop; //size*nranks = 8192 * 3072; nchunksperloop = 16 * 12 or 16 * 24 = 384
      const int rowsPerScclChunk = sizePerScclChunk/ld;

      PRIMS_WRAPPER prims{args, tid, &recvPeer, &sendPeer, thisOutput, channel, ld, rows, chunkld, nchunksPerLoop, rows, scclAlgo};
      
      uint32_t scclMaxAllowedCount = args->scclMaxAllowedCount;
      // sccl flags all start out with 0. this is used as a part of the flag to make sure different work items deal with different synchronization flags
      // this still needs more work. when we make a way around the queue, the flag might have been set to undesired values. will be fixed in subsequent versions.
      const int workIndex = args->index+1;
      volatile struct scclFlag* scclFlags = comm->scclAlgo.flags;

      auto chunkSize = prims.chunkSize;
      auto numChunks = prims.numRealChunks;
      auto chunkRows = prims.chunkRows;
      auto cols = chunkld;

      int gridChunkIdx = 0;
      const int numTotalChunks = (rows/chunkRows * ld/chunkld);
      const int numScclChunks2D = numTotalChunks/scclAlgo->nchunksPerLoop;
    
      assert(numTotalChunks % scclAlgo->nchunksPerLoop == 0);
      int iter = 0;

      for (gridChunkIdx = 0; gridChunkIdx < numScclChunks2D; gridChunkIdx++) {
        T* srcPointer, * dstPointer;

        for (int i = 0; i < scclTB->nsteps; i++){
          prims.initIter(iter, i);
          struct scclTransfer* sccltran = &scclTB->transfers[i];
          // first wait if there is a dependence
          int8_t dependentBid = sccltran->dependentBid;
          const int8_t dependentStep = sccltran->dependentStep;
          const int flagsPerBlock = scclAlgo->flagsPerBlock;

          if (DO_SYNC && dependentBid >= 0){
              // if (tid == sync_tid){
              //   uint64_t goalFlag = COMPUTE_FLAG(workIndex);
              //   // printf("index %d gloalFlag %ld\n", dependentBid * flagsPerBlock + iter, goalFlag);
              //   while (scclFlags[dependentBid * flagsPerBlock + iter].flag < goalFlag){};
              //   // printf("197:");
              // }
              __syncthreads();
          }

          srcPointer = (sccltran->srcbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->srcbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          dstPointer = (sccltran->dstbuffer == SCCL_INPUT_BUFFER) ? thisInput : ((sccltran->dstbuffer == SCCL_OUTPUT_BUFFER) ? thisOutput : thisScratch);
          
          int count = sccltran->count;

          for (int c = 0; c < count; c += scclMaxAllowedCount) {     
            int dstChunkIdx = gridChunkIdx + (sccltran->dstoffset + c)*numScclChunks2D;
            int srcChunkIdx = gridChunkIdx + (sccltran->srcoffset + c)*numScclChunks2D;
            
            int thisCount = min(scclMaxAllowedCount, count-c);
            
            Block2D srcBlock, dstBlock;

            initBlock2D(srcBlock, size*nranks, srcChunkIdx, chunkSize, numChunks, chunkRows, cols, rows, ld);
            initBlock2D(dstBlock, size*nranks, dstChunkIdx, chunkSize, numChunks, chunkRows, cols, rows, ld);

            switch (sccltran->type) {
              case SCCL_SEND:
              // if (threadIdx.x == 0)printf("%d rank %d flagsPerBlock %d iter %d (%d, %d):(%dx%d) index %d , srcPointer[13370240] %f\n", bid, comm->rank, flagsPerBlock, iter, srcBlock.startRow, srcBlock.startCol, srcBlock.rows, srcBlock.cols, dependentBid * flagsPerBlock + iter, 
              //   !DO_SYNC ? 0 : (float)((half*)srcPointer)[13370240]);
                prims.send(i, srcPointer, &srcBlock, nelemBlock2D(srcBlock)*thisCount);
                break;
              case SCCL_RECV:
                prims.recv(i, dstPointer, &dstBlock, nelemBlock2D(dstBlock)*thisCount);
                break;
              case SCCL_RECV_COPY_SEND:
                prims.recvCopySend(i, dstPointer, &dstBlock, nelemBlock2D(dstBlock)*thisCount);
                break;
              case SCCL_RECV_REDUCE_SEND:
              // if (threadIdx.x == 0)printf("%d rank %d flagsPerBlock %d iter %d (%d, %d):(%dx%d) index %d\n", bid, comm->rank, flagsPerBlock, iter, srcBlock.startRow, srcBlock.startCol, srcBlock.rows, srcBlock.cols, dependentBid * flagsPerBlock + iter);
                prims.recvReduceSend(i, srcPointer, &srcBlock, nelemBlock2D(srcBlock)*thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY_SEND:
              // if (threadIdx.x == 0)printf("rrcs: %d rank %d flagsPerBlock %d iter %d (%d, %d):(%dx%d) index %d srcPointer[13370240] %f\n",bid,  comm->rank, flagsPerBlock, iter, srcBlock.startRow, srcBlock.startCol, srcBlock.rows, srcBlock.cols, dependentBid * flagsPerBlock + iter, 
              //   !DO_SYNC ? 0 : (float)((half*)srcPointer)[13370240]);
                prims.recvReduceCopySend(i, srcPointer, dstPointer, &srcBlock, &dstBlock, nelemBlock2D(srcBlock)*thisCount);
                break;
              case SCCL_RECV_REDUCE_COPY: 
              // if (threadIdx.x == 0)printf("%d rank %d flagsPerBlock %d iter %d (%d, %d):(%dx%d) index %d\n",bid,  comm->rank, flagsPerBlock, iter, srcBlock.startRow, srcBlock.startCol, srcBlock.rows, srcBlock.cols, dependentBid * flagsPerBlock + iter);
                prims.recvReduceCopy(i, srcPointer, dstPointer, &srcBlock, &dstBlock, nelemBlock2D(srcBlock)*thisCount);
                break;
              case SCCL_NO_OP:
                break;
              default:
                return;
            }
          }
          if (DO_SYNC && tid == sync_tid && sccltran->has_dependence){
            __threadfence();
            // uint64_t curFlag = COMPUTE_FLAG(workIndex);
            // scclFlags[COMPUTE_FLAG_INDEX(dependentBid, iter, dependentStep)].flag = curFlag;
          }
          iter++;
        }
      }
    }
};

template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives2D : public ncclPrimitives<UNROLL, SLICESPERCHUNK, SLICESTEPS, T, NRECV, NSEND, DIRECT, FUNC> {
protected:
  const Block2D* invalidBlock = nullptr;

  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp(const T* srcPtr, T* dstPtr, const Block2D* srcBlock, const Block2D* dstBlock, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = this->stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);
    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (SRC) {
        auto _realSize = (realSize / srcBlock->cols) * srcBlock->cols;
        // if (threadIdx.x == 0)
        //   printf("realSize %d _realSize %d %d\n", realSize, _realSize, srcBlock->cols);
        realSize = _realSize;
      } else if (DST) {
        auto _realSize = (realSize / dstBlock->cols) * dstBlock->cols;
        // if (threadIdx.x == 0)
        //   printf("realSize %d _realSize %d %d\n", realSize, _realSize, dstBlock->cols);
        realSize = _realSize;
      }
      // else if (DST) {
      //   realSize = (realSize / srcBlock->cols) * srcBlock->cols;
      // }
      if (this->tid < this->nworkers) {
        if (SRC && (this->role & ROLE_SRC)) this->srcs[0] = srcPtr;//+offset;
        if (RECV && (this->role & ROLE_WAIT_RECV)) this->waitRecv<SRC, DIRECTRECV>(directOffset+offset);
        if (DST && (this->role & ROLE_DST)) this->dsts[0] = dstPtr;//+offset;
        if (SEND && (this->role & ROLE_WAIT_SEND)) this->waitSend<DST, DIRECTSEND>(directOffset+offset, realSize*sizeof(T));
        if (realSize > 0) {
          this->subBarrier();
          if (DIRECTRECV && this->srcs[0] == this->dsts[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              // (1-SEND) is only there to avoid compilation errors in case NSEND=0 (and SEND=0).
              // ReduceOrCopyMulti2D<UNROLL, FUNC, T, 1, 1, 1, (1-SEND)+NSEND, SRC, DST, Block2D>(this->tid, this->nworkers, 1, this->srcs, this->nsend, this->dsts+1, offset, srcBlock, dstBlock, matrixRows, matrixCols, realSize);
            }
          } else {
            ReduceOrCopyMulti2D<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, SRC, DST, Block2D>(this->tid, this->nworkers, RECV*this->nrecv+SRC, this->srcs, SEND*this->nsend+DST, this->dsts, 
                                                                                                                        offset, srcBlock, dstBlock, matrixRows, matrixCols, realSize);
          }
        }
      }
      this->barrier();
      if (SEND && (this->role & ROLE_POST_SEND) && realSize > 0 && this->index == 0) __threadfence_system();
      __syncwarp();
      if (SEND && (this->role & ROLE_POST_SEND)) this->postSend();
      if (RECV && (this->role & ROLE_POST_RECV)) this->postRecv();
      offset += realSize;
    }
  }

public:
  size_t matrixRows, matrixCols;
  __device__ __forceinline__
  ncclPrimitives2D(const int tid, const int nworkers, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, struct ncclShmemPtrs* ptrs, int group):
    ncclPrimitives<UNROLL, SLICESPERCHUNK, SLICESTEPS, T, NRECV, NSEND, DIRECT, FUNC>(tid, nworkers, recvPeers, sendPeers, directBuff, stepSize, channel, comm, ptrs, group)
  {}
  
  __device__ __forceinline__ void
  send(const T* src, const Block2D* srcBlock, int offset, int nelem) {
    GenericOp<0, 0, 0, 1, 1, 0>(src, NULL, srcBlock, invalidBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recv(T* dst, const Block2D* dstBlock, int offset, int nelem) {
    GenericOp<0, 0, 1, 0, 0, 1>(NULL, dst, invalidBlock, dstBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, const Block2D* dstBlock, int offset, int nelem) {
    GenericOp<0, 0, 1, 1, 0, 1>(NULL, dst, invalidBlock, dstBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, const Block2D* srcBlock, const Block2D* dstBlock, int offset, int nelem) {
    GenericOp<0, 0, 1, 0, 1, 1>(src, dst, srcBlock, dstBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, const Block2D* srcBlock, int offset, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 0>(src, NULL, srcBlock, invalidBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, const Block2D* srcBlock, const Block2D* dstBlock, int offset, int nelem) {
    GenericOp<0, 0, 1, 1, 1, 1>(src, dst, srcBlock, dstBlock, nelem, offset);
  }
};

template<class FUNC, typename T, int UNROLL>
struct SimpleWrapper2D {
  const int nthreads;
  const int stepSize;
  int chunkSize;
  int numRealChunks;
  int rank;
  int chunkRows;
  int chunkld;
  ssize_t numScclChunks;
  
  ncclPrimitives2D<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ __forceinline__ void initBlock2D(Block2D& block, const ssize_t size, const int chunkIdx, const int chunkSize, const int numChunks, 
                                      const int chunkRows, const int cols, const int rows, const int ld) {
    block.startRow = chunkIdx / numChunks * chunkRows;
    block.startCol = chunkIdx % numChunks * cols;
    int nelem = MIN(chunkSize, (int)(size - (block.startRow * ld + (rows - block.startRow) * (ld - (ld - block.startCol)))));
    block.rows = MIN(MIN(nelem/cols, chunkRows), rows - block.startRow);
    block.cols = cols;
  }

  __device__ __forceinline__ int nelemBlock2D(const Block2D& block) {
    return block.rows * block.cols;
  }

  __device__ __forceinline__ SimpleWrapper2D(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, ssize_t size, ssize_t sizePerScclChunk,
                                             T * thisOutput, struct ncclChannel* channel)
    : nthreads(args->nThreads-WARP_SIZE),
      stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
      rank(args->comm->rank),
      prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0) {
        struct scclAlgorithm* scclAlgo = &args->comm->scclAlgo;
        int ld = args->ld;
        int rows = size/ld;  
        chunkld = scclAlgo->chunkld;
        int nchunksPerLoop = scclAlgo->nchunksPerLoop;
        prims.matrixRows = rows;
        prims.matrixCols = ld;

        //Align chunk size to the number of columns.
        const int maxChunkSize = stepSize * SCCL_CHUNKSTEPS;
        chunkSize = min(maxChunkSize, DIVUP((ld * rows), nchunksPerLoop));
        if (ROUNDUP(chunkSize, ld) < maxChunkSize) {
          //If possible increase chunkSize to align with chunk cols
          chunkSize = ALIGN_SIZE(chunkSize, ld);
        } else {
          //Otherwise decrease
          chunkSize = ALIGN_DOWN(chunkSize, ld);
        }

        //chunkSize should not have more than 'matrixRows' rows.
        chunkRows = min((chunkSize/chunkld), (int)rows);
        //Make chunkRows a perfect divisor of matrixRows;
        for (; chunkRows >= 1; chunkRows--) {
          if (rows % chunkRows == 0) {
            break;
          }
        }

        // if (threadIdx.x == 0 && blockIdx.x == 0 && rank == 0) {
        //   printf("chunkRows %d chunkld  %d maxChunkSize %d\n", chunkRows, chunkld, maxChunkSize);
        // }

        chunkSize = chunkRows * chunkld;
        numRealChunks = ld/chunkld;
        const int numTotalChunks = (rows/chunkRows * ld/chunkld);
        //TODO: Instead of division, DIVUP might also work
        numScclChunks = numTotalChunks/scclAlgo->nchunksPerLoop;
        assert(numTotalChunks % scclAlgo->nchunksPerLoop == 0);
      }


  __device__ __forceinline__ Block2D initIter(int iter, int step) {
    //Does nothing here
    return Block2D{-1,-1,-1,-1};
  }

  __device__ Block2D getOffset(Block2D chunkOffset, ssize_t gridChunk, int sccltranOffset, int count, ssize_t sizePerScclChunk) {
    Block2D block;
    int srcChunkIdx = gridChunk + (sccltranOffset + count)*numScclChunks;
    initBlock2D(block, prims.matrixRows*prims.matrixCols, srcChunkIdx, chunkSize, numRealChunks, chunkRows, chunkld, prims.matrixRows, prims.matrixCols);
    return block;
  }

  const bool toPrint = false;
  __device__ __forceinline__ void send(T * src, const Block2D& srcBlock, const Block2D& dstBlock, int count) {
    if (toPrint && threadIdx.x == 0 && blockIdx.x == 0) {
      // printf("%d [%d, %d] step %d [%d, %d]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, srcBlock->startRow, srcBlock->startCol, srcBlock->rows, srcBlock->cols);
    }
    int nelem = nelemBlock2D(srcBlock);
    prims.send(src, &srcBlock, 0, nelem*count);
  }

  __device__ __forceinline__ void recv(T * dst, const Block2D& dstBlock, int count) {
    if (toPrint && threadIdx.x == 0 && blockIdx.x == 0) {
      // printf("%d [%d, %d] step %d  [%d, %d]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, dstBlock->startRow, dstBlock->startCol, dstBlock->rows, dstBlock->cols);
    }
    int nelem = nelemBlock2D(dstBlock);
    prims.recv(dst, &dstBlock, 0, nelem*count);
  }

  __device__ __forceinline__ void recvCopySend(T * dst, const Block2D& dstBlock, int count) {
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      // printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, dstBlock.nelem(), dstBlock.startRow, dstBlock.startCol, dstBlock.chunkRows, dstBlock.cols);
    }
    int nelem = nelemBlock2D(dstBlock);
    prims.recvCopySend(dst, &dstBlock, 0, nelem*count);
  }
  
  __device__ __forceinline__ void recvReduceSend(T * src, const Block2D& srcBlock, int count) {
    if (toPrint && threadIdx.x == 0 && blockIdx.x == 0) {
      // printf("%d [%d, %d] step %d  [%d, %d]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, srcBlock->startRow, srcBlock->startCol, srcBlock->rows, srcBlock->cols);
    }
    int nelem = nelemBlock2D(srcBlock);
    prims.recvReduceSend(src, &srcBlock, 0, nelem*count);
  }

  __device__ __forceinline__ void recvReduceCopy(T * src, const Block2D& srcBlock, T * dst, const Block2D& dstBlock, int count) {
    if (toPrint && threadIdx.x == 0 && blockIdx.x == 0) {
      //  printf("%d [%d, %d] step %d  src: [%d, %d]; [%d, %d] nelem %d, dst: [%d, %d]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, srcBlock->startRow, srcBlock->startCol, srcBlock->rows, srcBlock->cols,
      //   dstBlock->startRow, dstBlock->startCol, dstBlock->rows, dstBlock->cols);
    }
    int nelem = nelemBlock2D(srcBlock);
    prims.recvReduceCopy(src, dst, &srcBlock, &dstBlock, 0, nelem*count);
  }
  
  __device__ __forceinline__ void recvReduceCopySend(T * src, const Block2D& srcBlock, T * dst, const Block2D& dstBlock, int count) {
    if (toPrint && threadIdx.x == 0 && rank == 0 && blockIdx.x == 0) {
      // printf("%d [%d, %d] step %d nelem %d, [%ld, %ld]; [%d, %d] \n", __LINE__, rank, blockIdx.x, step, dstBlock.nelem(), dstBlock.startRow, dstBlock.startCol, dstBlock.chunkRows, dstBlock.cols);
    }
    int nelem = nelemBlock2D(srcBlock);
    prims.recvReduceCopySend(src, dst, &srcBlock, &dstBlock, 0, nelem*count);
  }

  __device__ void reduce(T * srcPointer, const Block2D& srcoffset, T * dstPointer, const Block2D& dstoffset, int count) {
    // T* srcChunkPointer = srcPointer + srcoffset;
    // T* dstChunkPointer = dstPointer + dstoffset;
    // prims.reduce(srcChunkPointer, dstChunkPointer, nelem*count);
  }

  __device__ void localCopy(T * srcPointer, const Block2D& srcoffset, T * dstPointer, const Block2D& dstoffset, int count) {
    // T* srcChunkPointer = srcPointer + srcoffset;
    // T* dstChunkPointer = dstPointer + dstoffset;
    // prims.localCopy(srcChunkPointer, dstChunkPointer, nelem*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunction2DSimple : public scclFunction<T, Block2D, SimpleWrapper2D<FUNC, T, UNROLL>> {};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective2D, NCCL_ALGO_SCCL, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      scclFunction2DSimple<FUNC, T, UNROLL> scclfunc;
      scclfunc.run(args, 1);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective2D, NCCL_ALGO_SCCL, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      scclFunctionLL128<FUNC, T, UNROLL> scclfunc;
      scclfunc.run(args, 1);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective2D, NCCL_ALGO_SCCL, NCCL_PROTO_LL, FUNC, T, UNROLL> {
    public:
    __device__ void run(struct ncclWorkElem* args) {
      scclFunctionSimple<FUNC, T, UNROLL> scclfunc;
      scclfunc.run(args, 1);
    }
};

//FIXME: Find a way to remove below declarations for RING, TREE, and COLLNET.
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective2D, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective2D, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective2D, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
    }
};
