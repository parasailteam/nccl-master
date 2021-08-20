/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "sccl_interpreter.h"

/*Support for Custom Collectives operations on 2D arrays*/

//

//Represents a 2D chunk in the outside array with start coordinate (row, col)
//and number of rows and columns of chunk.
struct Chunk2D {
  int startRow;
  int startCol;
  int rows;
  int cols;
  __device__ __forceinline__ Chunk2D() {
    startRow = startCol = rows = cols = -1;
  }

  __device__ __forceinline__ Chunk2D(const ssize_t size, const int chunkIdx, const int chunkRows, const int chunkCols,
                                     const int numChunksInCols, const int matrixRows, const int matrixCols) {
    //FIXME: Division is for performance. 
    //Performance can be significantly improved if numChunksInCols is known at compile time or is always a power of 2.
    startCol = (chunkIdx % numChunksInCols) * chunkCols;
    startRow = (chunkIdx / numChunksInCols) * chunkRows;
    //
    rows = min(chunkRows, matrixRows - startRow);
    cols = chunkCols;
  }

  __device__ __forceinline__ int nelem() const {
    return rows * cols;
  }
};

//Child class of ncclPrimitives that works on 2D chunks.
template <int UNROLL, int SLICESPERCHUNK, int SLICESTEPS, typename T, int NRECV, int NSEND, int DIRECT, class FUNC>
class ncclPrimitives2D : public ncclPrimitives<UNROLL, SLICESPERCHUNK, SLICESTEPS, T, NRECV, NSEND, DIRECT, FUNC> {
protected:
  const Chunk2D* invalidBlock = nullptr;

  //Similar to ncclPrimitives::GenericOp function but takes Chunk2D instead of offsets.
  //Chunk2D represents the block of src/dst to process.
  template <int DIRECTRECV, int DIRECTSEND, int RECV, int SEND, int SRC, int DST>
  inline __device__ void
  GenericOp2D(const T* srcPtr, T* dstPtr, const Chunk2D* srcBlock, const Chunk2D* dstBlock, int nelem, ssize_t directOffset) {
    int offset = 0;
    int sliceSize = this->stepSize*SLICESTEPS;
    int dataSize = max(DIVUP(nelem, 16*SLICESPERCHUNK)*16, sliceSize/32);
    #pragma unroll
    for (int slice=0; slice<SLICESPERCHUNK; ++slice) {
      int realSize = max(0, min(dataSize, nelem-offset));
      if (this->tid < this->nworkers) {
        if (SRC && (this->role & ROLE_SRC)) this->srcs[0] = srcPtr;//Do not add offset
        if (RECV && (this->role & ROLE_WAIT_RECV)) this->waitRecv<SRC, DIRECTRECV>(directOffset+offset);
        if (DST && (this->role & ROLE_DST)) this->dsts[0] = dstPtr;//Do not add offset
        if (SEND && (this->role & ROLE_WAIT_SEND)) this->waitSend<DST, DIRECTSEND>(directOffset+offset, realSize*sizeof(T));
        if (realSize > 0) {
          this->subBarrier();
          if (DIRECTRECV && this->srcs[0] == this->dsts[0]) {
            // We can only have one direct receive. Since srcs[0] == dstPtr+offset, skip one copy
            if (SEND) {
              // (1-SEND) is only there to avoid compilation errors in case NSEND=0 (and SEND=0).
              ReduceOrCopyMultiChunk2D<UNROLL, FUNC, T, 1, 1, 1, 1-SEND+NSEND, SRC, DST, Chunk2D>(this->tid, this->nworkers, RECV*this->nrecv+SRC, this->srcs, SEND*this->nsend+DST, this->dsts, 
                                                                                                  offset, srcBlock, dstBlock, matrixCols, realSize);
            }
          } else {
            ReduceOrCopyMultiChunk2D<UNROLL, FUNC, T, RECV+SRC, RECV*NRECV+SRC, SEND+DST, SEND*NSEND+DST, SRC, DST, Chunk2D>(this->tid, this->nworkers, RECV*this->nrecv+SRC, this->srcs, SEND*this->nsend+DST, this->dsts, 
                                                                                                                             offset, srcBlock, dstBlock, matrixCols, realSize);
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
  size_t matrixCols;
  __device__ __forceinline__
  ncclPrimitives2D(const int tid, const int nworkers, int* recvPeers, int* sendPeers, T* directBuff, int stepSize, struct ncclChannel* channel, struct ncclDevComm* comm, struct ncclShmemPtrs* ptrs, int group, int matrixCols):
    ncclPrimitives<UNROLL, SLICESPERCHUNK, SLICESTEPS, T, NRECV, NSEND, DIRECT, FUNC>(tid, nworkers, recvPeers, sendPeers, directBuff, stepSize, channel, comm, ptrs, group), matrixCols(matrixCols)
  {}
  
  __device__ __forceinline__ void
  send(const T* src, const Chunk2D* srcBlock, int offset, int nelem) {
    GenericOp2D<0, 0, 0, 1, 1, 0>(src, NULL, srcBlock, invalidBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recv(T* dst, const Chunk2D* dstBlock, int offset, int nelem) {
    GenericOp2D<0, 0, 1, 0, 0, 1>(NULL, dst, invalidBlock, dstBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvCopySend(T* dst, const Chunk2D* dstBlock, int offset, int nelem) {
    GenericOp2D<0, 0, 1, 1, 0, 1>(NULL, dst, invalidBlock, dstBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvReduceCopy(const T* src, T* dst, const Chunk2D* srcBlock, const Chunk2D* dstBlock, int offset, int nelem) {
    GenericOp2D<0, 0, 1, 0, 1, 1>(src, dst, srcBlock, dstBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvReduceSend(const T* src, const Chunk2D* srcBlock, int offset, int nelem) {
    GenericOp2D<0, 0, 1, 1, 1, 0>(src, NULL, srcBlock, invalidBlock, nelem, offset);
  }

  __device__ __forceinline__ void
  recvReduceCopySend(const T* src, T* dst, const Chunk2D* srcBlock, const Chunk2D* dstBlock, int offset, int nelem) {
    GenericOp2D<0, 0, 1, 1, 1, 1>(src, dst, srcBlock, dstBlock, nelem, offset);
  }

  __device__ __forceinline__ 
  void reduce(const T* src, const Chunk2D* srcBlock, T* dst, const Chunk2D* dstBlock, int nelem) {
    GenericOp2D<0, 0, 0, 0, 2, 1>(src, dst, srcBlock, dstBlock, nelem, 0);
  }

  __device__ __forceinline__ 
  void localCopy(const T* src, const Chunk2D* srcBlock, T* dst, const Chunk2D* dstBlock, int nelem) {
    GenericOp2D<0, 0, 0, 0, 1, 1>(src, dst, srcBlock, dstBlock, nelem, 0);
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
  int chunkCols;
  ssize_t numScclChunks;
  int matrixRows;
  int matrixCols;

  ncclPrimitives2D<UNROLL, SCCL_CHUNKSTEPS/SCCL_SLICESTEPS, SCCL_SLICESTEPS, T, 1, 1, 1, FUNC> prims;

  __device__ __forceinline__ SimpleWrapper2D(struct ncclWorkElem* args, int tid, int* recvPeer, int* sendPeer, ssize_t size, ssize_t sizePerScclChunk,
                                             T * thisOutput, struct ncclChannel* channel)
    : nthreads(args->nThreads-WARP_SIZE),
      stepSize(args->comm->buffSizes[NCCL_PROTO_SIMPLE] / (sizeof(T)*NCCL_STEPS)),
      rank(args->comm->rank),
      prims(tid, nthreads, recvPeer, sendPeer, thisOutput, stepSize, channel, args->comm, ncclShmem->ptrs, 0, args->ld) {
        struct scclAlgorithm* scclAlgo = &args->comm->scclAlgo;
        chunkCols = scclAlgo->chunkld;
        int nchunksPerLoop = scclAlgo->nchunksPerLoop;
        matrixCols = args->ld; //TODO: make ld to  matrixCols
        matrixRows = size/matrixCols;

        //Align chunk size to the number of columns.
        const int maxChunkSize = stepSize * SCCL_CHUNKSTEPS;
        //Divide size equally into chunks
        chunkSize = min(maxChunkSize, DIVUP(size, nchunksPerLoop));
        //Align chunk size with number of columns
        if (ROUNDUP(chunkSize, matrixCols) < maxChunkSize) {
          //If possible increase chunkSize to align with chunk cols
          chunkSize = ALIGN_SIZE(chunkSize, matrixCols);
        } else {
          //Otherwise decrease
          chunkSize = ALIGN_DOWN(chunkSize, matrixCols);
        }

        //Limit number of rows in chunk to matrix rows
        chunkRows = min(chunkSize/chunkCols, matrixRows);
        //FIXME: Need to make chunkRows a perfect divisor of matrixRows.
        //Find a better way to acheive this
        for (; chunkRows >= 1; chunkRows--) {
          if (matrixRows % chunkRows == 0) {
            break;
          }
        }
        //Get the final size of chunk
        chunkSize = chunkRows * chunkCols;
        //Number of chunks in the columns
        numRealChunks = matrixCols/chunkCols;
        //Total chunks in the matrix
        const int numTotalChunks = (matrixRows/chunkRows) * (matrixCols/chunkCols);
        //FIXME: Instead of division, DIVUP might also work
        numScclChunks = numTotalChunks/nchunksPerLoop;
        assert(numTotalChunks % nchunksPerLoop == 0);
      }


  __device__ __forceinline__ Chunk2D initIter(int iter, int step) {
    //Do nothing here
    return Chunk2D();
  }

  __device__ Chunk2D getOffset(Chunk2D& chunkOffset, ssize_t gridChunk, int sccltranOffset, int count, ssize_t sizePerScclChunk) {
    int chunkIdx = gridChunk + (sccltranOffset + count)*numScclChunks;
    return Chunk2D(matrixRows*matrixCols, chunkIdx, chunkRows, chunkCols, numRealChunks, matrixRows, matrixCols);
  }
  
  __device__ __forceinline__ void send(T * src, const Chunk2D& srcBlock, const Chunk2D& dstBlock, int count) {
    prims.send(src, &srcBlock, 0, srcBlock.nelem()*count);
  }

  __device__ __forceinline__ void recv(T * dst, const Chunk2D& dstBlock, int count) {
    prims.recv(dst, &dstBlock, 0, dstBlock.nelem()*count);
  }

  __device__ __forceinline__ void recvCopySend(T * dst, const Chunk2D& dstBlock, int count) {
    prims.recvCopySend(dst, &dstBlock, 0, dstBlock.nelem()*count);
  }
  
  __device__ __forceinline__ void recvReduceSend(T * src, const Chunk2D& srcBlock, int count) {
    prims.recvReduceSend(src, &srcBlock, 0, srcBlock.nelem()*count);
  }

  __device__ __forceinline__ void recvReduceCopy(T * src, const Chunk2D& srcBlock, T * dst, const Chunk2D& dstBlock, int count) {
    prims.recvReduceCopy(src, dst, &srcBlock, &dstBlock, 0, srcBlock.nelem()*count);
  }
  
  __device__ __forceinline__ void recvReduceCopySend(T * src, const Chunk2D& srcBlock, T * dst, const Chunk2D& dstBlock, int count) {
    prims.recvReduceCopySend(src, dst, &srcBlock, &dstBlock, 0, srcBlock.nelem()*count);
  }

  __device__ void reduce(T * src, const Chunk2D& srcBlock, T * dst, const Chunk2D& dstBlock, int count) {
    prims.reduce(src, &srcBlock, dst, &dstBlock, srcBlock.nelem()*count);
  }

  __device__ void localCopy(T * src, const Chunk2D& srcBlock, T * dst, const Chunk2D& dstBlock, int count) {
    prims.reduce(src, &srcBlock, dst, &dstBlock, srcBlock.nelem()*count);
  }
};

template<class FUNC, typename T, int UNROLL>
class scclFunction2DSimple : public scclFunction<T, Chunk2D, SimpleWrapper2D<FUNC, T, UNROLL>> {};

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
      if (threadIdx.x == 0) {
        printf("CustomCollective 2D does not work in LL128 protocol.\n");
      }

      // scclFunctionLL128<FUNC, T, UNROLL> scclfunc;
      // scclfunc.run(args, 1);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective2D, NCCL_ALGO_SCCL, NCCL_PROTO_LL, FUNC, T, UNROLL> {
    public:
    __device__ void run(struct ncclWorkElem* args) {
      if (threadIdx.x == 0) {
        printf("CustomCollective 2D does not work in LL protocol.\n");
      }
      // scclFunctionSimple<FUNC, T, UNROLL> scclfunc;
      // scclfunc.run(args, 1);
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
