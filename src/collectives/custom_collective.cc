/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclCustomCollective, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclCustomCollective(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  
  struct ncclInfo info = { ncclFuncCustomCollective, "CustomCollective",
    sendbuff, recvbuff, count, datatype, comm->scclAlgo.redOp, 0, comm, stream, /* Args */
    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS };
  return ncclEnqueueCheck(&info);
}

NCCL_API(ncclResult_t, ncclCustomCollective2D, const void* sendbuff, void* recvbuff, const size_t cols, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclCustomCollective2D(const void* sendbuff, void* recvbuff, const size_t cols, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  
  if (comm->scclAlgo.chunkCols <= 0) {
    WARN("Chunk Columns in SCCL XML set by 'chunkCols=\"%d\"' should be greater than 0\n", comm->scclAlgo.chunkCols);
    return ncclInvalidUsage;
  }

  if (cols % comm->scclAlgo.chunkCols != 0) {
    WARN("Columns %ld need to be divislbe by chunk columns %d\n.", cols, comm->scclAlgo.chunkCols);
    return ncclInvalidArgument;
  }

  struct ncclInfo info = { ncclFuncCustomCollective2D, "CustomCollective2D",
    sendbuff, recvbuff, count, datatype, comm->scclAlgo.redOp, 0, comm, stream, /* Args */
    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS};
  info.cols = cols;
  return ncclEnqueueCheck(&info);
}
