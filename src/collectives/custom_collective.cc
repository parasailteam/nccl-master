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

NCCL_API(ncclResult_t, ncclCustomCollective2D, const void* sendbuff, void* recvbuff, const size_t ld, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclCustomCollective2D(const void* sendbuff, void* recvbuff, const size_t ld, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  
  if (ld % comm->scclAlgo.chunkld != 0) {
    WARN("Lead dimension %ld need to be divislbe by chunk leading dimension %d\n.", ld, comm->scclAlgo.chunkld);
    return ncclInvalidArgument;
  }

  struct ncclInfo info = { ncclFuncCustomCollective2D, "CustomCollective2D",
    sendbuff, recvbuff, count, datatype, comm->scclAlgo.redOp, 0, comm, stream, /* Args */
    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS};
  info.ld = ld;
  return ncclEnqueueCheck(&info);
}