/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclCustomCollective, const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int scclAlgorithmIndex, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclCustomCollective(const void* sendbuff, void* recvbuff, size_t count,
    ncclDataType_t datatype, int scclAlgorithmIndex, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  
  struct ncclInfo info = { ncclFuncCustomCollective, "CustomCollective",
    sendbuff, recvbuff, (recvbuff == sendbuff), count, datatype, comm->scclAlgos[scclAlgorithmIndex].redOp, 0, comm, stream, /* Args */
    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS };
  info.scclAlgoIndex = scclAlgorithmIndex;
  return ncclEnqueueCheck(&info);
}
