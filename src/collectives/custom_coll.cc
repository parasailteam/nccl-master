/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "enqueue.h"

NCCL_API(ncclResult_t, ncclCustomColl, const void* sendbuff, void* recvbuff, size_t count,
    void* bias, int biasSize, float dropoutProb, void* residual,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclCustomColl(const void* sendbuff, void* recvbuff, size_t count,
    void* bias, int biasSize, void* residual, float dropoutProb,
    ncclDataType_t datatype, ncclRedOp_t op, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  FusedDropoutBiasParams fusedDropoutBiasParams = {bias, biasSize, dropoutProb, residual};
  
  struct ncclInfo info = { ncclFuncCustomColl, "CustomColl",
    sendbuff, recvbuff, count, datatype, op, 0, comm, stream, /* Args */
    SCKL_CHUNKSTEPS, SCKL_SLICESTEPS };
  info.fusedDropoutBiasParams = fusedDropoutBiasParams;
  return ncclEnqueueCheck(&info);
}
