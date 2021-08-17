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

  struct ncclInfo info = { ncclFuncCustomCollective, "CustomCollective",
    sendbuff, recvbuff, count, datatype, comm->scclAlgo.redOp, 0, comm, stream, /* Args */
    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS};
  info.ld = ld;
  return ncclEnqueueCheck(&info);
}


NCCL_API(ncclResult_t, ncclCustomCollective2DInfo, std::vector<std::vector<NCCLChunk>>& ncclChunks, scclFlag** deviceScclFlags, 
    int* flagsPerBlock, const size_t ld, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream);
ncclResult_t ncclCustomCollective2DInfo(std::vector<std::vector<NCCLChunk>>& ncclChunks, scclFlag** deviceScclFlags, int* flagsPerBlock, const size_t ld, size_t count,
    ncclDataType_t datatype, ncclComm* comm, cudaStream_t stream) {
  NVTX3_FUNC_RANGE_IN(nccl_domain);
  
  if (ld % comm->scclAlgo.chunkld != 0) {
    WARN("Lead dimension %ld need to be divislbe by chunk leading dimension %d\n.", ld, comm->scclAlgo.chunkld);
    return ncclInvalidArgument;
  }

  //FIXME: Assumption of "number of channels is equal to number of thread blocks" is wrong.
  int maxSteps = 0;
  for (int bid = 0; bid < comm->scclAlgo.nChannels; bid++) {
    struct scclThreadBlock* scclTB = &comm->scclAlgo.scclTB[bid];
    maxSteps = MAX(maxSteps, scclTB->nsteps);
  }
  int maxIter = 65536; //FIXME: setting maximum iterations to 1024 is wrong.
  const int perChannelNCCLChunks = maxIter * maxSteps;
  const int numNCCLChunks = comm->scclAlgo.nChannels * perChannelNCCLChunks;
  NCCLChunk* dNCCLChunks;
  CUDACHECK(cudaMalloc(&dNCCLChunks, sizeof(NCCLChunk) * numNCCLChunks));
  struct ncclInfo info = { ncclFuncCustomCollectiveInfo, "CustomCollectiveInfo",
    dNCCLChunks, dNCCLChunks, count, datatype, comm->scclAlgo.redOp, 0, comm, stream, /* Args */
    SCCL_CHUNKSTEPS, SCCL_SLICESTEPS};
  info.ld = ld;
  ncclResult_t res = ncclEnqueueCheck(&info);
  if (res != ncclSuccess) {
    CUDACHECK(cudaFree(dNCCLChunks));
    return res;
  }

  //Wait for the interpreter kernel to complete
  CUDACHECK(cudaStreamSynchronize(stream));
  //Transfer NCCLChunk data from device to host.
  NCCLChunk* hNCCLChunks = new NCCLChunk[numNCCLChunks];
  CUDACHECK(cudaMemcpy(hNCCLChunks, dNCCLChunks, sizeof(NCCLChunk) * numNCCLChunks, cudaMemcpyDeviceToHost));
  CUDACHECK(cudaFree(dNCCLChunks));

  int maxChunksPerBlock, totalIterations = 0;

  //Store in a row major 2D format, where a row "i" represents NCCLChunks for a block "i"
  for (int bid = 0; bid < comm->scclAlgo.nChannels; bid++) {
    ncclChunks.push_back(std::vector<NCCLChunk>());
    for (int i = bid * perChannelNCCLChunks; i < (bid+1)*perChannelNCCLChunks; i++) {
      auto block = hNCCLChunks[i].chunk;
      if (block.startRow == -1)
        break;
      
      (*(ncclChunks.end() - 1)).push_back(hNCCLChunks[i]);
      totalIterations = MAX(totalIterations, hNCCLChunks[i].iter);
    }

    maxChunksPerBlock = MAX(maxChunksPerBlock, (*(ncclChunks.end() - 1)).size());
  }
  
  delete hNCCLChunks;
  totalIterations += 1;
  //Size of flags = (number of blocks * maximum number of chunks per block)
  if (comm->scclAlgo.flags != nullptr)
    CUDACHECK(cudaFree(comm->scclAlgo.flags));
  scclFlag* flags;
  NCCLCHECK(ncclCudaCalloc(&flags, comm->scclAlgo.nChannels * totalIterations));
  *deviceScclFlags = comm->scclAlgo.flags = comm->hostDevComm.scclAlgo.flags = flags;
  *flagsPerBlock = comm->scclAlgo.flagsPerBlock = comm->hostDevComm.scclAlgo.flagsPerBlock = totalIterations;
  size_t ptrVal = (size_t)(flags);
  CUDACHECK(cudaMemcpyAsync(&comm->devComm->scclAlgo.flags, &ptrVal, 1 * sizeof(void*), cudaMemcpyHostToDevice, stream));
  CUDACHECK(cudaMemcpyAsync(&comm->devComm->scclAlgo.flagsPerBlock, &totalIterations, 1 * sizeof(int), cudaMemcpyHostToDevice, stream));
  
  return ncclSuccess;
}