//nvcc matmul-allreduce.cu -std=c++11 -Xcompiler -fopenmp,-O3 -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/mpi -I.. -I/usr/local/cuda/include/ -I ../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c && mpicxx matmul-allreduce.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o matmul-allreduce

#include "header.h"

bool mpiRef(float* input, float* bias, size_t size, int biasSize, int comm_size, float* toCheck)
{
  float* allreduceOutput = new float[size];
  MPI_Allreduce(input, allreduceOutput, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  for (size_t i0 = 0; i0 < size; i0++) {
    float ref = allreduceOutput[i0] + bias[i0%biasSize];
    if (!eqFloat(toCheck[i0],ref)) {
      printf("Mismatch at %ld : ref '%f', computed '%f'\n",i0, ref, toCheck[i0]);
      return false;
    }
  }

  delete allreduceOutput;

  return true;
}

#define MAX_CHANNELS 80

int main(int argc, char** argv){
  printf("matmul-allreduce f16\n");
  const int N_GPUS = 16;
  
  MPI_Init(&argc, &argv);  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comm;
  CUDACHECK(cudaSetDevice(rank % N_GPUS));
  //initializing NCCL
  ncclUniqueId id;
  if (rank == 0) ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  ncclCommInitRank(&comm, comm_size, id, rank);

  cudaStream_t stream;
  int leastStreamPriority = 0, highestStreamPriority = 0;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(&leastStreamPriority, &highestStreamPriority));
  cudaStreamCreateWithPriority(&stream, cudaStreamDefault, highestStreamPriority);




  MPI_Barrier(MPI_COMM_WORLD);

  // int BATCH_SIZE[] = {8, 8, 8, 8, 8};
  // int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 1024, /*1.2B Model is 1536*/ 2048, /*2.5B Model is 1920*/ 2048, 
  //                            /*4.2B is 2304*/ 2048, /*8.3B is 3072*/ 4096};
  
  for (int p = 20; p < 30; p++) {
    size_t SZ = 1L << p;
    printf("SZ %ld\n", SZ);
    int biasSize = 2048;
    // Inputs
    float* bias;
    CUDACHECK(cudaMalloc(&bias, biasSize * sizeof(float)));

    memset_value(bias, 1.0f, biasSize);
    float* input;
    CUDACHECK(cudaMalloc(&input, SZ * sizeof(float)));
    memset_value(input, 1.0f, SZ);

    float* hbias = new float[biasSize];
    CUDACHECK(cudaMemcpy(hbias, bias, biasSize * sizeof(float), cudaMemcpyDeviceToHost));

    float* hinput = new float[SZ];
    CUDACHECK(cudaMemcpy(hinput, input, SZ * sizeof(float), cudaMemcpyDeviceToHost));

    float* output;
    CUDACHECK(cudaMalloc(&output, SZ * sizeof(float)));

    float* houtput = new float[SZ];

    MPI_Barrier(MPI_COMM_WORLD);
    
    for(int iter = 0; iter < 1; iter++) {
      if (rank == 0)
        printf("iter %d\n", iter);
      NCCLCHECK(ncclFusedAllReduceDropoutBias((const void*)input, (void*)output, SZ, (void*)bias, biasSize, 2.0f, ncclFloat, ncclSum, comm, stream));

      CUDACHECK(cudaStreamSynchronize(stream));

      CUDACHECK(cudaMemcpy(houtput, output, SZ * sizeof(float), cudaMemcpyDeviceToHost));
      MPI_Barrier(MPI_COMM_WORLD);

      assert(mpiRef(hinput, hbias, SZ, biasSize, comm_size, houtput));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
        printf("{Size: %ld} Fused AllReduce+Dropout-Bias-LayerNorm Resullts Checked\n", SZ);
    
    CUDACHECK(cudaFree(input));
    CUDACHECK(cudaFree(output));
    CUDACHECK(cudaFree(bias));
    delete houtput;
    delete hinput;
    delete hbias;
  }


  MPI_Finalize();
}