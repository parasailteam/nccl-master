#include <stdio.h>
#include "cuda_runtime.h"
#include "nccl.h"
#include <assert.h>
#include <cublas_v2.h>
#include <cstdint>
#include <curand.h>
#include <mpi.h>
#include <stdlib.h>
       #include <unistd.h>


#define CURANDCHECK(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
  printf("Error at %s:%d\n",__FILE__,__LINE__);            \
  assert(false);}} while(0)

#define CUBLASCHECK(cmd) do {                       \
  cublasStatus_t e = cmd;                           \
  if (e != CUBLAS_STATUS_SUCCESS) {                 \
    printf("Failed: CUBLAS error %s: %d '%d'\n",    \
           __FILE__, __LINE__, cmd);                \
    assert(false);                                  \
  }                                                 \
} while(0)                                          \

#define CUDACHECK(cmd) do {                         \
  cudaError_t e = cmd;                              \
  if( e != cudaSuccess ) {                          \
    printf("Failed: Cuda error %s:%d '%s'\n",             \
        __FILE__,__LINE__,cudaGetErrorString(e));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)


#define NCCLCHECK(cmd) do {                         \
  ncclResult_t r = cmd;                             \
  if (r!= ncclSuccess) {                            \
    printf("Failed, NCCL error %s:%d '%s'\n",             \
        __FILE__,__LINE__,ncclGetErrorString(r));   \
    exit(EXIT_FAILURE);                             \
  }                                                 \
} while(0)



// #include "header.h"

float absRelDiff(float u, float v) {
  return abs((u-v)/u);
}
bool eqFloat(float u, float v) {
  if (u == 0.0f || v == 0.0f)
    return u == v;
  return absRelDiff(u, v) <= 1e-5;
}


//Check results of each epoch
template<class T>
bool check_sccl_allreduce(const uint64_t size,
                        int rank, int iter, int comm_size,
                        T* d_minibatch_gradients, 
                        T* d_allreduced_gradient)
{
  bool passed = true;
  T *h_minibatch_gradients = (T*)malloc(size * sizeof(T));
  const size_t grad_array_size = size*sizeof(T);

  //Check AllReduced
  CUDACHECK(cudaMemcpy(h_minibatch_gradients, d_minibatch_gradients, 
			  grad_array_size, cudaMemcpyDeviceToHost));

  T *h_reduced_grad = (T*)malloc(grad_array_size);
  CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient, 
                       grad_array_size, cudaMemcpyDeviceToHost));
  T *h_reduced_grad_mpi = (T*)malloc(size * sizeof(T));
  if (sizeof(T) == 4)
    MPI_Allreduce(h_minibatch_gradients, h_reduced_grad_mpi, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
  else
    MPI_Allreduce(h_minibatch_gradients, h_reduced_grad_mpi, size, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  for (uint64_t i = 0; i < size; i++) {
    if (not eqFloat(h_reduced_grad_mpi[i], h_reduced_grad[i])) {
      printf ("[%d] Mismatch in h_reduced_grad at '%d': ref '%f' computed '%f', h_minibatch_gradients '%f'\n", rank, i, h_reduced_grad_mpi[i], h_reduced_grad[i], h_minibatch_gradients[i]);
      passed = false;
      break;
    }
  }
  //Correct these to free
  free(h_minibatch_gradients);
  free(h_reduced_grad_mpi);
  return passed;
}

template<class T>
bool check_sccl_reducescatter(const uint64_t size,
                        int rank, int iter,
                        const int comm_size,
                        T* d_minibatch_gradients, 
                        T* d_allreduced_gradient)
{
  bool passed = true;
  const size_t grad_array_size = size*sizeof(T);
  T *h_minibatch_gradients = (T*)malloc(grad_array_size);

  //Check AllReduced
  CUDACHECK(cudaMemcpy(h_minibatch_gradients, d_minibatch_gradients, 
			  grad_array_size, cudaMemcpyDeviceToHost));

  T *h_reduced_grad = (T*)malloc(grad_array_size/comm_size);
  CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient, 
                       grad_array_size/comm_size, cudaMemcpyDeviceToHost));
  T *h_reduced_grad_mpi = (T*)malloc(size * comm_size * sizeof(float));
  // //if (sizeof(T) == 4)
  // MPI_Allgather(h_minibatch_gradients, size, MPI_FLOAT, h_reduced_grad_mpi, size, MPI_FLOAT, MPI_COMM_WORLD);
  // else
    // MPI_AllGather(h_minibatch_gradients, h_reduced_grad_mpi, size, MPI_DOUBLE, MPI_COMM_WORLD);
  float ref = 0.0f;
  for (int i = 0; i < comm_size; i++) {
    ref += (float)(1 << i);
  }
  for (uint64_t i = 0; i < size/comm_size; i++) {
    if (not eqFloat(h_reduced_grad[i], ref)) {
      printf ("[%d] Mismatch in h_reduced_grad at '%d': ref '%f' computed '%f',h_minibatch_gradients '%f'\n", rank, i, ref, h_reduced_grad[i], h_minibatch_gradients[i]);
      passed = false;
      break;
    }
  }
  //Correct these to free
  free(h_minibatch_gradients);
  // free(h_reduced_grad_mpi);
  return passed;
}

template<class T>
bool check_sccl_collective(const uint64_t size,
                        int rank, int iter,
                        const int comm_size,
                        T* d_minibatch_gradients, 
                        T* d_allreduced_gradient)
{
  bool passed = true;
  const size_t grad_array_size = size*sizeof(T);
  T *h_minibatch_gradients = (T*)malloc(grad_array_size);

  //Check AllReduced
  CUDACHECK(cudaMemcpy(h_minibatch_gradients, d_minibatch_gradients, 
			  grad_array_size, cudaMemcpyDeviceToHost));

  T *h_reduced_grad = (T*)malloc(grad_array_size * comm_size);
  CUDACHECK(cudaMemcpy(h_reduced_grad, d_allreduced_gradient, 
                       grad_array_size * comm_size, cudaMemcpyDeviceToHost));
  T *h_reduced_grad_mpi = (T*)malloc(size * comm_size * sizeof(float));
  // //if (sizeof(T) == 4)
  MPI_Allgather(h_minibatch_gradients, size, MPI_FLOAT, h_reduced_grad_mpi, size, MPI_FLOAT, MPI_COMM_WORLD);
  // else
    // MPI_AllGather(h_minibatch_gradients, h_reduced_grad_mpi, size, MPI_DOUBLE, MPI_COMM_WORLD);
  for (uint64_t i = 0; i < size * comm_size; i++) {
    if (not eqFloat(h_reduced_grad_mpi[i], h_reduced_grad[i])) {
      printf ("[%d] Mismatch in h_reduced_grad at '%d': ref '%f' computed '%f',h_minibatch_gradients '%f'\n", rank, i, h_reduced_grad_mpi[i], h_reduced_grad[i], h_minibatch_gradients[i]);
      passed = false;
      break;
    }
  }
  //Correct these to free
  free(h_minibatch_gradients);
  // free(h_reduced_grad_mpi);
  return passed;
}



template<class T>
void traditional_weight_update(const ncclComm_t& comm, const uint64_t size,
                               T* minibatch_gradients, 
                               T* allreduced_gradient, 
                               cudaStream_t& s,
                               ncclDataType_t datatype)
{
  
}

template<class T>
void cudaMemRandInt(T* dst, size_t nelems)
{
  curandGenerator_t gen;
  CURANDCHECK(curandCreateGenerator(&gen,
                                    CURAND_RNG_PSEUDO_DEFAULT));
  CURANDCHECK(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));
  if (sizeof(T) == sizeof(float))
    CURANDCHECK(curandGenerateUniform(gen, (float*)dst, nelems));
  else
    CURANDCHECK(curandGenerateUniformDouble(gen, (double*)dst, nelems));
  CURANDCHECK(curandDestroyGenerator(gen));
}

// template<class T>
// __global__ void gpu_memset_kernel(T* f, T v, size_t nelems)
// {
//   uint idx = threadIdx.x + blockIdx.x*blockDim.x;
//   if (idx >= nelems)
//     return;
  
//   f[idx] = v;
// }

template<class T>
void memset_value(T*f, T v, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);

  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = v;
  }

  CUDACHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}

template<class T>
void memset_identity(T*f, size_t nelems) 
{
  T* h_buff = (T*)malloc(sizeof(T)*nelems);

  for (uint64_t i = 0; i < nelems; i++) {
    h_buff[i] = (T)(i);
  }

  CUDACHECK(cudaMemcpy(f, h_buff, sizeof(T)*nelems, cudaMemcpyHostToDevice));
  free(h_buff);
}

template<class T>
float run(int rank, const int64_t size, const ncclDataType_t datatype)
{
  int comm_size;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comm;
  CUDACHECK(cudaSetDevice(rank % 16));
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  if (rank == 0) printf("size %d\n", size);
  enum CollType {AllGather, ReduceScatter, AllReduce} ;
  CollType collType = AllReduce;
  const int epochs = 1000;

  //allocating and initializing device buffers
  T* minibatch_gradients;
  T* allreduced_gradient;


  CUDACHECK(cudaMalloc(&minibatch_gradients, size * sizeof(T)));
  CUDACHECK(cudaMalloc(&allreduced_gradient, size * sizeof(T)));
  cudaStream_t s;
  
  //cudaMemRandInt(minibatch_gradients, size);
  CUDACHECK(cudaMemset(allreduced_gradient, 0, size * sizeof(T)));
  if (collType == AllGather) {
    minibatch_gradients = allreduced_gradient + rank * (size/comm_size);
    memset_value(minibatch_gradients, (float)(1<<rank), size/comm_size);
  } else if (collType == ReduceScatter || collType == AllReduce) {
    // minibatch_gradients = allreduced_gradient + rank * (size/comm_size);
    //memset_value(minibatch_gradients, (float)(1<<rank), size);
    cudaMemRandInt(minibatch_gradients, size);
    //memset_identity(minibatch_gradients, size);
  }
  
  //CUDACHECK(cudaMemset(weights, 0, size * sizeof(T)));
  CUDACHECK(cudaStreamCreate(&s));

  //initializing NCCL
  ncclUniqueId id;
  if (rank == 0)
	  ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);

  assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
  int nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
  char filename[256] = {0};

  if (collType == AllGather) {
    sprintf(filename, "allgather_ring_%d_ranks_%d_channel_2D_chunks.xml", comm_size, nChannels);
  } else if (collType == ReduceScatter) {
    sprintf(filename, "reduce_scatter_ring_%d_ranks_%d_channel.xml", comm_size, nChannels);
  } else if (collType == AllReduce) {
    sprintf(filename, "allreduce_ring_%d_ranks_%d_channel_2D_chunks.xml", comm_size, nChannels);
  }

  ncclCommInitRankWithScclXML(&comm, comm_size, id, rank, filename);
  MPI_Barrier(MPI_COMM_WORLD);
  
  if (rank == -1) {
    printf("PID %d on ready for attach\n", getpid());
    fflush(stdout);

    printf("waiting for input\n");
    int c= getchar();
  }

  MPI_Barrier(MPI_COMM_WORLD);
  // gpu_memset_kernel<<<size/256 + 1,256, 0, s>>>(minibatch_gradients, (T)rank, size);
  int warmup = 1;
  for (int iter = 0; iter < 1; iter++) {
  // #define ALLREDUCE
  #ifdef ALLREDUCE
    NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients, 
            (void*)allreduced_gradient, size, datatype, ncclSum, comm, s));

    CUDACHECK(cudaStreamSynchronize(s));
    if (iter == 0)
       assert(check_sccl_allreduce(size, rank, iter, comm_size, minibatch_gradients, allreduced_gradient));
  #else
    if (collType == AllGather) {
      T* minibatch_gradients2;
      CUDACHECK(cudaMalloc(&minibatch_gradients2, size * sizeof(T)));
      CUDACHECK(cudaMemcpy(minibatch_gradients2, minibatch_gradients, size/comm_size * sizeof(T), cudaMemcpyDeviceToDevice));
      NCCLCHECK(ncclCustomCollective((const void*)minibatch_gradients, 
              (void*)allreduced_gradient, size/comm_size, datatype, comm, s));

      CUDACHECK(cudaStreamSynchronize(s));

      // NCCLCHECK(ncclCustomCollective(customAllGatherColl, (const void*)minibatch_gradients, 
      //         (void*)allreduced_gradient, size/comm_size, datatype, ncclSum, comm, s));
      // CUDACHECK(cudaStreamSynchronize(s));

      assert(check_sccl_collective(size/comm_size, rank, iter, comm_size, minibatch_gradients2, allreduced_gradient));
    } else if (collType == ReduceScatter) {
      T* minibatch_gradients2;
      CUDACHECK(cudaMalloc(&minibatch_gradients2, size * sizeof(T)));
      CUDACHECK(cudaMemcpy(minibatch_gradients2, minibatch_gradients, size/comm_size * sizeof(T), cudaMemcpyDeviceToDevice));
      NCCLCHECK(ncclCustomCollective((const void*)minibatch_gradients, 
              (void*)allreduced_gradient, size/comm_size, datatype, comm, s));

      CUDACHECK(cudaStreamSynchronize(s));
      assert(check_sccl_reducescatter(size, rank, iter, comm_size, minibatch_gradients, allreduced_gradient));
    } else if (collType == AllReduce) {
      NCCLCHECK(ncclCustomCollective2D((const void*)minibatch_gradients, 
              (void*)allreduced_gradient, 1024, size/comm_size, datatype, comm, s));

      CUDACHECK(cudaStreamSynchronize(s));
      assert(check_sccl_allreduce(size, rank, iter, comm_size, minibatch_gradients, allreduced_gradient));
    }
  #endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  NCCLCHECK(ncclCommDestroy(comm));


  return 100;

  cudaEvent_t start, stop;
  float elapsedTime;
  CUDACHECK(cudaEventCreate(&start));
  CUDACHECK(cudaEventCreate(&stop));
  MPI_Barrier(MPI_COMM_WORLD);
  CUDACHECK(cudaEventRecord(start,0));

  for (int iter = 0; iter < 100; iter++) {
    NCCLCHECK(ncclAllReduce((const void*)minibatch_gradients, 
            (void*)allreduced_gradient, size, datatype, ncclSum, comm, s));

    CUDACHECK(cudaStreamSynchronize(s));
  }
  
  MPI_Barrier(MPI_COMM_WORLD);
  CUDACHECK(cudaEventRecord(stop,0));
  CUDACHECK(cudaEventSynchronize(stop));
  CUDACHECK(cudaEventElapsedTime(&elapsedTime, start, stop));

  //free device buffers
  CUDACHECK(cudaFree(minibatch_gradients));
  CUDACHECK(cudaFree(allreduced_gradient));
  CUDACHECK(cudaStreamDestroy(s));

  //finalizing NCCL
  ncclCommDestroy(comm);

  return elapsedTime;
}

int main(int argc, char* argv[])
{
  //This code implements the backward pass of a large layer, using model parallelism
  //The whole dataset is divided into several minibatches, with one minibatch assigned to one gpu.
  //The gradients are received from each gpu, then reduced, and weights are updated as:
  //w = w - alpha * g
  //Before running this program do "export NCCL_PROTO=LL"

  MPI_Init(&argc, &argv);

  int rank;
  const int size = 8*1024*1024;//8192*1024;//128*1024*1024;
  float elapsedTime1 = run<float>(rank, size, ncclFloat);

  printf("Success time: %f\n", elapsedTime1);
  MPI_Finalize();
  return 0;
}
