//nvcc matmul-allreduce.cu -std=c++11 -Xcompiler -fopenmp,-O3 -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/mpi -I.. -I/usr/local/cuda/include/ -I ../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c && mpicxx matmul-allreduce.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o matmul-allreduce

#include "header.h"
#define GPT2_PARAMS
#ifdef GPT2_PARAMS
  #include "cutlass-matmul.h"
#else
  #include "cutlass-matmul.h"
#endif
#include <cuda_profiler_api.h>

#include <map> 

bool mpiRef(const float* m1, const float* m2, float* m1m2, int M, int N, int K, int comm_size, int rank = -1)
{
  // printf("Starting Matmul\n");
  // float* expected = new float[M*N];
  // for (int i = 0; i < M; ++i) {
  //   for (int j = 0; j < N; ++j) {
  //     int k = 0;
  //     for (k = 0; k < K; ++k) 
  //     { 
  //           expected[i*N +j] += m1[i*K + k] * m2[k*N + j];
  //     }
  //   }
  // }
  // printf("Starting AllReduce\n");
  // float* allreduceOut = new float[M*N];
  // MPI_Allreduce(expected, allreduceOut, M*N, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);

  // for (size_t i0 = 0; i0 < M*N; i0++) {
  //   if (!eqFloat(allreduceOut[i0], m1m2[i0])) {
  //     printf("Mismatch at %ld : ref '%f', computed '%f'\n",i0, allreduceOut[i0], m1m2[i0]);
  //     return false;
  //   }
  // }

  for (size_t i0 = 0; i0 < M*N; i0++) {
    float ref = K*comm_size;
    if (!eqFloat(ref, m1m2[i0])) {
      printf("rankk %d Mismatch at %ld : ref '%f', computed '%f'\n",rank, i0, ref, m1m2[i0]);
      return false;
    }
  }
  return true;
}

#define MAX_CHANNELS 80

int main(int argc, char** argv){
  const int N_GPUS = 16;
  
  MPI_Init(&argc, &argv);  int comm_size, rank;
  MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ncclComm_t comm;
  CUDACHECK(cudaSetDevice(rank % N_GPUS));
  //initializing NCCL
  enum CollType {AllGather, ReduceScatter, AllReduce} ;
  CollType collType = AllReduce;
  const int epochs = 1000;
  
  //CUDACHECK(cudaMemset(weights, 0, size * sizeof(T)));

  //initializing NCCL
  ncclUniqueId id;
  if (rank == 0)
	  ncclGetUniqueId(&id);
  MPI_Bcast(&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD);
  // #define ALLREDUCE

  int nChannels = 12;

  if (getenv ("NCCL_MIN_NCHANNELS") != NULL) {
    assert (atoi(getenv ("NCCL_MIN_NCHANNELS")) == atoi(getenv ("NCCL_MAX_NCHANNELS")));
    nChannels = atoi(getenv ("NCCL_MIN_NCHANNELS"));
  } 
  char filename[256] = {0};

  if (collType == AllGather) {
    sprintf(filename, "allgather_ring_%d_ranks_%d_channel_2D_chunks.xml", comm_size, nChannels);
  } else if (collType == ReduceScatter) {
    sprintf(filename, "reduce_scatter_ring_%d_ranks_%d_channel_2D_chunks.xml", comm_size, nChannels);
  } else if (collType == AllReduce) {
    sprintf(filename, "allreduce_ring_%d_ranks_%d_channel_2D_chunks.xml", comm_size, nChannels);
  }
  printf("filename '%s'\n", filename);
  scclAlgorithm_t scclAlgo;
  ncclCommInitRankWithScclXML(&comm, comm_size, id, rank, filename, &scclAlgo);
  
  cudaStream_t stream;
  int leastStreamPriority = 0, highestStreamPriority = 0;
  CUDACHECK(cudaDeviceGetStreamPriorityRange(&leastStreamPriority, &highestStreamPriority));
  printf("highestStreamPriority %d\n", highestStreamPriority);
  cudaStreamCreateWithPriority(&stream, cudaStreamDefault, highestStreamPriority);

  cudaStream_t cutlassStream;
  cudaStreamCreateWithPriority(&cutlassStream, cudaStreamDefault, leastStreamPriority);

  MPI_Barrier(MPI_COMM_WORLD);

  // int BATCH_SIZE[] = {8, 8, 8, 8, 8};
  // int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 1024, /*1.2B Model is 1536*/ 2048, /*2.5B Model is 1920*/ 2048, 
  //                            /*4.2B is 2304*/ 2048, /*8.3B is 3072*/ 4096};
  
  #ifdef GPT2_PARAMS
    int SEQUENCE_LENGTH = 1024;  
    // int MODEL_PARALLEL_GPUS[] = {1, 2, 4, 8, 16};
    // float MODEL_PARAMS[] = {0.345, 1.2, 2.5, 4.2, 8.3};

    int BATCH_SIZE[] = {8, 16, 32, 64};
    // int BATCH_SIZE[] = {32, 64, 512, 1024, 2048};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 4096, /*1.2B Model is 1536*/ 4096, /*2.5B Model is 1920*/ 4096, 
                              /*4.2B is 2304*/ 4096};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {3072, /*345M Model*/ 3072, /*1.2B Model is 1536*/ 3072, /*2.5B Model is 1920*/ 3072, 
                                          /*4.2B is 2304*/ 3072};
    int MODEL_PARALLEL_GPUS[] = {2, 2, 2, 2};
    float MODEL_PARAMS[] = {8.3, 8.3, 8.3, 8.3, 8.3};
  #else
    int SEQUENCE_LENGTH = 2048;  
    // int MODEL_PARALLEL_GPUS[] = {1, 2, 4, 8, 16};
    // float MODEL_PARAMS[] = {0.345, 1.2, 2.5, 4.2, 8.3};

    int BATCH_SIZE[] = {1, 2, 4, 6};
    // int BATCH_SIZE[] = {32, 64, 512, 1024, 2048};
    int HIDDEN_DIMENSIONS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288, 12288};
    int HIDDEN_DIMENSIONS_12CHANNELS[] = {/*345M Model*/ 12288, /*1.2B Model is 1536*/ 12288, /*2.5B Model is 1920*/ 12288, 12288};
    int MODEL_PARALLEL_GPUS[] = {16, 16, 16, 16};
    float MODEL_PARAMS[] = {137, 137, 137, 137};
  #endif
  
  int workIndex = 0;

  for (int model = 0; model < sizeof(HIDDEN_DIMENSIONS)/sizeof(HIDDEN_DIMENSIONS[0]); model++) {
    for (int matMulType = 1; matMulType < 2; matMulType++) {

      int M = BATCH_SIZE[model] * SEQUENCE_LENGTH;
      int N = HIDDEN_DIMENSIONS_12CHANNELS[model];
      int K = N/MODEL_PARALLEL_GPUS[model] * ((matMulType == 0) ? 1 : 4);

      if (rank == 0)
        printf("Model Size %.2f B Params , MatMul: [%d, %d] X [%d, %d]\n", MODEL_PARAMS[model], M, K, K, N);
            
      if (comm_size != MODEL_PARALLEL_GPUS[model])
        continue;
      // Inputs
      half* m1;
      CUDACHECK(cudaMalloc(&m1, M*K * sizeof(half)));
      // cudaMemRandInt(m1, M*K);
      memset_value(m1, __float2half(1.0f), M*K);
      half* m2;
      CUDACHECK(cudaMalloc(&m2, K*N * sizeof(half)));
      // cudaMemRandInt(m2, K*N);
      memset_value(m2, __float2half(1.0f), K*N);
      half* m1m2;
      CUDACHECK(cudaMalloc(&m1m2,  M*N* sizeof(half)));
      
      half* _m1m2;
      CUDACHECK(cudaMalloc(&_m1m2,  M*N* sizeof(half)));

       half* __m1m2;
      CUDACHECK(cudaMalloc(&__m1m2,  M*N* sizeof(half)));

      MPI_Barrier(MPI_COMM_WORLD);
      
      float totalTime = 0;
      float cublasTime = 0;
      float allReduceTime = 0;
      float matmulTime = 0;

      memset_value(m1m2, __float2half(0.0f), M*N);

      totalTime = 0.0;
      allReduceTime = 0;
      matmulTime = 0;
      
      MPI_Barrier(MPI_COMM_WORLD);

      {
         int length_m = M;
        int length_n = N;
        int length_k = K;
        cutlass::gemm::GemmCoord problem_size(M, N, K);

        cutlass::TensorRef<ElementInputA, LayoutInputA> tensor_a((cutlass::half_t*)m1, LayoutInputA::packed(problem_size.mk()));
        cutlass::TensorRef<ElementInputB, LayoutInputB> tensor_b((cutlass::half_t*)m2, LayoutInputA::packed(problem_size.kn()));
        cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_c((cutlass::half_t*)_m1m2, LayoutInputA::packed(problem_size.mn()));
        cutlass::TensorRef<ElementOutput, LayoutOutput> tensor_d((cutlass::half_t*)m1m2, LayoutInputA::packed(problem_size.mn()));

        // Initialize alpha and beta for dot product computation
        ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
        ElementComputeEpilogue beta = ElementComputeEpilogue(0);

        // Split K dimension into 1 partitions
        int split_k_slices = 1;

        NCCLChunk* dNCCLChunks;
        const int numNCCLChunks = 24 * 128 * 512;
        printf("sizeof cunkinfo %ld\n", sizeof(NCCLChunk) * numNCCLChunks);
        CUDACHECK(cudaMalloc(&dNCCLChunks, sizeof(NCCLChunk) * numNCCLChunks));
        
        ncclCustomCollective2DInfo(dNCCLChunks, N, (M*N)/comm_size, ncclHalf, comm, stream);
        CUDACHECK(cudaDeviceSynchronize());
        printf("204\n");
        std::vector<NCCLChunk> hNCCLChunks = std::vector<NCCLChunk>(numNCCLChunks);
        CUDACHECK(cudaMemcpy(&hNCCLChunks[0], dNCCLChunks, sizeof(NCCLChunk) * numNCCLChunks, cudaMemcpyDeviceToHost));
        CUDACHECK(cudaFree(dNCCLChunks));

        typename SCCLGemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                          tensor_a,  // <- reference to matrix A on device
                                          tensor_b,  // <- reference to matrix B on device
                                          tensor_c,  // <- reference to matrix C on device
                                          tensor_d,  // <- reference to matrix D on device
                                          scclAlgo,
                                          hNCCLChunks,
                                          {alpha, beta},          // <- tuple of alpha and beta
                                          split_k_slices};        // <- k-dimension split factor


        // Instantiate CUTLASS kernel depending on templates
        SCCLGemm gemm_op;

        // Using the arguments, query for extra workspace required for matrix multiplication computation
        size_t workspace_size = gemm_op.get_workspace_size(arguments);

        // Allocate workspace memory
        cutlass::device_memory::allocation<uint8_t> workspace(workspace_size);



        // Check the problem size is supported or not 
        cutlass::Status status = gemm_op.can_implement(arguments);
        CUTLASS_CHECK(status)
        status = gemm_op.initialize(arguments, workIndex, workspace.get());
        CUTLASS_CHECK(status);
// cudaProfilerStart();
          // CUDACHECK(cudaFuncSetAttribute(dummyKernel<80>,
          //                           cudaFuncAttributeMaxDynamicSharedMemorySize,
          //                           96*1024));
        float minSampleTime = 10000000.0f;
        float cutlassTime = 0;
        float sampleTime;

        int gemmParts = 32;
        int startParts = 24;
        int lastParts = gemmParts - startParts;
        float firstCutlassTime = 0, firstCutlassT2=0, firstCutlassT1 = 0;
        for(int iter = 0; iter < 110; iter++) {
          //CUDACHECK(cudaMemset(tileIdx, 0, sizeof(int)));

          // CUDACHECK(cudaMemset(tileStatusMap, 0, numTiles * sizeof(int)));
 
          if (rank == 0 && iter %20 == 0)
            printf("iter %d\n", iter);
          cudaEvent_t startpipe, stoppipe;
          cudaEvent_t cutlassStartPipe, cutlassStopPipe;
          float elapsedTimepipe, cutlassElapsedTimepipe;
          // MPI_Barrier(MPI_COMM_WORLD);

          CUDACHECK(cudaEventCreate(&startpipe));
          CUDACHECK(cudaEventCreate(&stoppipe));
          CUDACHECK(cudaEventCreate(&cutlassStartPipe));
          CUDACHECK(cudaEventCreate(&cutlassStopPipe));
          CUDACHECK(cudaEventRecord(startpipe, stream));
          CUDACHECK(cudaEventRecord(cutlassStartPipe, cutlassStream));

          double t1 = getCurrentTime();     
                    
          status = gemm_op(cutlassStream);
          CUTLASS_CHECK(status);

          CUDACHECK(cudaDeviceSynchronize());
          NCCLCHECK(ncclCustomCollective2D((const void*)m1m2, 
                  (void*)m1m2, N, (M*N)/comm_size, ncclHalf, comm, stream));
          CUDACHECK(cudaDeviceSynchronize());
          // NCCLCHECK(ncclAllReduceMatrix(m1m2, M*N, M, N, N, ncclHalf, ncclSum, comm, stream));

          // Wait for kernels to finish
          // CUDACHECK(cudaDeviceSynchronize());

          // CUBLASCHECK(cublasGemmEx(handleWithCutlassStream, CUBLAS_OP_N, CUBLAS_OP_N, 
          // N, M, K, 
          // dAlpha,
          // m2, CUDA_R_16F, N,
          // m1, CUDA_R_16F, K,
          // dBeta, 
          // m1m2, CUDA_R_16F, N,
          // CUDA_R_16F, CUBLAS_GEMM_DFALT_TENSOR_OP));
          
          // Check processed order of tiles by cutlass.
          // CUDACHECK(cudaDeviceSynchronize());
          // int* hTileProcessedOrder = new int[numTiles*2];
          // CUDACHECK(cudaMemcpy(hTileProcessedOrder, tileStatusMap + numTiles, 2*numTiles*sizeof(int), cudaMemcpyDeviceToHost));
          // if (true) {
          //   for (int i = 0; i < numTiles; i++) {
          //     if (hTileProcessedOrder[2*i] != tileOrder[2*i]) {
          //       printf("1392: hTileProcessedOrder[%d] = %d, tileOder[%d] = %d\n", i, hTileProcessedOrder[2*i], i, tileOrder[2*i]);
          //     }
          //     if (hTileProcessedOrder[2*i + 1] != tileOrder[2*i + 1]) {
          //       printf("1396: hTileProcessedOrder[%d] = %d\n", i, hTileProcessedOrder[i]);
          //       break;
          //     }
          //   }
          // }

          CUDACHECK(cudaEventRecord(cutlassStopPipe, cutlassStream));
          CUDACHECK(cudaEventSynchronize(cutlassStopPipe));
          CUDACHECK(cudaEventElapsedTime(&cutlassElapsedTimepipe, cutlassStartPipe,cutlassStopPipe));
          // printf("cutlassElapsedTimepipe %f\n", cutlassElapsedTimepipe);
          CUDACHECK(cudaEventRecord(stoppipe, stream));
          CUDACHECK(cudaEventSynchronize(stoppipe));
          double t2 = getCurrentTime();

          CUDACHECK(cudaEventElapsedTime(&elapsedTimepipe, startpipe,stoppipe));
          CUDACHECK(cudaEventElapsedTime(&cutlassElapsedTimepipe, cutlassStartPipe,cutlassStopPipe));
          
          if (iter >= 10) {
            totalTime += (t2-t1)*1000.0f;
            allReduceTime += elapsedTimepipe;
            cutlassTime += cutlassElapsedTimepipe;
            sampleTime += (t2-t1)*1000.0f;
            // firstCutlassTime += (firstCutlassT2 - firstCutlassT1)*1000.0f;

            if (iter > 10 && iter % 10 == 0) {
              minSampleTime = std::min(minSampleTime, sampleTime*10);
              sampleTime = 0;//(t2-t1)*1000.0f;
            }
          }
          workIndex++;
          if (iter == 0) 
          { 
            MPI_Barrier(MPI_COMM_WORLD);
            float *hm1 = new float[M*K];
            float *hm2 = new float[N*K];
            float *hm1m2 = new float[M*N];
            
            cudaMemcpyHalfDevice2FloatHost(hm1, m1, M*K);
            cudaMemcpyHalfDevice2FloatHost(hm2, m2, N*K);
            cudaMemcpyHalfDevice2FloatHost(hm1m2, m1m2, M*N);
            
            if (rank == 0)
              printf("[%d]: checking results at iter %d\n", rank, iter);

            if (!mpiRef(hm1, hm2, hm1m2, M, N, K, comm_size, rank))
              assert(false);
          }
        }
// cudaProfilerStop();

        MPI_Barrier(MPI_COMM_WORLD);

        if (rank == 0)
          printf("rank %d Overlapped(AllReduce, cutlass) Time: %f ms cutlass: %f ms, allreduceTime: %f ms, minSampleTime: %f ms firstCutlassT: %f ms\n", rank, totalTime, cutlassTime, allReduceTime, minSampleTime, firstCutlassTime);
        
        // printf("rank %d cutlass %f\n", rank, cutlassTime);
      }

      CUDACHECK(cudaFree(m1));
      CUDACHECK(cudaFree(m2));
      CUDACHECK(cudaFree(m1m2));
    }
  }


  MPI_Finalize();
}