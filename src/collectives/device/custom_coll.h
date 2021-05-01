/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #include "devcomm.h"
 #include "primitives.h"
 #include "collectives.h"
 #include "sckl_interpreter.h"

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomColl, NCCL_ALGO_RING, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomColl, NCCL_ALGO_TREE, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<int PROTO, class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomColl, NCCL_ALGO_COLLNET, PROTO, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {}
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomColl, NCCL_ALGO_SCKL, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      SCKLFunctionSimple<FUNC, T, UNROLL> scklfunc;
      scklfunc.run(args);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomColl, NCCL_ALGO_SCKL, NCCL_PROTO_LL128, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      SCKLFunctionLL128<FUNC, T, UNROLL> scklfunc;
      scklfunc.run(args);
    }
};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomColl, NCCL_ALGO_SCKL, NCCL_PROTO_LL, FUNC, T, UNROLL> {
  public:
    __device__ void run(struct ncclWorkElem* args) {
      SCKLFunctionLL<FUNC, T, UNROLL> scklfunc;
      scklfunc.run(args);
    }
};