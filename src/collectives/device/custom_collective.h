/*************************************************************************
 * Copyright (c) 2015-2020, NVIDIA CORPORATION. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

 #include "sccl_interpreter.h"
 
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_SCCL, NCCL_PROTO_SIMPLE, FUNC, T, UNROLL> : public scclFunction<T, SimpleWrapper<FUNC, T, UNROLL>, false> {};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_SCCL, NCCL_PROTO_LL128, FUNC, T, UNROLL> : public scclFunction<T, LL128Wrapper<FUNC, T>> {};

template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_SCCL, NCCL_PROTO_LL, FUNC, T, UNROLL> : public scclFunction<T, LLWrapper<FUNC, T>> {};

//FIXME: Find a way to remove below declarations for RING, TREE, and COLLNET.
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_RING, NCCL_PROTO_LL, FUNC, T, UNROLL> : public scclFunction<T, LLWrapper<FUNC, T>> {};
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_TREE, NCCL_PROTO_LL, FUNC, T, UNROLL> : public scclFunction<T, LLWrapper<FUNC, T>> {};
template<class FUNC, typename T, int UNROLL>
class ncclFunction<ncclFuncCustomCollective, NCCL_ALGO_COLLNET, NCCL_PROTO_LL, FUNC, T, UNROLL> : public scclFunction<T, LLWrapper<FUNC, T>> {};