all: test-sccl

test-sccl: test-sccl.cpp
	mpicxx $< -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o $@ -Wall -lcurand -std=c++11  -O0 -g


test-sccl-run: test-sccl
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1 ./test-sccl

test-sccl-4-run: test-sccl
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1 ./test-sccl

test-sccl-4-run-2-channels: test-sccl
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2 ./test-sccl

test-sccl-16-run-12-channels: test-sccl
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 ./test-sccl

test-sccl-16-run: test-sccl
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x SCCL_XML_FILE=allgather_ring_2_ranks_1_channel.xml -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 ./test-sccl

test-allreduce-run: test-sccl
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 ./test-sccl
