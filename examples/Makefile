all: test-sccl

test-sccl: test-sccl.cpp
	mpicxx $< -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -o $@ -Wall -lcurand -std=c++11  -O0 -g

test-sccl-cutlass: test-sccl-cutlass.cu cutlass-matmul.h /msrhyper-ddn/hai8/saemal/abhinav/cutlass/include/cutlass/gemm/device/scclgemm.h
	nvcc test-sccl-cutlass.cu -arch=sm_70 -O3 -Xptxas -O3 -DNDEBUG -std=c++11 -Xcompiler -fopenmp,-O3 -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/mpi -I.. \
	-I/usr/local/cuda/include/ -I ../build/include/ -I ../../cutlass/include -I ../../cutlass/tools/util/include -I ../../cutlass/examples/common -L../build/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c && \
	mpicxx test-sccl-cutlass.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o test-sccl-cutlass

test-sccl-cutlass-run: test-sccl-cutlass
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1 ./test-sccl-cutlass

test-sccl-cutlass-2-run-2-channels: test-sccl-cutlass
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2 ./test-sccl-cutlass

test-sccl-cutlass-4-run-4-channels: test-sccl-cutlass
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=4 -x NCCL_MAX_NCHANNELS=4 ./test-sccl-cutlass

test-sccl-cutlass-16-run-16-channels: test-sccl-cutlass
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 ./test-sccl-cutlass

test-sccl-cutlass-4-run-16-channels: test-sccl-cutlass
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 ./test-sccl-cutlass

test-sccl-cutlass-16-run-24-channels: test-sccl-cutlass
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_BUFFSIZE=6291456 -x NCCL_THREADS=256 -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=24 -x NCCL_MAX_NCHANNELS=24 ./test-sccl-cutlass


matmul-allreduce-release: matmul-allreduce.cu
	nvcc matmul-allreduce.cu -arch=sm_70 -O3 -Xptxas -v,-O3 -DNDEBUG -std=c++11 -Xcompiler -fopenmp,-O3 -lcudadevrt -lcudart -I/usr/include/x86_64-linux-gnu/mpi -I.. \
	-I/usr/local/cuda/include/ -I ../build/include/ -I ../../cutlass/include -I ../../cutlass/tools/util/include -I ../../cutlass/examples/common -L../build/lib/ -L/usr/local/cuda/lib64/ -lnccl -lcublas -lcurand -c && \
	mpicxx matmul-allreduce.o -I/usr/local/cuda/include/ -I../build/include/ -L../build/lib/ -L/usr/local/cuda/lib64/ -lcudart -lnccl -lcublas -Wall -lcurand -lcudadevrt -std=c++11 -fopenmp -o matmul-allreduce

matmul-allreduce-run: matmul-allreduce-release
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1 ./matmul-allreduce

matmul-allreduce-2-run-2-channels: matmul-allreduce-release
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2 ./matmul-allreduce

matmul-allreduce-16-run-24-channels: matmul-allreduce-release
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=24 -x NCCL_MAX_NCHANNELS=24 ./matmul-allreduce

test-sccl-run: test-sccl
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1 ./test-sccl

test-sccl-2-run-2-channels: test-sccl
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2 ./test-sccl

test-sccl-2-run-6-channels: test-sccl
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=6 -x NCCL_MAX_NCHANNELS=6 ./test-sccl

test-sccl-2-run-3-channels: test-sccl
	mpirun -np 2 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=3 -x NCCL_MAX_NCHANNELS=3 ./test-sccl

test-sccl-4-run: test-sccl
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=1 -x NCCL_MAX_NCHANNELS=1 ./test-sccl

test-sccl-4-run-2-channels: test-sccl
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=2 -x NCCL_MAX_NCHANNELS=2 ./test-sccl

test-sccl-4-run-4-channels: test-sccl
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=4 -x NCCL_MAX_NCHANNELS=4 ./test-sccl

test-sccl-4-run-16-channels: test-sccl
	mpirun -np 4 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 ./test-sccl

test-sccl-16-run-12-channels: test-sccl
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 ./test-sccl

test-sccl-16-run-24-channels: test-sccl
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_BUFFSIZE=6291456 -x NCCL_THREADS=256 -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=24 -x NCCL_MAX_NCHANNELS=24 ./test-sccl

test-sccl-16-run-16-channels: test-sccl
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=SCCL -x NCCL_MIN_NCHANNELS=16 -x NCCL_MAX_NCHANNELS=16 ./test-sccl

test-allreduce-run: test-sccl
	mpirun -np 16 -x LD_LIBRARY_PATH="../build/lib/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH" -x NCCL_INFO=DEBUG -x NCCL_ALGO=Ring -x NCCL_PROTO=Simple -x NCCL_MIN_NCHANNELS=12 -x NCCL_MAX_NCHANNELS=12 ./test-sccl
