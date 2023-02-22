cd /scratch/s388885/sspp/sspp-assignment/cuda && . env.sh && cmake . && cd matvec && make cuda

# g++ -fopenmp -O4 -lgomp -std=c++11 main.cpp src/mmio.h src/mmio.c