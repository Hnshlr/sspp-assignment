cd /scratch/s388885/sspp/sspp-assignment/ && module load fosscuda/2019b && export CXX=$(which g++) && g++ -fopenmp -O4 -lgomp -std=c++11 main.cpp src/mmio.h src/mmio.c

# g++ -fopenmp -O4 -lgomp -std=c++11 main.cpp src/mmio.h src/mmio.c