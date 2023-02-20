#include <iostream>
#include <random>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include "mmio.h"

// The max number of threads per block is 1024
// For a first implementation, use a 1d block of 1024 threads; where each thread computes one element of the result vector
#define BDIM 1024
const dim3 BLOCK_DIM(BDIM);

double** build_matrix(int M, int N, int nz, const int *I, const int *J, const double *val) {
    auto** matrix = new double*[M];
    for (int i = 0; i < M; i++) {
        matrix[i] = new double[N];
        for (int j = 0; j < N; j++) {
            matrix[i][j] = 0.0;
        }
    }
    for (int i = 0; i < nz; i++) {
        matrix[I[i]][J[i]] = val[i];
    }
    return matrix;
}
__global__ void csr_kernel(int *d_IRP, int *d_JA, double *d_AS, int *d_vector, double *d_csr_result, int M, int nz) {
    // NOTE: Each block is of dimension 1, and size BDIM:
    // Each thread computes one element of the result vector, ie one row of the result vector.
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double t = 0.0;
        for (int i = d_IRP[row]; i < d_IRP[row + 1]; i++) {
            t += d_AS[i] * d_vector[d_JA[i]];
        }
        d_csr_result[row] = t;
    }
}

int main(int argc, char** argv) {

    // MATRIX SETTINGS:
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int *I, *J;
    double *val;

    // SETTINGS (MATRIX PATH):
    if (argc < 2) { fprintf(stderr, "ERROR: Please specify a valid .mtx path.\n"); exit(1); }
    else if ((f = fopen(argv[1], "r")) == NULL) exit(1);
    if (mm_read_banner(f, &matcode) != 0) { printf("ERROR: Could not process Matrix Market banner.\n"); exit(1); }
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)) { printf("ERROR: Sorry, this application does not support "); printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode)); exit(1); }
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) exit(1);

    // SETTINGS (OPERATION TYPE):
    std::string optype = argv[2];
    if ((optype == "CSR") || (optype == "csr") || (optype == "ELLPACK") || (optype == "ellpack") || (optype == "ALL") || (optype == "all")) {
        if (optype == "CSR") {
            optype = "csr";
        } else if (optype == "ELLPACK") {
            optype = "ellpack";
        }
        else if (optype == "ALL") {
            optype = "all";
        }
    } else {
        printf("ERROR: Wrong format. Please choose between CSR and ELLPACK.");
        exit(1);
    }

    // SETTINGS (ANNOUNCEMENT):
    std::string matrix_name = argv[1];
    matrix_name = matrix_name.substr(matrix_name.find_last_of("/\\") + 1);
//    printf("SETTINGS = { LIBRARY=CUDA, OP=Mat/Vec, MATRIX=\"%s\", SIZE=%dx%d }\n", matrix_name.c_str(), M, N);
    std::cout << "SETTINGS = { LIBRARY=CUDA, OP=Mat/Vec, MATRIX=\"" << matrix_name << "\", SIZE=" << M << "x" << N << " }" << std::endl;


    // ______________________________________________________________________________________________________________ //

    // CONSTRUCT MATRIX:
    // ALLOCATE MEMORY FOR THE SPARSE MATRIX:
    I = (int *) malloc(2*nz * sizeof(int));
    J = (int *) malloc(2*nz * sizeof(int));
    val = (double *) malloc(2*nz * sizeof(double));
    // READ THE SPARSE MATRIX, AND REMOVE THE REMAINING ZEROES:
    int lineCounter = 0;
    while (fscanf(f, "%d %d %lg\n", &I[lineCounter], &J[lineCounter], &val[lineCounter]) != EOF) {
        if (val[lineCounter] != 0) {
            I[lineCounter]--;  /* adjust from 1-based to 0-based */
            J[lineCounter]--;
            lineCounter++;
        }
    }
    nz = lineCounter;
    I = (int *) realloc(I, 2*nz * sizeof(int));
    J = (int *) realloc(J, 2*nz * sizeof(int));
    val = (double *) realloc(val, 2*nz * sizeof(double));
    if(mm_is_symmetric(matcode)) {
        int symmCounter = 0;
        for (int i = 0; i < nz; i++) {
            if (I[i] != J[i]) {
                I[nz + symmCounter] = J[i];
                J[nz + symmCounter] = I[i];
                val[nz + symmCounter] = val[i];
                symmCounter++;
            }
        }
        nz = nz + symmCounter;
        I = (int *) realloc(I, nz * sizeof(int));
        J = (int *) realloc(J, nz * sizeof(int));
        val = (double *) realloc(val, nz * sizeof(double));
    }
    if (f !=stdin) fclose(f);

    // CONSTRUCT RANDOM VECTOR:
    int *vector = (int *) malloc(M * sizeof(int));
    // Fill the vector with random numbers, using C++11 random library:
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 10);
    for (int i = 0; i < M; i++) {
        vector[i] = dis(gen);
    }

    // ______________________________________________________________________________________________________________ //

    int *IRP = (int *) malloc((M + 1) * sizeof(int));
    int *JA = (int *) malloc(nz * sizeof(int));
    auto *AS = (double *) malloc(nz * sizeof(double));
    for (int i = 0; i < M + 1; i++) {
        IRP[i] = 0;
    }
    for (int i = 0; i < nz; i++) {
        IRP[I[i] + 1]++;
    }
    for (int i = 0; i < M; i++) {
        IRP[i + 1] += IRP[i];
    }
    for (int i = 0; i < nz; i++) {
        int row = I[i];
        int dest = IRP[row];
        JA[dest] = J[i];
        AS[dest] = val[i];
        IRP[row]++;
    }
    for (int i = M; i > 0; i--) {
        IRP[i] = IRP[i - 1];
    }
    IRP[0] = 0;

    // COMPUTATION:
    // METHOD 0: SERIAL:
    int row, col;
    auto *serial_csr_result = (double *) malloc(M * sizeof(double));
    auto *csr_result = (double *) malloc(M * sizeof(double));
    StopWatchInterface* timer = 0;
    sdkCreateTimer(&timer);
    timer->start();
    for (row = 0; row < M; row++) {
        double sum = 0;
        for (int i = IRP[row]; i < IRP[row + 1]; i++) {
            col = JA[i];
            sum += AS[i] * vector[col];
        }
        serial_csr_result[row] = sum;
    }
    timer->stop();
    double time = timer->getTime() / 1000.0;
    double gflops = 2.0 * nz / time / 1e9;
//    printf("RESULTS = { METHOD=serial, FORMAT=CSR, TIME=%f, GFLOPS=%f }\n", time, gflops);
    std::cout << "RESULTS = { METHOD=serial, FORMAT=CSR, TIME=" << time << ", GFLOPS=" << gflops << " }" << std::endl;

    // ______________________________________________________________________________________________________________ //

    // METHOD 1: CUDA CSR:
    // ALLOCATE MEMORY FOR THE SPARSE MATRIX:
    int *d_IRP, *d_JA, *d_vector;
    double *d_AS, *d_csr_result;
    checkCudaErrors(cudaMalloc((void **) &d_IRP, (M + 1) * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_JA, nz * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_AS, nz * sizeof(double)));
    checkCudaErrors(cudaMalloc((void **) &d_vector, M * sizeof(int)));
    checkCudaErrors(cudaMalloc((void **) &d_csr_result, M * sizeof(double)));
    // COPY THE SPARSE MATRIX TO THE GPU:
    checkCudaErrors(cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_JA, JA, nz * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_AS, AS, nz * sizeof(double), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_vector, vector, M * sizeof(int), cudaMemcpyHostToDevice));
    // COMPUTE THE MATRIX-VECTOR PRODUCT:
    // CALCULATE THE GRID DIMENSION. A 1D GRID SUFFICES:
    // Since there are M rows to compute, we need M threads. Each thread will compute one row.
    // Each block is of dimension BLOCK_DIM.x * 1. We need (M + BLOCK_DIM.x - 1) / BLOCK_DIM.x blocks.
    const dim3 GRID_DIM((M + BLOCK_DIM.x - 1) / BLOCK_DIM.x, 1, 1);
    timer->reset();
    timer->start();
    csr_kernel <<< GRID_DIM, BLOCK_DIM >>> (d_IRP, d_JA, d_AS, d_vector, d_csr_result, M, nz);
    checkCudaErrors(cudaDeviceSynchronize());
    timer->stop();
    time = timer->getTime() / 1000.0;
    gflops = 2.0 * nz / time / 1e9;
    // COPY THE RESULT BACK TO THE CPU:
    checkCudaErrors(cudaMemcpy(csr_result, d_csr_result, M * sizeof(double), cudaMemcpyDeviceToHost));
    // MAX AND REL DIFF:
    double maxDiff = 0.0;
    double relDiff = 0.0;
    for (int row = 0; row < M; row++) {
        float maxAbs = std::max(std::abs(serial_csr_result[row]), std::abs(csr_result[row]));
        if (maxAbs ==0) maxAbs = 1;
        relDiff = std::max(relDiff, std::abs(serial_csr_result[row] - csr_result[row]) / maxAbs);
        maxDiff = std::max(maxDiff, std::abs(serial_csr_result[row] - csr_result[row]));
    }
//    printf("RESULTS = { METHOD=cuda, FORMAT=CSR, TIME=%f, GFLOPS=%f, MAX_DIFF=%f, REL_DIFF=%f }\n",time, gflops, maxDiff, relDiff);
    std::cout << "RESULTS = { METHOD=cuda, FORMAT=CSR, TIME=" << time << ", GFLOPS=" << gflops << ", MAX_DIFF=" << maxDiff << ", REL_DIFF=" << relDiff << " }" << std::endl;
    // FREE THE GPU MEMORY:
    checkCudaErrors(cudaFree(d_IRP));
    checkCudaErrors(cudaFree(d_JA));
    checkCudaErrors(cudaFree(d_AS));
    checkCudaErrors(cudaFree(d_vector));
    checkCudaErrors(cudaFree(d_csr_result));

    // ______________________________________________________________________________________________________________ //

    return 0;
}
