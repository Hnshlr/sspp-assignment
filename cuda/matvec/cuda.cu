#include <iostream>
#include <random>
#include <cuda_runtime.h>  // For CUDA runtime API
#include <helper_cuda.h>  // For checkCudaError macro
#include <helper_timer.h>  // For CUDA SDK timers
#include "mmio.h"

#define BDIM 1024
#define XBD 32
#define YBD 32
const dim3 BLOCK_DIM(BDIM);
const dim3 BLOCK_DIM2(XBD,YBD);

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
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double t = 0.0;
        for (int i = d_IRP[row]; i < d_IRP[row + 1]; i++) {
            t += d_AS[i] * d_vector[d_JA[i]];
        }
        d_csr_result[row] = t;
    }
}
__global__ void ellpack_kernel(int *d_JA, double *d_AS, int *d_vector, double *d_ellpack_result, int M, int max_row_length) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M) {
        double t = 0.0;
        for (int i = 0; i < max_row_length; i++) {
            int col = d_JA[row * max_row_length + i];
            double val = d_AS[row * max_row_length + i];
            t += val * d_vector[col];
        }
        d_ellpack_result[row] = t;
    }
}

__device__ void rowReduce(volatile double *sdata, int tid, int s) {
    switch(s){
    case 16:  sdata[tid] += sdata[tid + 16];
    case  8:  sdata[tid] += sdata[tid +  8];
    case  4:  sdata[tid] += sdata[tid +  4];
    case  2:  sdata[tid] += sdata[tid +  2];
    case  1:  sdata[tid] += sdata[tid +  1];
    }
}
__global__ void csr_kernel_2d(int *d_IRP, int *d_JA, double *d_AS, int *d_vector, double *d_csr_result, int M, int nz) {
    __shared__ double ax[YBD][XBD];
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int row = blockIdx.x*blockDim.y + tr;
    int s;
    ax[tr][tc] = 0.0;

    if (row < M) {
        double t = 0.0;
        int idx = d_IRP[row];
        int end = d_IRP[row+1];
        for (int j=idx+tc; j<end; j+=XBD){
            t += d_AS[j]*d_vector[d_JA[j]];
        }
        ax[tr][tc] = t;
    }
    __syncthreads();

    for (s=XBD/2; s>=32; s>>=1){ // unroll for efficiency
        if (tc<s)
            ax[tr][tc] += ax[tr][tc+s];
        __syncthreads();
    }

    s = min(16,XBD/2);
    if (tc < s) rowReduce(&(ax[tr][0]), tc, s);

    if ((tc == 0) && (row < M)) {
        d_csr_result[row] = ax[tr][tc];
    }
}
__global__ void ellpack_kernel_2d(int *d_JA, double *d_AS, int *d_vector, double *d_ellpack_result, int M, int max_row_length) {
    __shared__ double ax[YBD][XBD];
    int tr = threadIdx.y;
    int tc = threadIdx.x;
    int row = blockIdx.x*blockDim.y + tr;
    int s;
    ax[tr][tc] = 0.0;

    if (row < M) {
        double t = 0.0;
        for (int i=tc; i<max_row_length; i+=XBD){
            int col = d_JA[row*max_row_length+i];
            double val = d_AS[row*max_row_length+i];
            t += val*d_vector[col];
        }
        ax[tr][tc] = t;
    }
    __syncthreads();

    for (s=XBD/2; s>=32; s>>=1){ // unroll for efficiency
        if (tc<s)
            ax[tr][tc] += ax[tr][tc+s];
        __syncthreads();
    }

    s = min(16,XBD/2);
    if (tc < s) rowReduce(&(ax[tr][0]), tc, s);

    if ((tc == 0) && (row < M)) {
        d_ellpack_result[row] = ax[tr][tc];
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
    else if ((f = fopen(argv[1], "r")) == NULL) { fprintf(stderr, "ERROR: Could not open the .mtx file.\n"); exit(1); }
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


    // CONSTRUCT MATRIX:
    // ALLOCATE MEMORY FOR THE SPARSE MATRIX:
    I = (int *) malloc(2*nz * sizeof(int));
    J = (int *) malloc(2*nz * sizeof(int));
    val = (double *) malloc(2*nz * sizeof(double));
    // READ THE SPARSE MATRIX, AND REMOVE THE REMAINING ZEROES:
    int lineCounter = 0;
    while (fscanf(f, "%d %d %lg\n", &I[lineCounter], &J[lineCounter], &val[lineCounter]) != EOF) {
        I[lineCounter]--;  /* adjust from 1-based to 0-based */
        J[lineCounter]--;
        lineCounter++;
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

    // METHOD 1: CSR:
    if ((optype == "csr") || (optype == "all")) {
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
        // METHOD 1-0: SERIAL CSR:
        int row, col;
        auto *serial_csr_result = (double *) malloc(M * sizeof(double));
        auto *csr_result = (double *) malloc(M * sizeof(double));
        StopWatchInterface *timer = 0;
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
        std::cout << "RESULTS = { METHOD=serial, FORMAT=CSR, TIME=" << time << ", GFLOPS=" << gflops << " }"
                  << std::endl;

        // ______________________________________________________________________________________________________________ //

        // METHOD 1-2: CUDA CSR:
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
        const dim3 GRID_DIM((M + BLOCK_DIM.x - 1) / BLOCK_DIM.x, 1);
        const dim3 GRID_DIM2((M + BLOCK_DIM2.y - 1) / BLOCK_DIM2.y, 1);
        timer->reset();
        timer->start();
//        csr_kernel <<< GRID_DIM, BLOCK_DIM >>>(d_IRP, d_JA, d_AS, d_vector, d_csr_result, M, nz);
        csr_kernel_2d <<< GRID_DIM2, BLOCK_DIM2 >>>(d_IRP, d_JA, d_AS, d_vector, d_csr_result, M, nz);
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
            if (maxAbs == 0) maxAbs = 1;
            relDiff = std::max(relDiff, std::abs(serial_csr_result[row] - csr_result[row]) / maxAbs);
            maxDiff = std::max(maxDiff, std::abs(serial_csr_result[row] - csr_result[row]));
        }
        std::cout << "RESULTS = { METHOD=cuda, FORMAT=CSR, TIME=" << time << ", GFLOPS=" << gflops << ", MAX_DIFF="
                  << maxDiff << ", REL_DIFF=" << relDiff << " }" << std::endl;
        // FREE THE GPU MEMORY:
        checkCudaErrors(cudaFree(d_IRP));
        checkCudaErrors(cudaFree(d_JA));
        checkCudaErrors(cudaFree(d_AS));
        checkCudaErrors(cudaFree(d_vector));
        checkCudaErrors(cudaFree(d_csr_result));
        // FREE THE CPU MEMORY:
        free(IRP);
        free(JA);
        free(AS);
    }

    // ______________________________________________________________________________________________________________ //

    // METHOD 2: ELLPACK:
    if (optype == "ellpack" || optype == "all") {
        int max_row_length = 0;
        int *max_row_lengths = (int *) malloc(M * sizeof(int));
        for (int i = 0; i < M; i++) {
            max_row_lengths[i] = 0;
        }
        // Find the maximum row length, which is the max number of non-zero elements in a row, using I, J and val; sorted by J:
        for (int i = 0; i < nz; i++) {
            max_row_lengths[I[i]]++;
        }
        for (int i = 0; i < M; i++) {
            if (max_row_lengths[i] > max_row_length) {
                max_row_length = max_row_lengths[i];
            }
        }
        int **JA = (int **) malloc(M * sizeof(int *));
        for (int i = 0; i < M; i++) {
            JA[i] = (int *) malloc(max_row_length * sizeof(int));
            for (int j = 0; j < max_row_length; j++) {
                JA[i][j] = 0;
            }
        }
        auto **AS = (double **) malloc(M * sizeof(double *));
        for (int i = 0; i < M; i++) {
            AS[i] = (double *) malloc(max_row_length * sizeof(double));
            for (int j = 0; j < max_row_length; j++) {
                AS[i][j] = 0.0;
            }
        }
        int *row_fill = (int *) malloc(M * sizeof(int));
        for (int i = 0; i < M; i++) {
            row_fill[i] = 0;
        }
        int row, col;
        for (int i = 0; i < nz; i++) {
            row = I[i];
            col = J[i];
            JA[row][row_fill[row]] = col;
            AS[row][row_fill[row]] = val[i];
            row_fill[row]++;
        }

        // METHOD 2-1: SERIAL ELLPACK:
        auto *serial_ellpack_result = (double *) malloc(M * sizeof(double));
        auto *ellpack_result = (double *) malloc(M * sizeof(double));
        StopWatchInterface *timer = 0;
        sdkCreateTimer(&timer);
        timer->start();
        for (row = 0; row < M; row++) {
            double sum = 0;
            for (int j = 0; j < max_row_length; j++) {
                sum = sum + AS[row][j] * vector[JA[row][j]];
            }
            serial_ellpack_result[row] = sum;
        }
        timer->stop();
        double time = timer->getTime() / 1000.0;
        double gflops = 2.0 * nz / (time * 1e9);
        std::cout << "RESULTS = { METHOD=serial, FORMAT=ELLPACK, TIME=" << time << ", GFLOPS=" << gflops << " }" << std::endl;


        // METHOD 2-2: CUDA ELLPACK:
        int *d_JA, *d_vector;
        double *d_AS, *d_ellpack_result;
        // FLATTEN JA AND AS:
        int *JA_flat = (int *) malloc(M * max_row_length * sizeof(int));
        double *AS_flat = (double *) malloc(M * max_row_length * sizeof(double));
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < max_row_length; j++) {
                JA_flat[i * max_row_length + j] = JA[i][j];
                AS_flat[i * max_row_length + j] = AS[i][j];
            }
        }
        // ALLOCATE MEMORY ON DEVICE:
        checkCudaErrors(cudaMalloc((void **) &d_JA, M * max_row_length * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &d_AS, M * max_row_length * sizeof(double)));
        checkCudaErrors(cudaMalloc((void **) &d_vector, M * sizeof(int)));
        checkCudaErrors(cudaMalloc((void **) &d_ellpack_result, M * sizeof(double)));
        checkCudaErrors(cudaMemcpy(d_JA, JA_flat, M * max_row_length * sizeof(int), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_AS, AS_flat, M * max_row_length * sizeof(double), cudaMemcpyHostToDevice));
        checkCudaErrors(cudaMemcpy(d_vector, vector, M * sizeof(int), cudaMemcpyHostToDevice));
        // CALCULATE THE GRID DIMENSION. A 1D GRID SUFFICES:
//        const dim3 GRID_DIM((M + BLOCK_DIM.x - 1) / BLOCK_DIM.x, 1);
        const dim3 GRID_DIM2((M + BLOCK_DIM2.y - 1) / BLOCK_DIM2.y, 1);
        timer->reset();
        timer->start();
//        ellpack_kernel <<< GRID_DIM, BLOCK_DIM >>>(d_JA, d_AS, d_vector, d_ellpack_result, M, max_row_length);
        ellpack_kernel_2d <<< GRID_DIM2, BLOCK_DIM2 >>>(d_JA, d_AS, d_vector, d_ellpack_result, M, max_row_length);
        checkCudaErrors(cudaDeviceSynchronize());
        timer->stop();
        time = timer->getTime() / 1000.0;
        gflops = 2.0 * nz / time / 1e9;
        // COPY THE RESULT BACK TO THE CPU:
        checkCudaErrors(cudaMemcpy(ellpack_result, d_ellpack_result, M * sizeof(double), cudaMemcpyDeviceToHost));
        // MAX AND REL DIFF:
        double maxDiff = 0.0;
        double relDiff = 0.0;
        for (int row = 0; row < M; row++) {
            float maxAbs = std::max(std::abs(serial_ellpack_result[row]), std::abs(ellpack_result[row]));
            if (maxAbs == 0) maxAbs = 1;
            relDiff = std::max(relDiff, std::abs(serial_ellpack_result[row] - ellpack_result[row]) / maxAbs);
            maxDiff = std::max(maxDiff, std::abs(serial_ellpack_result[row] - ellpack_result[row]));
        }
        std::cout << "RESULTS = { METHOD=cuda, FORMAT=ELLPACK, TIME=" << time << ", GFLOPS=" << gflops << ", MAX_DIFF=" << maxDiff << ", REL_DIFF=" << relDiff << " }" << std::endl;
        // FREE THE GPU MEMORY:
        checkCudaErrors(cudaFree(d_JA));
        checkCudaErrors(cudaFree(d_AS));
        checkCudaErrors(cudaFree(d_vector));
        checkCudaErrors(cudaFree(d_ellpack_result));
    }

    return 0;
}

