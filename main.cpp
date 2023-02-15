#include <omp.h>
#include <random>

#include "src/mmio.h"

//// TEMP: DEBUG PURPOSES FUNCTIONS:
void print_list(int nz, int *I, int *J, double *val) {
    int i;
    for (i=0; i<nz; i++) {
        printf("%d %d %20.19g\n", I[i] + 1, J[i] + 1, val[i]);
    }
}
void print_IRP(int M, int *IRP) {
    printf("IRP=[");
    for (int i = 0; i < M+1; i++) {
        printf("%d ", IRP[i]);
    }
    printf("]\n");
}
void print_JA(int nz, int *JA) {
    printf("JA=[");
    for (int i = 0; i < nz; i++) {
        printf("%d ", JA[i]);
    }
    printf("]\n");
}
void print_AS(int nz, double *AS) {
    printf("AS=[");
    for (int i = 0; i < nz; i++) {
        printf("%.2f ", AS[i]);
    }
    printf("]\n");
}
void print_matrix_using_arrays(int M, int N, int nz, const int *I, const int *J, const double *val) {
    printf("M=%d, N=%d, nz=%d\n", M, N, nz);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double vall = 0.0;
            for (int k = 0; k < nz; k++) {
                if (I[k] == i && J[k] == j) {
                    vall = val[k];
                    break;
                }
            }
            printf("%.2f ", vall);
        }
        printf("\n");
    }
}
void print_matrix(int M, int N, double **matrix) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}
void print_matrix_using_CSR(int M, int N, int nz, const int *IRP, const int *JA, const double *AS) {
    printf("M=%d, N=%d, nz=%d\n", M, N, nz);
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double val = 0.0;
            for (int k = IRP[i]; k < IRP[i+1]; k++) {
                if (JA[k] == j) {
                    val = AS[k];
                    break;
                }
            }
            printf("%.2f ", val);
        }
        printf("\n");
    }
}
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
//// END OF TEMP: DEBUG PURPOSES FUNCTIONS.

// RUN FRONTEND: ./a.out "src/data/input/Cube_Coup_dt0/Cube_Coup_dt0.mtx" "CSR"

int main(int argc, char *argv[]) {
    //// DEBUG PURPOSES:
//    char* path_cage4 = "../src/data/input/cage4/cage4.mtx";
//    char* path_cubecoup = "../src/data/input/Cube_Coup_dt0/Cube_Coup_dt0.mtx";
//    argc+=2;                        // TEMP
//    argv[1] = path_cage4;        // TEMP
//    argv[2] = "ALL";            // TEMP
    //// END OF DEBUG PURPOSES.

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
    printf("SETTINGS = { LIBRARY=OpenMP, OP=Mat/Vec, MATRIX=\"%s\", SIZE=%dx%d, THREADS=%d }\n", matrix_name.c_str(), M, N, omp_get_max_threads());

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

    // WRITE THE SPARSE MATRIX TO OUTPUT FILE:
//    mm_write_banner(stdout, matcode);
//    mm_write_mtx_crd_size(stdout, M, N, nz);
//    for (int i=0; i<nz; i++) fprintf(stdout, "%d %d %20.19g\n", I[i], J[i], val[i]);


    // CONVERT THE MATRIX TO CSR FORMAT:
    if (optype == "csr" || optype == "all") {
        int *IRP = (int *) malloc((M + 1) * sizeof(int));
        int *JA = (int *) malloc(nz * sizeof(int));
        auto *AS = (double *) malloc(nz * sizeof(double));
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
        auto *csr_result = (double *) malloc(M * sizeof(double));
        double start = omp_get_wtime();
        for (row = 0; row < M; row++) {
            double sum = 0;
            for (int i = IRP[row]; i < IRP[row + 1]; i++) {
                col = JA[i];
                sum += AS[i] * vector[col];
            }
            csr_result[row] = sum;
        }
        double end = omp_get_wtime();
        double time = end - start;
        double gflops = 2.0 * nz / (end - start) / 1e9;
        printf("RESULTS = { METHOD=serial, FORMAT=CSR, TIME=%f, GFLOPS=%f }\n", time, gflops);


        // METHOD 1: PARALLEL WITHOUT ANY OPTIMIZATIONS:
        double bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            csr_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, N, nz, IRP, JA, AS, vector, csr_result) private(row, col)
            for (row = 0; row < M; row++) {
                double sum = 0;
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    sum = sum + AS[j] * vector[JA[j]];
                }
                csr_result[row] = sum;
            }
            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=parallel, FORMAT=CSR, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n",
               bestTimeAfter10Trials, omp_get_max_threads(), gflops);

        // METHOD 2: VARIOUS CHUNK SIZES:
        std::vector<int> chunk_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536};
        int bestChunkSize = 0;
        double bestTime = 1000000;
        for (int chunk_size: chunk_sizes) {
            bestTimeAfter10Trials = 1000000;
            for (int i = 0; i < 10; i++) {
                // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
                csr_result = (double *) malloc(M * sizeof(double));
                row = 0;
                col = 0;
                // PARALLEL TIMER START:
                start = omp_get_wtime();
#pragma omp parallel for schedule(static, chunk_size) default(none) shared(M, N, nz, IRP, JA, AS, vector, csr_result, chunk_size) private(row, col)
                for (row = 0; row < M; row++) {
                    double sum = 0;
                    for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                        sum = sum + AS[j] * vector[JA[j]];
                    }
                    csr_result[row] = sum;
                }
                // STOP TIMER(S):
                end = omp_get_wtime();
                // TAKE THE BEST TIME AFTER 10 TRIALS:
                if (end - start < bestTimeAfter10Trials) {
                    bestTimeAfter10Trials = end - start;
                }
            }
            // TAKE THE BEST CHUNK SIZE:
            if (bestTimeAfter10Trials < bestTime) {
                bestTime = bestTimeAfter10Trials;
                bestChunkSize = chunk_size;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTime * 1e9);
        printf("RESULTS = { METHOD=chunks, FORMAT=CSR, BEST_TIME(10)=%fs, THREADS=%d, CHUNK_SIZE(BEST)=%d, GFLOPS=%f }\n", bestTime,
               omp_get_max_threads(), bestChunkSize, gflops);


        // METHOD 3: UNROLLING 2:
        bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            csr_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, N, nz, IRP, JA, AS, vector, csr_result) private(row, col)
            for (row = 0; row < M - M % 2; row += 2) {
                double sum1 = 0;
                double sum2 = 0;
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    sum1 = sum1 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 1]; j < IRP[row + 2]; j++) {
                    sum2 = sum2 + AS[j] * vector[JA[j]];
                }
                csr_result[row] = sum1;
                csr_result[row + 1] = sum2;
            }
            // HANDLE THE REMAINING ROWS:
            for (row = M - M % 2; row < M; row++) {
                double sum = 0;
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    sum = sum + AS[j] * vector[JA[j]];
                }
                csr_result[row] = sum;
            }
            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=unroll-2, FORMAT=CSR, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n", bestTimeAfter10Trials,
               omp_get_max_threads(), gflops);


        // METHOD 4: UNROLLING 4:
        bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            csr_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, N, nz, IRP, JA, AS, vector, csr_result) private(row, col)
            for (row = 0; row < M - M % 4; row += 4) {
                double sum1 = 0;
                double sum2 = 0;
                double sum3 = 0;
                double sum4 = 0;
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    sum1 = sum1 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 1]; j < IRP[row + 2]; j++) {
                    sum2 = sum2 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 2]; j < IRP[row + 3]; j++) {
                    sum3 = sum3 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 3]; j < IRP[row + 4]; j++) {
                    sum4 = sum4 + AS[j] * vector[JA[j]];
                }
                csr_result[row] = sum1;
                csr_result[row + 1] = sum2;
                csr_result[row + 2] = sum3;
                csr_result[row + 3] = sum4;
            }
            // HANDLE THE REMAINING ROWS:
            for (row = M - M % 4; row < M; row++) {
                double sum = 0;
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    sum = sum + AS[j] * vector[JA[j]];
                }
                csr_result[row] = sum;
            }
            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=unroll-4, FORMAT=CSR, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n", bestTimeAfter10Trials,
               omp_get_max_threads(), gflops);


        // METHOD 5: UNROLLING 8:
        bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            csr_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, N, nz, IRP, JA, AS, vector, csr_result) private(row, col)
            for (row = 0; row < M - M % 8; row += 8) {
                double sum1 = 0;
                double sum2 = 0;
                double sum3 = 0;
                double sum4 = 0;
                double sum5 = 0;
                double sum6 = 0;
                double sum7 = 0;
                double sum8 = 0;
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    sum1 = sum1 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 1]; j < IRP[row + 2]; j++) {
                    sum2 = sum2 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 2]; j < IRP[row + 3]; j++) {
                    sum3 = sum3 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 3]; j < IRP[row + 4]; j++) {
                    sum4 = sum4 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 4]; j < IRP[row + 5]; j++) {
                    sum5 = sum5 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 5]; j < IRP[row + 6]; j++) {
                    sum6 = sum6 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 6]; j < IRP[row + 7]; j++) {
                    sum7 = sum7 + AS[j] * vector[JA[j]];
                }
                for (int j = IRP[row + 7]; j < IRP[row + 8]; j++) {
                    sum8 = sum8 + AS[j] * vector[JA[j]];
                }
                csr_result[row] = sum1;
                csr_result[row + 1] = sum2;
                csr_result[row + 2] = sum3;
                csr_result[row + 3] = sum4;
                csr_result[row + 4] = sum5;
                csr_result[row + 5] = sum6;
                csr_result[row + 6] = sum7;
                csr_result[row + 7] = sum8;
            }
            // HANDLE THE REMAINING ROWS:
            for (row = M - M % 8; row < M; row++) {
                double sum = 0;
                for (int j = IRP[row]; j < IRP[row + 1]; j++) {
                    sum = sum + AS[j] * vector[JA[j]];
                }
                csr_result[row] = sum;
            }
            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=unroll-8, FORMAT=CSR, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n", bestTimeAfter10Trials,
               omp_get_max_threads(), gflops);
    }

    // CONVERT THE MATRIX TO ELLPACK FORMAT:
    if (optype == "ellpack" || optype == "all") {
        int max_row_length = 0;
        int *max_row_lengths = (int *) malloc(M * sizeof(int));
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
        }
        auto **AS = (double **) malloc(M * sizeof(double *));
        for (int i = 0; i < M; i++) {
            AS[i] = (double *) malloc(max_row_length * sizeof(double));
        }
        int *row_fill = (int *) malloc(M * sizeof(int));
        int row, col;
        for (int i = 0; i < nz; i++) {
            row = I[i];
            col = J[i];
            JA[row][row_fill[row]] = col;
            AS[row][row_fill[row]] = val[i];
            row_fill[row]++;
        }

        // COMPUTATION:
        // METHOD 0: SERIAL:
        auto *ellpack_result = (double *) malloc(M * sizeof(double));
        double start = omp_get_wtime();
        for (row = 0; row < M; row++) {
            double sum = 0;
            for (int j = 0; j < max_row_length; j++) {
                sum = sum + AS[row][j] * vector[JA[row][j]];
            }
            ellpack_result[row] = sum;
        }
        double end = omp_get_wtime();
        double serialTime = end - start;
        double gflops = 2.0 * nz / (serialTime * 1e9);
        printf("RESULTS = { METHOD=serial, FORMAT=ELLPACK, TIME=%fs, GFLOPS=%f }\n", serialTime, gflops);

        // METHOD 1: PARALLEL WITHOUT ANY OPTIMIZATION:
        double bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            ellpack_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, max_row_length, max_row_lengths, JA, AS, vector, ellpack_result)
            for (row = 0; row < M; row++) {
                double sum = 0;
                for (int j = 0; j < max_row_lengths[row]; j++) {
                    sum = sum + AS[row][j] * vector[JA[row][j]];
                }
                ellpack_result[row] = sum;
            }
            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=parallel, FORMAT=ELLPACK, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n", bestTimeAfter10Trials,
               omp_get_max_threads(), gflops);



        // METHOD 2: CHUNKS:
        std::vector<int> chunk_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        int bestChunkSize = 0;
        double bestTime = 1000000;
        for (int chunk_size: chunk_sizes) {
            bestTimeAfter10Trials = 1000000;
            for (int i = 0; i < 10; i++) {
                // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
                ellpack_result = (double *) malloc(M * sizeof(double));
                row = 0;
                col = 0;
                // PARALLEL TIMER START:
                start = omp_get_wtime();
#pragma omp parallel for schedule(static, chunk_size) default(none) shared(M, N, nz, max_row_length, max_row_lengths, JA, AS, vector, ellpack_result, chunk_size) private(row, col)
                for (row = 0; row < M; row++) {
                    double sum = 0;
                    for (int j = 0; j < max_row_lengths[row]; j++) {
                        sum = sum + AS[row][j] * vector[JA[row][j]];
                    }
                    ellpack_result[row] = sum;
                }

                // STOP TIMER(S):
                end = omp_get_wtime();
                // TAKE THE BEST TIME AFTER 10 TRIALS:
                if (end - start < bestTimeAfter10Trials) {
                    bestTimeAfter10Trials = end - start;
                }
            }
            if (bestTimeAfter10Trials < bestTime) {
                bestTime = bestTimeAfter10Trials;
                bestChunkSize = chunk_size;
            }
        }
        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTime * 1e9);
        printf("RESULTS = { METHOD=chunks, FORMAT=ELLPACK, BEST_TIME(10)=%fs, THREADS=%d, CHUNK_SIZE(BEST)=%d, GFLOPS=%f }\n",
               bestTime, omp_get_max_threads(), bestChunkSize, gflops);


        // METHOD 3: UNROLL-2:
        bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            ellpack_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, N, nz, max_row_length, max_row_lengths, JA, AS, vector, ellpack_result) private(row, col)
            for (row = 0; row < M - M % 2; row += 2) {
                double sum1 = 0;
                double sum2 = 0;
                for (int j = 0; j < max_row_lengths[row]; j++) {
                    sum1 = sum1 + AS[row][j] * vector[JA[row][j]];
                    sum2 = sum2 + AS[row + 1][j] * vector[JA[row + 1][j]];
                }
                ellpack_result[row] = sum1;
                ellpack_result[row + 1] = sum2;
            }
            for (row = M - M % 2; row < M; row++) {
                double sum = 0;
                for (int j = 0; j < max_row_lengths[row]; j++) {
                    sum = sum + AS[row][j] * vector[JA[row][j]];
                }
                ellpack_result[row] = sum;
            }

            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=unroll-2, FORMAT=ELLPACK, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n",
               bestTimeAfter10Trials, omp_get_max_threads(), gflops);


        // METHOD 4: UNROLL-4:
        bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            ellpack_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, N, nz, max_row_length, max_row_lengths, JA, AS, vector, ellpack_result) private(row, col)
            for (row = 0; row < M - M % 4; row += 4) {
                double sum1 = 0;
                double sum2 = 0;
                double sum3 = 0;
                double sum4 = 0;
                for (int j = 0; j < max_row_lengths[row]; j++) {
                    sum1 = sum1 + AS[row][j] * vector[JA[row][j]];
                    sum2 = sum2 + AS[row + 1][j] * vector[JA[row + 1][j]];
                    sum3 = sum3 + AS[row + 2][j] * vector[JA[row + 2][j]];
                    sum4 = sum4 + AS[row + 3][j] * vector[JA[row + 3][j]];
                }
                ellpack_result[row] = sum1;
                ellpack_result[row + 1] = sum2;
                ellpack_result[row + 2] = sum3;
                ellpack_result[row + 3] = sum4;
            }
            for (row = M - M % 4; row < M; row++) {
                double sum = 0;
                for (int j = 0; j < max_row_lengths[row]; j++) {
                    sum = sum + AS[row][j] * vector[JA[row][j]];
                }
                ellpack_result[row] = sum;
            }

            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=unroll-4, FORMAT=ELLPACK, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n",
               bestTimeAfter10Trials, omp_get_max_threads(), gflops);


        // METHOD 4: UNROLL-8:
        bestTimeAfter10Trials = 1000000;
        for (int i = 0; i < 10; i++) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            ellpack_result = (double *) malloc(M * sizeof(double));
            row = 0;
            col = 0;
            // PARALLEL TIMER START:
            start = omp_get_wtime();
#pragma omp parallel for default(none) shared(M, N, nz, max_row_length, max_row_lengths, JA, AS, vector, ellpack_result) private(row, col)
            for (row = 0; row < M - M % 8; row += 8) {
                double sum1 = 0;
                double sum2 = 0;
                double sum3 = 0;
                double sum4 = 0;
                double sum5 = 0;
                double sum6 = 0;
                double sum7 = 0;
                double sum8 = 0;
                for (int j = 0; j < max_row_lengths[row]; j++) {
                    sum1 = sum1 + AS[row][j] * vector[JA[row][j]];
                    sum2 = sum2 + AS[row + 1][j] * vector[JA[row + 1][j]];
                    sum3 = sum3 + AS[row + 2][j] * vector[JA[row + 2][j]];
                    sum4 = sum4 + AS[row + 3][j] * vector[JA[row + 3][j]];
                    sum5 = sum5 + AS[row + 4][j] * vector[JA[row + 4][j]];
                    sum6 = sum6 + AS[row + 5][j] * vector[JA[row + 5][j]];
                    sum7 = sum7 + AS[row + 6][j] * vector[JA[row + 6][j]];
                    sum8 = sum8 + AS[row + 7][j] * vector[JA[row + 7][j]];
                }
                ellpack_result[row] = sum1;
                ellpack_result[row + 1] = sum2;
                ellpack_result[row + 2] = sum3;
                ellpack_result[row + 3] = sum4;
                ellpack_result[row + 4] = sum5;
                ellpack_result[row + 5] = sum6;
                ellpack_result[row + 6] = sum7;
                ellpack_result[row + 7] = sum8;
            }
            for (row = M - M % 8; row < M; row++) {
                double sum = 0;
                for (int j = 0; j < max_row_lengths[row]; j++) {
                    sum = sum + AS[row][j] * vector[JA[row][j]];
                }
                ellpack_result[row] = sum;
            }

            // STOP TIMER(S):
            end = omp_get_wtime();
            // TAKE THE BEST TIME AFTER 10 TRIALS:
            if (end - start < bestTimeAfter10Trials) {
                bestTimeAfter10Trials = end - start;
            }
        }

        // MEASUREMENTS:
        gflops = 2.0 * nz / (bestTimeAfter10Trials * 1e9);
        printf("RESULTS = { METHOD=unroll-8, FORMAT=ELLPACK, BEST_TIME(10)=%fs, THREADS=%d, GFLOPS=%f }\n",
               bestTimeAfter10Trials, omp_get_max_threads(), gflops);

    }
    return 0;
}