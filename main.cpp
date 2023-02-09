#include <omp.h>
#include <random>

#include "src/mmio.h"

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

// COMPILE AND SUBMIT: cd /scratch/s388885/sspp/sspp-assignment && ./scripts/omp-compile.sh && cd jobs && qsub ../queues/omp.sub && clear && qstat -a && ls
// RUN FRONTEND: ./a.out "src/data/input/Cube_Coup_dt0/Cube_Coup_dt0.mtx" "CSR"

int main(int argc, char *argv[]) {
    //// DEBUG PURPOSES:
    char* path_cage4 = "../src/data/input/cage4/cage4.mtx";
    char* path_cubecoup = "../src/data/input/Cube_Coup_dt0/Cube_Coup_dt0.mtx";
    argc+=2;                    // TEMP
    argv[1] = path_cubecoup;       // TEMP
    argv[2] = "CSR";            // TEMP
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
    if ((optype == "CSR") || (optype == "csr") || (optype == "ELLPACK") || (optype == "ellpack")) {
        if (optype == "CSR") {
            optype = "csr";
        } else if (optype == "ELLPACK") {
            optype = "ellpack";
        }
    } else {
        printf("ERROR: Wrong format. Please choose between CSR and ELLPACK.");
        exit(1);
    }

    // SETTINGS (ANNOUNCEMENT):
    std::string matrix_name = argv[1];
    matrix_name = matrix_name.substr(matrix_name.find_last_of("/\\") + 1);
    printf("SETTINGS = { MATRIX=\"%s\", OP=MAT/VEC, FORMAT=%s, SIZE=%dx%d, THREADS=%d }\n", matrix_name.c_str(), argv[2], M, N, omp_get_max_threads());

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
    if (optype == "csr") {
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

        // COMPUTATION:
        std::vector<int> chunk_sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
        for (int chunk_size : chunk_sizes) {
            // ALLOCATE MEMORY FOR THE RESULT, ROWS AND COLS:
            auto *csr_result = (double *) malloc(M * sizeof(double));
            int row, col;
            // PARALLEL TIMER START:
            double start = omp_get_wtime();
#pragma omp parallel for schedule(static, chunk_size) default(none) shared(M, N, nz, IRP, JA, AS, vector, csr_result, chunk_size) private(row, col)
            for (int i = 0; i < M; i++) {
                double sum = 0;
                for (int j = IRP[i-1]; j < IRP[i]; j++) {
                    sum = sum + AS[j] * vector[JA[j]];
                }
                csr_result[i] = sum;
            }
            // STOP TIMER(S):
            double end = omp_get_wtime();

            // MEASUREMENTS:
            double gflops = 2.0 * nz / ((end - start) * 1e9);
            printf("RESULTS = { TIME=%fs, THREADS=%d, CHUNK_SIZE=%d, GFLOPS=%f }\n", end - start, omp_get_max_threads(), chunk_size, gflops);
        }
    return 0;
    }
}