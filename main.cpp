#include <cstdio>
#include <cstdlib>
#include <omp.h>
#include <iostream>
#include <string>

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
void print_matrix_using_CSR(int M, int N, int nz, int *IRP, int *JA, double *AS) {
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

// Execution line: ./a.out /scratch/s388885/sspp/sspp-assignment/src/data/input/Cube_Coup_dt0/Cube_Coup_dt0.mtx CSR
int main(int argc, char *argv[]) {
    //// DEBUG PURPOSES:
    char* path_cage4 = "../src/data/input/cage4/cage4.mtx";
    char* path_cubecoup = "../src/data/input/Cube_Coup_dt0/Cube_Coup_dt0.mtx";
    argc+=2;                    // TEMP
    argv[1] = path_cage4;       // TEMP
    argv[2] = "CSR";            // TEMP
    //// END OF DEBUG PURPOSES.

    // SETTINGS:
    std::string optype = argv[2];
    if ((optype == "CSR") || (optype == "csr") || (optype == "ELLPACK") || (optype == "ellpack")) {
        if (optype == "CSR") {
            optype = "csr";
        } else if (optype == "ELLPACK") {
            optype = "ellpack";
        }
    } else {
        std::cout << "ERROR: Wrong format. Please choose between CSR and ELLPACK." << std::endl;
        exit(1);
    }

    // MATRIX SETTINGS:
    int ret_code;
    MM_typecode matcode;
    FILE *f;
    int M, N, nz;
    int *I, *J;
    double *val;

    // READ MATRIX FROM FILE:
    if (argc < 2) { fprintf(stderr, "ERROR: Please specify a valid .mtx path.\n"); exit(1); }
    else if ((f = fopen(argv[1], "r")) == NULL) exit(1);
    if (mm_read_banner(f, &matcode) != 0) { printf("ERROR: Could not process Matrix Market banner.\n"); exit(1); }
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode)) { printf("ERROR: Sorry, this application does not support "); printf("Matrix Market type: [%s]\n", mm_typecode_to_str(matcode)); exit(1); }
    if ((ret_code = mm_read_mtx_crd_size(f, &M, &N, &nz)) !=0) exit(1);
    std::string matrix_name = argv[1];
    matrix_name = matrix_name.substr(matrix_name.find_last_of("/\\") + 1);
    printf("SETTINGS = { MATRIX=\"%s\"  ,   FORMAT=%s   ,   SIZE=%dx%d  ,   THREADS=%d }\n", matrix_name.c_str(), optype.c_str(), M, N, omp_get_max_threads());

    // CONSTRUCT MATRIX:
    printf("Started matrix construction.\n");
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
    printf("Finished matrix construction.\n");

    // CONSTRUCT RANDOM VECTOR:
    printf("Started vector construction.\n");
    int *vector = (int *) malloc(M * sizeof(int));
    for (int i = 0; i < M; i++) {
        vector[i] = rand() % 10;
    }
    printf("Finished vector construction.\n");

    // PRINT THE SPARSE MATRIX:
//    print_list(nz, I, J, val);

    // WRITE THE SPARSE MATRIX TO OUTPUT FILE:
//    mm_write_banner(stdout, matcode);
//    mm_write_mtx_crd_size(stdout, M, N, nz);
//    for (int i=0; i<nz; i++) fprintf(stdout, "%d %d %20.19g\n", I[i], J[i], val[i]);


    // CONVERT THE MATRIX TO CSR FORMAT:
    if (optype == "csr") {
        printf("Started CSR conversion.\n");
        int *IRP = (int *) malloc((M+1) * sizeof(int));
        int *JA = (int *) malloc(nz * sizeof(int));
        auto *AS = (double *) malloc(nz * sizeof(double));
        IRP[0] = 0;
        // Construct CSR, given three arrays I, J, val; ordered by J:
        int counter = 0;
        for (int i = 0; i < M; i++) {
            for (int j = 0; j < nz; j++) {
                if (I[j] == i) {
                    JA[counter] = J[j];
                    AS[counter] = val[j];
                    counter++;
                }
            }
            IRP[i+1] = counter;
        }
        printf("Finished CSR conversion.\n");
        // PRINT THE CSR MATRIX:
    //    print_IRP(M, IRP);
    //    print_JA(nz, JA);
    //    print_AS(nz, AS);
    //    print_matrix_using_CSR(M, N, nz, IRP, JA, AS);

        // TIMER START:
        double start = omp_get_wtime();
        printf("Starting computation.\n...\n");

        // COMPUTATION:
        auto *result = (double *) malloc(M * sizeof(double));
        int row, col;
#pragma omp parallel for default(none) shared(M, N, nz, IRP, JA, AS, vector, result) private(row, col)
        for (row = 0; row < M; row++) {
            result[row] = 0;
            for (col = IRP[row]; col < IRP[row+1]; col++) {
                result[row] += AS[col] * vector[JA[col]];
            }
        }

        // STOP TIMER:
        double end = omp_get_wtime();
        double gflops = 2.0 * nz / ((end - start) * 1e9);
        printf("Finished computation.\nComputation took %f seconds, with a speed of %f GFLOPS using %d threads.\n", end - start, gflops, omp_get_max_threads());
    }

    return 0;
}