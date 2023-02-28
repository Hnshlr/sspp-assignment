# Small Scale Parallel Programming Assignment 2022-2023
Development of a sparse matrix-vector product kernel and its parallelization exploiting computing capabilities of OpenMP and CUDA.

© Copyright 2023, All rights reserved to Hans Haller, CSTE-CIDA Student at Cranfield Uni. SATM, Cranfield, UK.

https://www.github.com/Hnshlr

### Compile and run OpenMP version:
To compile the OpenMP version of this code on Crescent, make sure you're in the main directory.
Then run the following command:

``` module load fosscuda/2019b && export CXX=$(which g++) && g++ -fopenmp -O4 -lgomp -std=c++11 main.cpp src/mmio.h src/mmio.c ```

The compilation should be successful (just ignore the warnings).

Then run the executable file with the following command:

``` ./a.out "path/to/the/matrix.mtx" "ALL" ```

If you wish to only run either the CSR or ELLPACK format, replace "ALL" with "CSR" or "ELLPACK" respectively.

To submit a job to Crescent, you can use the omp.sub file. To do so, run the following command:

``` qsub omp.sub ```

The .sub files are located in the queues/ directory. 

### Compile and run CUDA version:
To compile the CUDA version of this code on Crescent, make sure you're in the main directory.

You then need to ``` cd cuda/``` in order to enter the subdirectory allocated specifically for CUDA. Then follow these instructions:

```. env.sh``` in order to load the necessary modules.

```cmake .``` in order to generate the Makefile.

```cd matvec/``` in order to enter the subdirectory containing the source code.

```make cuda``` in order to compile the main cuda.cu file.

Once the code is compiled, you must run it through a Crescent job that will run on the GPU queue.

To submit such a job, use the cuda.sub file in the queues/ subdirectory of the main directory.  To do so, run the following command:

```qsub cuda.sub```

### All-in-one scripts:

If the above instructions are too much of a hassle, you can use the all-in-one scripts in the queues/ directory, made specifically for both OpenMP and CUDA.
These scripts handle the entire process of compiling and submitting the job to Crescent.
To use them, go in the scripts/ sub-directory of the main directory (```cd scripts/```) and run the following command:

```./cuda-full.sh ```

or

```./omp-full.sh ```

IMPORTANT: These scripts will submit a "bulk" job, which will run the code on the 30 matrices provided in the data/input/ directory.

### You're set to go!

If you have any questions, please contact me at ```hans.haller.885@cranfield.ac.uk```.


### Acknowledgements

This code was developed as part of the Small Scale Parallel Programming course at Cranfield University, UK. 