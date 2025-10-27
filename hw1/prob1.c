#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

#define INF 1e15

void init_graph(double *A, int N) {
    srand(42);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            long long idx = (long long)i * N + j;
            if (i == j) {
                A[idx] = 0.0;
            } else if (rand() % 100 < 20) {  
                A[idx] = 1.0 + rand() % 10;  
            } else {
                A[idx] = INF;
            }
        }
    }
}

void floyd_serial(double *A, int N) {
    for (int k = 0; k < N; k++) {
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                long long idx_ij = (long long)i * N + j;
                long long idx_ik = (long long)i * N + k;
                long long idx_kj = (long long)k * N + j;
                if (A[idx_ik] + A[idx_kj] < A[idx_ij]) {
                    A[idx_ij] = A[idx_ik] + A[idx_kj];
                }
            }
        }
    }
}

void floyd_parallel(double *A, int N) {
    for (int k = 0; k < N; k++) {
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                long long idx_ij = (long long)i * N + j;
                long long idx_ik = (long long)i * N + k;
                long long idx_kj = (long long)k * N + j;
                if (A[idx_ik] + A[idx_kj] < A[idx_ij]) {
                    A[idx_ij] = A[idx_ik] + A[idx_kj];
                }
            }
        }
    }
}

double check_result(double *A1, double *A2, int N) {
    double max_diff = 0.0;
    for (long long i = 0; i < (long long)N * N; i++) {
        double diff = fabs(A1[i] - A2[i]);
        if (diff > max_diff) {
            max_diff = diff;
        }
    }
    return max_diff;
}

int main(int argc, char **argv) {
    int N = (argc >= 2) ? atoi(argv[1]) : 512;
    int threads = (argc >= 3) ? atoi(argv[2]) : 4;
    if (threads > 0) {
        omp_set_num_threads(threads);
    }
    
    long long size = (long long)N * N;
    double *A0 = malloc(sizeof(double) * size);
    double *A1 = malloc(sizeof(double) * size);
    double *A2 = malloc(sizeof(double) * size);
    
    init_graph(A0, N);
    memcpy(A1, A0, sizeof(double) * size);
    memcpy(A2, A0, sizeof(double) * size);
    
    double t1 = omp_get_wtime();
    floyd_serial(A1, N);
    double t2 = omp_get_wtime();
    
    double t3 = omp_get_wtime();
    floyd_parallel(A2, N);
    double t4 = omp_get_wtime();
    
    double max_diff = check_result(A1, A2, N);
    
    printf("[prob1-FloydWarshall] N=%d, threads=%d\n", N, threads);
    printf("Serial Time:     %.6f s\n", t2 - t1);
    printf("Parallel Time:   %.6f s\n", t4 - t3);
    printf("Speedup:         %.2f×\n", (t2 - t1) / ((t4 - t3) > 0 ? (t4 - t3) : 1e-9));
    printf("Max Abs Diff:    %.3e\n", max_diff);
    
    free(A0);
    free(A1);
    free(A2);
    return 0;
}
