// prob3.c — Problem 3: Parallel 2D Convolution with OpenMP Reduction
//
// In convolutional neural networks, the 2D convolution operation is a key computation.
// Given a 2D input matrix (size: MxN) and a convolution filter (size: KxK),
// the output matrix is obtained by performing a dot product between the filter
// and a sliding window over the input.
//
// This implementation uses OpenMP to parallelize the convolution process with
// reduction mechanisms to accumulate results during the dot product operation.
// Uses "valid" mode (no padding), so output size is (M-K+1) x (N-K+1).
//
// Usage:
//   ./prob3 M N K seed [threads]
// Example:
//   ./prob3 1024 1024 5 42 8

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

static void init_data(float *in, int M, int N, float *ker, int K, unsigned int seed){
    unsigned short xi[3] = { (unsigned short)seed, (unsigned short)(seed>>8), (unsigned short)(seed>>16) };
    #pragma omp parallel
    {
        unsigned short lxi[3] = { (unsigned short)(xi[0]^omp_get_thread_num()*123u),
                                  (unsigned short)(xi[1]^omp_get_thread_num()*321u),
                                  (unsigned short)(xi[2]^omp_get_thread_num()*777u) };
        #pragma omp for schedule(static)
        for (int i=0;i<M*N;i++){
            in[i] = (float)(erand48(lxi)*2.0 - 1.0); 
        }
    }
        unsigned short ker_xi[3] = { 
        (unsigned short)(seed ^ 0xABCDu),
        (unsigned short)((seed>>8) ^ 0x1234u),
        (unsigned short)((seed>>16) ^ 0x5678u)
    };
    for (int i=0; i<K*K; i++) {
        ker[i] = (float)(erand48(ker_xi)*2.0 - 1.0); 
    }
}

int main(int argc, char **argv){
    int M = (argc>=2)? atoi(argv[1]) : 512;
    int N = (argc>=3)? atoi(argv[2]) : 512;
    int K = (argc>=4)? atoi(argv[3]) : 3;
    unsigned int seed = (argc>=5)? (unsigned)atoi(argv[4]) : 42u;
    int threads = (argc>=6)? atoi(argv[5]) : 0;
    if (threads>0) omp_set_num_threads(threads);
    if (M<=0 || N<=0 || K<=0 || K>M || K>N){
        fprintf(stderr,"Invalid sizes. Need M,N>=1 and 1<=K<=min(M,N)\n"); return 1;
    }
    int OH = M-K+1, OW = N-K+1;

    float *input  = (float*)malloc(sizeof(float)*M*N);
    float *kernel = (float*)malloc(sizeof(float)*K*K);
    float *out_s  = (float*)malloc(sizeof(float)*OH*OW);
    float *out_p  = (float*)malloc(sizeof(float)*OH*OW);
    if(!input||!kernel||!out_s||!out_p){ fprintf(stderr,"alloc failed\n"); return 2; }

    init_data(input,M,N,kernel,K,seed);

    // Serial
    double t0 = omp_get_wtime();
    for (int i=0;i<OH;i++){
        for (int j=0;j<OW;j++){
            float sum = 0.0f;
            for (int u=0;u<K;u++){
                for (int v=0;v<K;v++){
                    sum += input[(i+u)*N + (j+v)] * kernel[u*K + v];
                }
            }
            out_s[i*OW + j] = sum;
        }
    }
    double t1 = omp_get_wtime();

    // Parallel
    double t2 = omp_get_wtime();
    
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < OH; i++) {
        for (int j = 0; j < OW; j++) {
            float sum = 0.0f;
            #pragma omp simd reduction(+:sum)
            for (int u = 0; u < K; u++) {
                for (int v = 0; v < K; v++) {
                    sum += input[(i+u)*N + (j+v)] * kernel[u*K + v];
                }
            }
            out_p[i*OW + j] = sum;
        }
    }
    
    double t3 = omp_get_wtime();

    double maxd = 0.0;
    for (long long idx=0; idx<(long long)OH*OW; ++idx){
        double d = (double)out_s[idx] - (double)out_p[idx];
        if (d<0) d=-d;
        if (d>maxd) maxd=d;
    }

    int used_threads=1;
    #pragma omp parallel
    {
        #pragma omp master
        used_threads=omp_get_num_threads();
    }

    printf("[prob3-Conv2D] M=%d, N=%d, K=%d, threads=%d\n", M, N, K, used_threads);
    printf("Input Matrix:    %d x %d\n", M, N);
    printf("Filter Size:     %d x %d\n", K, K);
    printf("Output Matrix:   %d x %d (valid mode)\n", OH, OW);
    printf("Serial Time:     %.6f s\n", t1-t0);
    printf("Parallel Time:   %.6f s\n", t3-t2);
    printf("Speedup:         %.2f×\n", (t1-t0)/((t3-t2)>0?(t3-t2):1e-9));
    printf("Max Abs Diff:    %.3e\n", maxd);

    free(input); free(kernel); free(out_s); free(out_p);
    return (maxd<1e-6)? 0:4;
}
