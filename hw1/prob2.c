#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

int main(int argc, char **argv){
    if (argc < 5){
        fprintf(stderr,"Usage: %s samples xi0 xi1 xi2 [threads]\n", argv[0]);
        return 1;
    }

    long long samples = atoll(argv[1]);
    unsigned short base_xi[3];
    base_xi[0] = (unsigned short)atoi(argv[2]);
    base_xi[1] = (unsigned short)atoi(argv[3]);
    base_xi[2] = (unsigned short)atoi(argv[4]);
    int threads = (argc >= 6)? atoi(argv[5]) : 0;
    if (threads > 0) omp_set_num_threads(threads);

    // Serial
    unsigned short xi_serial[3] = { base_xi[0], base_xi[1], base_xi[2] };
    long long in_serial = 0;
    double t0 = omp_get_wtime();
    for (long long i = 0; i < samples; i++) {
        double x = erand48(xi_serial);
        double y = erand48(xi_serial);
        if (x*x + y*y <= 1.0) in_serial++;
    }
    double t1 = omp_get_wtime();
    double pi_serial = 4.0 * ((double)in_serial / samples);

    // Parallel
    long long in_parallel = 0;
    double t2 = omp_get_wtime();
    #pragma omp parallel reduction(+:in_parallel)
    {
        int tid = omp_get_thread_num();
        unsigned short xi[3] = {
            (unsigned short)(base_xi[0] ^ (tid*0x9e37u)),
            (unsigned short)(base_xi[1] ^ (tid*0x79b9u)),
            (unsigned short)(base_xi[2] ^ (tid*0x24dBu))
        };

        long long local_count = 0;
        #pragma omp for schedule(static)
        for (long long i = 0; i < samples; i++) {
            double x = erand48(xi);
            double y = erand48(xi);
            if (x*x + y*y <= 1.0) local_count++;
        }
        in_parallel += local_count;
    }
    double t3 = omp_get_wtime();
    double pi_parallel = 4.0 * ((double)in_parallel / samples);

    int used_threads = 1;
    #pragma omp parallel
    {
        #pragma omp master
        used_threads = omp_get_num_threads();
    }

    const double PI_TRUE = M_PI;  
    
    double error_serial = fabs(pi_serial - PI_TRUE);
    double error_parallel = fabs(pi_parallel - PI_TRUE);
    double relative_error_serial = error_serial / PI_TRUE * 100.0;
    double relative_error_parallel = error_parallel / PI_TRUE * 100.0;
    double diff_serial_parallel = fabs(pi_serial - pi_parallel);

    printf("[prob2-MonteCarloPI] samples=%lld, threads=%d\n", samples, used_threads);
    printf("True PI:         π = %.15f\n", PI_TRUE);
    printf("Serial Result:   π = %.8f\n", pi_serial);
    printf("Serial Time:     %.6f s\n", t1 - t0);
    printf("Serial Error:    %.8f (%.4f%%)\n", error_serial, relative_error_serial);
    printf("Parallel Result: π = %.8f\n", pi_parallel);
    printf("Parallel Time:   %.6f s\n", t3 - t2);
    printf("Parallel Error:  %.8f (%.4f%%)\n", error_parallel, relative_error_parallel);
    printf("Speedup:         %.2f×\n", (t1 - t0) / ((t3 - t2) > 0 ? (t3 - t2) : 1e-9));
    printf("|Serial - Parallel|: %.3e\n", diff_serial_parallel);

    return 0;
}
