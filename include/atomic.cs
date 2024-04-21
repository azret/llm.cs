using System;
using System.Threading;
using System.Threading.Tasks;

using static std;

public static unsafe class atomic {

    public static unsafe void atomicAdd(float* dest, float val) {
        int* address = (int*)dest;
        int old = *address, assumed;
        do {
            assumed = old;
            float desired = (val + *((float*)&assumed));
            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
            old = Interlocked.CompareExchange(
                ref address[0],
                *((int*)&desired),
                assumed);
        } while (assumed != old);
    }

#if atomic
    static int Main() {
        ulong seed = 37;

        int B = 8;
        int T = 64;
        int C = 1024;

        float* h_REF = (float*)malloc(sizeof(float));
        float* d_ = (float*)malloc(sizeof(float));

        for (int i = 0; i < B * T * C; i++) {
            *h_REF += (float)(Math.PI * Math.E);
        }

        Parallel.For(0, B * T * C, (i) => {
            atomicAdd(d_, (float)Math.PI);
        });

        validate_result(d_, h_REF, "atomicAdd", 1, 1e-5f);

        free(d_);
        free(h_REF);

        // float* h_REF_In = malloc_random_float(&seed, B * T * C);
        // float* h_REF_Weight = malloc_random_float(&seed, C);
        // float* h_REF_Mean = malloc_random_float(&seed, B * T);
        // float* h_REF_Rstd = malloc_random_float(&seed, B * T);
        // 
        // float* h_REF_DIn = malloc_zero_float(&seed, B * T * C);
        // float* h_REF_DWeight = malloc_zero_float(&seed, C);
        // float* h_REF_DBias = malloc_zero_float(&seed, C);
        // 
        // float* CUDA_DOut = malloc_random_float(&seed, B * T * C);
        // float* CUDA_In = malloc_random_float(&seed, B * T * C);
        // float* CUDA_Weight = malloc_random_float(&seed, C);
        // float* CUDA_Mean = malloc_random_float(&seed, B * T);
        // float* CUDA_Rstd = malloc_random_float(&seed, B * T);
        // 
        // float* CUDA_DIn = malloc_zero_float(&seed, B * T * C);
        // float* CUDA_DWeight = malloc_zero_float(&seed, C);
        // float* CUDA_DBias = malloc_zero_float(&seed, C);
        // 
        // memcpy(CUDA_DOut, h_REF_DOut, B * T * C * sizeof(float));
        // memcpy(CUDA_In, h_REF_In, B * T * C * sizeof(float));
        // memcpy(CUDA_Weight, h_REF_Weight, C * sizeof(float));
        // memcpy(CUDA_Mean, h_REF_Mean, B * T * sizeof(float));
        // memcpy(CUDA_Rstd, h_REF_Rstd, B * T * sizeof(float));
        // 
        // for (int n = 0; n < 256; n++) {
        //     layernorm_backward_0(
        //         h_REF_DIn, h_REF_DWeight, h_REF_DBias,
        //         h_REF_DOut, h_REF_In, h_REF_Weight, h_REF_Mean, h_REF_Rstd,
        //         B, T, C);
        // }
        // 
        // for (int n = 0; n < 256; n++) {
        //     layernorm_backward_1(
        //         CUDA_DIn, CUDA_DWeight, CUDA_DBias,
        //         CUDA_DOut, CUDA_In, CUDA_Weight, CUDA_Mean, CUDA_Rstd,
        //         B, T, C);
        // }
        // 
        // validate_result(CUDA_DOut, h_REF_DOut, "CUDA_DOut", B * T * C, 1e-5f);
        // validate_result(CUDA_In, h_REF_In, "CUDA_In", B * T * C, 1e-5f);
        // validate_result(CUDA_Weight, h_REF_Weight, "CUDA_Weight", C, 1e-5f);
        // validate_result(CUDA_Mean, h_REF_Mean, "CUDA_Mean", B * T, 1e-5f);
        // validate_result(CUDA_Rstd, h_REF_Rstd, "CUDA_Rstd", B * T, 1e-5f);
        // 
        // validate_result(CUDA_DIn, h_REF_DIn, "CUDA_DIn", B * T * C, 1e-4f);
        // validate_result(CUDA_DWeight, h_REF_DWeight, "CUDA_DWeight", C, 1e-4f);
        // validate_result(CUDA_DBias, h_REF_DBias, "CUDA_DBias", C, 1e-4f);

        Console.ReadKey();

        return 0;
    }

#endif
}