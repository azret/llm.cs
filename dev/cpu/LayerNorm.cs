using System;
using System.Security.Cryptography;
using System.Security.Policy;
using System.Threading;
using System.Threading.Tasks;

using static Common;
using static kernel32;
using static MathF;

public static unsafe class LayerNorm {
    public static unsafe void layernorm_forward_0(float* _Out,
                                                float* _Mean,
                                                float* _Rstd,
                                                float* _In,
                                                float* _Weight,
                                                float* _Bias,
                                                int B,
                                                int T,
                                                int C) {

        float eps = 1e-5f;
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                float* x = _In + b * T * C + t * C;
                // calculate the mean
                float m = 0.0f;
                for (int i = 0; i < C; i++) {
                    m += x[i];
                }
                m = m / C;
                // calculate the variance (without any bias correction)
                float v = 0.0f;
                for (int i = 0; i < C; i++) {
                    float xshift = x[i] - m;
                    v += xshift * xshift;
                }
                v = v / C;
                // calculate the rstd
                float s = 1.0f / sqrtf(v + eps);
                // seek to the output position in out[b,t,:]
                float* y = _Out + b * T * C + t * C;
                for (int i = 0; i < C; i++) {
                    float n = (s * (x[i] - m)); // normalized output
                    float o = n * _Weight[i] + _Bias[i]; // scale and shift it
                    y[i] = o; // write
                }
                // cache the mean and rstd for the backward pass later
                _Mean[b * T + t] = m;
                _Rstd[b * T + t] = s;
            }
        }
    }


    private static void layernorm_backward_kernel_0(int b,
                                                  int t,
                                                  float* dinp,
                                                  float* dweight,
                                                  float* dbias,
                                                  float* dout,
                                                  float* inp,
                                                  float* weight,
                                                  float* mean,
                                                  float* rstd,
                                                  int T,
                                                  int C) {

        float* dout_bt = dout + b * T * C + t * C;
        float* inp_bt = inp + b * T * C + t * C;
        float* dinp_bt = dinp + b * T * C + t * C;
        float mean_bt = mean[b * T + t];
        float rstd_bt = rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];
            // gradient contribution to bias
            dbias[i] += dout_bt[i];
            // gradient contribution to weight
            dweight[i] += norm_bti * dout_bt[i];
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            dinp_bt[i] += dval;
        }
    }

    public static unsafe void layernorm_backward_0(float* dinp, float* dweight, float* dbias,
                            float* dout, float* inp, float* weight, float* mean, float* rstd,
                            int B, int T, int C) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                layernorm_backward_kernel_0(b,
                                          t,
                                          dinp,
                                          dweight,
                                          dbias,
                                          dout,
                                          inp,
                                          weight,
                                          mean,
                                          rstd,
                                          T,
                                          C);
            }
        }
    }

    public static unsafe void layernorm_backward(float* dinp, float* dweight, float* dbias,
                            float* dout, float* inp, float* weight, float* mean, float* rstd,
                            int B, int T, int C) {

        Parallel.For(0, B * T, (bt) => {
            int b = bt / T;
            int t = bt % T;

            layernorm_backward_kernel_1(b,
                                      t,
                                      dinp,
                                      dweight,
                                      dbias,
                                      dout,
                                      inp,
                                      weight,
                                      mean,
                                      rstd,
                                      T,
                                      C);
        });
    }

    private static void layernorm_backward_kernel_1(int b,
                                              int t,
                                              float* dinp,
                                              float* dweight,
                                              float* dbias,
                                              float* dout,
                                              float* inp,
                                              float* weight,
                                              float* mean,
                                              float* rstd,
                                              int T,
                                              int C) {

        float* dout_bt = dout + b * T * C + t * C;
        float* inp_bt = inp + b * T * C + t * C;
        float* dinp_bt = dinp + b * T * C + t * C;
        float mean_bt = mean[b * T + t];
        float rstd_bt = rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = 0; i < C; i++) {
            float norm_bti = (inp_bt[i] - mean_bt) * rstd_bt;
            float dnorm_i = weight[i] * dout_bt[i];
            // gradient contribution to bias
            // dbias[i] += dout_bt[i];
            atomicAdd(&dbias[i], dout_bt[i]);
            // gradient contribution to weight
            // dweight[i] += norm_bti * dout_bt[i];
            atomicAdd(&dweight[i], norm_bti * dout_bt[i]);
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= rstd_bt; // final scale
            // dinp_bt[i] += dval;
            atomicAdd(&dinp_bt[i], dval);
        }
    }

    private static void layernorm_forward_cpu_kernel(int b,
                                                     int t,
                                                     float* _Out,
                                                     float* _Mean,
                                                     float* _Rstd,
                                                     float* _In,
                                                     float* _Weight,
                                                     float* _Bias,
                                                     int T,
                                                     int C) {
        const float eps = 1e-5f;

        float* x = _In + b * T * C + t * C;
        // calculate the mean
        float m = 0.0f;
        for (int i = 0; i < C; i++) {
            m += x[i];
        }
        m = m / C;
        // calculate the variance (without any bias correction)
        float v = 0.0f;
        for (int i = 0; i < C; i++) {
            float xshift = x[i] - m;
            v += xshift * xshift;
        }
        v = v / C;
        // calculate the rstd
        float s = 1.0f / sqrtf(v + eps);
        // seek to the output position in out[b,t,:]
        float* out_bt = _Out + b * T * C + t * C;
        for (int i = 0; i < C; i++) {
            float n = (s * (x[i] - m)); // normalized output
            float o = n * _Weight[i] + _Bias[i]; // scale and shift it
            out_bt[i] = o; // write
        }
        // cache the mean and rstd for the backward pass later
        _Mean[b * T + t] = m;
        _Rstd[b * T + t] = s;

        // atomicAdd(ref _Mean[b * T + t], 2);
    }

    public static unsafe void layernorm_forward_cpu_0(float* _Out,
                                                float* _Mean,
                                                float* _Rstd,
                                                float* _In,
                                                float* _Weight,
                                                float* _Bias,
                                                int B,
                                                int T,
                                                int C) {

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                layernorm_forward_cpu_kernel(b,
                                             t,
                                             _Out,
                                             _Mean,
                                             _Rstd,
                                             _In,
                                             _Weight,
                                             _Bias,
                                             T,
                                             C);
            }
        }
    }

    public static unsafe void layernorm_forward(float* _Out,
                                                float* _Mean,
                                                float* _Rstd,
                                                float* _In,
                                                float* _Weight,
                                                float* _Bias,
                                                int B,
                                                int T,
                                                int C) {

        Parallel.For(0, B * T, (bt) => {
            int b = bt / T;
            int t = bt % T;

            layernorm_forward_cpu_kernel(b,
                             t,
                             _Out,
                             _Mean,
                             _Rstd,
                             _In,
                             _Weight,
                             _Bias,
                             T,
                             C);
        });
    }

    /*
    __device__ double atomicAdd(double* address, double val) {
        unsigned long long int* address_as_ull =
                                  (unsigned long long int*)address;
        unsigned long long int old = *address_as_ull, assumed;

        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed,
                            __double_as_longlong(val +
                                   __longlong_as_double(assumed)));

            // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
        } while (assumed != old);

        return __longlong_as_double(old);
    }


    */

    public static unsafe float atomicAdd(float* dest, float val) {
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
        return *((float*)&old);
    }

#if LayerNorm
    static int Main() {
        ulong seed = 37;

        int B = 8;
        int T = 64;
        int C = 768;

        // backward pass on CPU

        float* h_REF_DOut = malloc_random_float(&seed, B * T * C);
        float* h_REF_In = malloc_random_float(&seed, B * T * C);
        float* h_REF_Weight = malloc_random_float(&seed, C);
        float* h_REF_Mean = malloc_random_float(&seed, B * T);
        float* h_REF_Rstd = malloc_random_float(&seed, B * T);

        float* h_REF_DIn = malloc_zero_float(&seed, B * T * C);
        float* h_REF_DWeight = malloc_zero_float(&seed, C);
        float* h_REF_DBias = malloc_zero_float(&seed, C);

        float* CUDA_DOut = malloc_random_float(&seed, B * T * C);
        float* CUDA_In = malloc_random_float(&seed, B * T * C);
        float* CUDA_Weight = malloc_random_float(&seed, C);
        float* CUDA_Mean = malloc_random_float(&seed, B * T);
        float* CUDA_Rstd = malloc_random_float(&seed, B * T);

        float* CUDA_DIn = malloc_zero_float(&seed, B * T * C);
        float* CUDA_DWeight = malloc_zero_float(&seed, C);
        float* CUDA_DBias = malloc_zero_float(&seed, C);

        memcpy(CUDA_DOut, h_REF_DOut, B * T * C * sizeof(float));
        memcpy(CUDA_In, h_REF_In, B * T * C * sizeof(float));
        memcpy(CUDA_Weight, h_REF_Weight, C * sizeof(float));
        memcpy(CUDA_Mean, h_REF_Mean, B * T * sizeof(float));
        memcpy(CUDA_Rstd, h_REF_Rstd, B * T * sizeof(float));

        // memcpy(CUDA_DIn, h_REF_DIn, B * T * C * sizeof(float));
        // memcpy(CUDA_DWeight, h_REF_DWeight, C * sizeof(float));
        // memcpy(CUDA_DBias, h_REF_DBias, C * sizeof(float));

        layernorm_backward_0(
            h_REF_DIn, h_REF_DWeight, h_REF_DBias,
            h_REF_DOut, h_REF_In, h_REF_Weight, h_REF_Mean, h_REF_Rstd,
            B, T, C);

        layernorm_backward_0(
            CUDA_DIn, CUDA_DWeight, CUDA_DBias,
            CUDA_DOut, CUDA_In, CUDA_Weight, CUDA_Mean, CUDA_Rstd,
            B, T, C);

        validate_result(CUDA_DOut, h_REF_DOut, "CUDA_DOut", B * T * C, 1e-5f);
        validate_result(CUDA_In, h_REF_In, "CUDA_In", B * T * C, 1e-5f);
        validate_result(CUDA_Weight, h_REF_Weight, "CUDA_Weight", C, 1e-5f);
        validate_result(CUDA_Mean, h_REF_Mean, "CUDA_Mean", B * T, 1e-5f);
        validate_result(CUDA_Rstd, h_REF_Rstd, "CUDA_Rstd", B * T, 1e-5f);

        validate_result(CUDA_DIn, h_REF_DIn, "CUDA_DIn", B * T * C, 1e-5f);
        validate_result(CUDA_DWeight, h_REF_DWeight, "CUDA_DWeight", C, 1e-5f);
        validate_result(CUDA_DBias, h_REF_DBias, "CUDA_DBias", C, 1e-5f);

        Console.ReadKey();
        return 0;
    }
    
#endif
}

