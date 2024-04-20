using System;
using System.Threading;
using System.Threading.Tasks;

using static stdio;
using static math;
using static atomic;

using static Common;
using System.Diagnostics;

public static unsafe class layernorm_backward {

    //--------------------------------------------------------------

    private static void layernorm_backward_kernel_0(int b,
                                                  int t,
                                                  float* dIn,
                                                  float* dWeight,
                                                  float* dBias,
                                                  float* dOut,
                                                  float* _In,
                                                  float* _Weight,
                                                  float* _Mean,
                                                  float* _Rstd,
                                                  int B,
                                                  int T,
                                                  int C) {

        float* dOut_bt = dOut + b * T * C + t * C;
        float* _In_bt = _In + b * T * C + t * C;
        float* _dIn_bt = dIn + b * T * C + t * C;
        float _Mean_bt = _Mean[b * T + t];
        float _Rstd_bt = _Rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
            float norm_bti = (_In_bt[i] - _Mean_bt) * _Rstd_bt;
            float dnorm_i = _Weight[i] * dOut_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = 0; i < C; i++) {
            float norm_bti = (_In_bt[i] - _Mean_bt) * _Rstd_bt;
            float dnorm_i = _Weight[i] * dOut_bt[i];
            // gradient contribution to bias
            dBias[i] += dOut_bt[i];
            // gradient contribution to weight
            dWeight[i] += norm_bti * dOut_bt[i];
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= _Rstd_bt; // final scale
            _dIn_bt[i] += dval;
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
                                          B,
                                          T,
                                          C);
            }
        }
    }

    //--------------------------------------------------------------

    private static void layernorm_backward_kernel_1(int b,
                                                  int t,
                                                  float* dIn,
                                                  float* dWeight,
                                                  float* dBias,
                                                  float* dOut,
                                                  float* _In,
                                                  float* _Weight,
                                                  float* _Mean,
                                                  float* _Rstd,
                                                  int B,
                                                  int T,
                                                  int C) {

        float* dOut_bt = dOut + b * T * C + t * C;
        float* _In_bt = _In + b * T * C + t * C;
        float* _dIn_bt = dIn + b * T * C + t * C;
        float _Mean_bt = _Mean[b * T + t];
        float _Rstd_bt = _Rstd[b * T + t];

        // first: two reduce operations
        float dnorm_mean = 0.0f;
        float dnorm_norm_mean = 0.0f;
        for (int i = 0; i < C; i++) {
            float norm_bti = (_In_bt[i] - _Mean_bt) * _Rstd_bt;
            float dnorm_i = _Weight[i] * dOut_bt[i];
            dnorm_mean += dnorm_i;
            dnorm_norm_mean += dnorm_i * norm_bti;
        }
        dnorm_mean = dnorm_mean / C;
        dnorm_norm_mean = dnorm_norm_mean / C;

        // now iterate again and accumulate all the gradients
        for (int i = 0; i < C; i++) {
            float norm_bti = (_In_bt[i] - _Mean_bt) * _Rstd_bt;
            float dnorm_i = _Weight[i] * dOut_bt[i];
            // gradient contribution to bias
            dBias[i] += (float)dOut_bt[i];
            // atomicAdd(&dBias[i], dOut_bt[i]);
            // gradient contribution to weight
            dWeight[i] += (float)(norm_bti * dOut_bt[i]);
            //atomicAdd(&dWeight[i], (norm_bti * dOut_bt[i]));
            // gradient contribution to input
            float dval = 0.0f;
            dval += dnorm_i; // term 1
            dval -= dnorm_mean; // term 2
            dval -= norm_bti * dnorm_norm_mean; // term 3
            dval *= _Rstd_bt; // final scale
            _dIn_bt[i] += (float)dval;
            //atomicAdd(&_dIn_bt[i], dval);
        }
    }

    public static unsafe void layernorm_backward_1(float* dinp, float* dweight, float* dbias,
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
                                      B,
                                      T,
                                      C);
        });
    }

    #region forward
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

    #endregion

#if layernorm_backward
    static int Main() {
        ulong seed = 113;

        int B = 4;
        int T = 64;
        int C = 1024;

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

        var layernorm_backward_0_start = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;

        for (int n = 0; n < 256; n++) {
            layernorm_backward_0(
                h_REF_DIn, h_REF_DWeight, h_REF_DBias,
                h_REF_DOut, h_REF_In, h_REF_Weight, h_REF_Mean, h_REF_Rstd,
                B, T, C);
        }

        var layernorm_backward_0_end = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;

        var layernorm_backward_1_start = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;

        for (int n = 0; n < 256; n++) {
            layernorm_backward_1(
                CUDA_DIn, CUDA_DWeight, CUDA_DBias,
                CUDA_DOut, CUDA_In, CUDA_Weight, CUDA_Mean, CUDA_Rstd,
                B, T, C);
        }

        var layernorm_backward_1_end = Stopwatch.GetTimestamp() / (double)TimeSpan.TicksPerSecond;

        printf("layernorm_backward_0: %f ms\n", (layernorm_backward_0_end - layernorm_backward_0_start) * 1000);
        printf("layernorm_backward_1: %f ms\n", (layernorm_backward_1_end - layernorm_backward_1_start) * 1000);

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

