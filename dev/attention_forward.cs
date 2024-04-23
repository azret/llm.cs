using System;

using static cuda;
using static nvrtc;
using static std;
using static common;
using static time;

public static unsafe class attention_forward {
    static unsafe void attention_forward_cpu(float* out_, float* preatt, float* att,
                       float* inp,
                       int B, int T, int C, int NH) {
        // input is (B, T, 3C) Q,K,V
        // preatt, att are (B, NH, T, T)
        // output is (B, T, C)
        int C3 = C * 3;
        int hs = C / NH; // head size
        float scale = 1.0f / sqrtf(hs);

        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int h = 0; h < NH; h++) {
                    float* query_t = inp + b * T * C3 + t * C3 + h * hs;
                    float* preatt_bth = preatt + b * NH * T * T + h * T * T + t * T;
                    float* att_bth = att + b * NH * T * T + h * T * T + t * T;

                    // pass 1: calculate query dot key and maxval
                    float maxval = -10000.0f; // TODO something better
                    for (int t2 = 0; t2 <= t; t2++) {
                        float* key_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C; // +C because it's key

                        // (query_t) dot (key_t2)
                        float val = 0.0f;
                        for (int i = 0; i < hs; i++) {
                            val += query_t[i] * key_t2[i];
                        }
                        val *= scale;
                        if (val > maxval) {
                            maxval = val;
                        }

                        preatt_bth[t2] = val;
                    }
                    // pad with -INFINITY outside of autoregressive region for debugging comparisons
                    for (int t2 = t + 1; t2 < T; t2++) {
                        preatt_bth[t2] = float.NegativeInfinity;
                    }

                    // pass 2: calculate the exp and keep track of sum
                    float expsum = 0.0f;
                    for (int t2 = 0; t2 <= t; t2++) {
                        float expv = expf(preatt_bth[t2] - maxval);
                        expsum += expv;
                        att_bth[t2] = expv;
                    }
                    float expsum_inv = expsum == 0.0f ? 0.0f : 1.0f / expsum;

                    // pass 3: normalize to get the softmax
                    for (int t2 = 0; t2 < T; t2++) {
                        if (t2 <= t) {
                            att_bth[t2] *= expsum_inv;
                        } else {
                            // causal attention mask. not strictly necessary to set to zero here
                            // only doing this explicitly for debugging and checking to PyTorch
                            att_bth[t2] = 0.0f;
                        }
                    }

                    // pass 4: accumulate weighted values into the output of attention
                    float* out_bth = out_ +b * T * C + t * C + h * hs;
                    for (int i = 0; i < hs; i++) { out_bth[i] = 0.0f; }
                    for (int t2 = 0; t2 <= t; t2++) {
                        float* value_t2 = inp + b * T * C3 + t2 * C3 + h * hs + C * 2; // +C*2 because it's value
                        float att_btht2 = att_bth[t2];
                        for (int i = 0; i < hs; i++) {
                            out_bth[i] += att_btht2 * value_t2[i];
                        }
                    }
                }
            }
        }
    }

    private static void attention_forward_1(int B, int T, int C, int NH, IntPtr d_Out, IntPtr d_preatt, IntPtr d_att, IntPtr d_inp, uint block_size, IntPtr attention_query_key_kernel, IntPtr attention_softmax_kernel, IntPtr attention_value_kernel) {
        int d_B = B;
        int d_T = T;
        int d_C = C;
        int d_NH = NH;

        void*[] args = new void*[] { &d_preatt, &d_inp, &d_B, &d_T, &d_C, &d_NH };

        checkCudaErrors(cuLaunchKernel(
            attention_query_key_kernel,
            CEIL_DIV((uint)(B * NH * T * T), block_size), 1, 1,
            block_size, 1, 1,
            0,
            IntPtr.Zero,
            args,
            null));

        args = new void*[] { &d_att, &d_preatt, &d_B, &d_T, &d_NH };

        checkCudaErrors(cuLaunchKernel(
            attention_softmax_kernel,
            CEIL_DIV((uint)(B * T * NH), block_size), 1, 1,
            block_size, 1, 1,
            0,
            IntPtr.Zero,
            args,
            null));

        args = new void*[] { &d_Out, &d_att, &d_inp, &d_B, &d_T, &d_C, &d_NH };

        checkCudaErrors(cuLaunchKernel(
            attention_value_kernel,
            CEIL_DIV((uint)(B * T * NH), block_size), 1, 1,
            block_size, 1, 1,
            0,
            IntPtr.Zero,
            args,
            null));
    }


#if attention_forward
    static unsafe int Main() {
        checkCudaErrors(cuInit());
        checkCudaErrors(cuDeviceGet(out var dev, 0));

        cuPrintDeviceInfo(dev);

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));
        checkCudaErrors(cuCtxSetCurrent(ctx));
        cuPrintCurrentContextInfo();

        ulong seed = 37;

        int B = 4;
        int T = 64;
        int C = 768;
        int NH = 12;

        float* outp = (float*)malloc(B * T * C * sizeof(float));
        float* preatt = (float*)malloc(B * NH * T * T * sizeof(float));
        float* att = (float*)malloc(B * NH * T * T * sizeof(float));
        float* inp = (float*)malloc(B * T * 3 * C * sizeof(float));

        for (int i = 0; i < B * T * C; i++) { outp[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * NH * T * T; i++) { preatt[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * NH * T * T; i++) { att[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * T * 3 * C; i++) { inp[i] = randf(&seed) * 2.0f - 1.0f; }

        IntPtr d_Out;
        IntPtr d_preatt;
        IntPtr d_att;
        IntPtr d_inp;

        checkCudaErrors(cuMemAlloc_v2(out d_Out, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemAlloc_v2(out d_preatt, (ulong)(B * NH * T * T * sizeof(float))));
        checkCudaErrors(cuMemAlloc_v2(out d_att, (ulong)(B * NH * T * T * sizeof(float))));
        checkCudaErrors(cuMemAlloc_v2(out d_inp, (ulong)(B * T * 3 * C * sizeof(float))));

        checkCudaErrors(cuMemcpyHtoD_v2(d_Out, outp, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_preatt, preatt, (ulong)(B * NH * T * T * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_att, att, (ulong)(B * NH * T * T * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_inp, inp, (ulong)(B * T * 3 * C * sizeof(float))));

        for (int i = 0; i < B * T * C; i++) { outp[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * NH * T * T; i++) { preatt[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * NH * T * T; i++) { att[i] = randf(&seed) * 2.0f - 1.0f; }

        attention_forward_cpu(
            outp,
            preatt, att, inp,
            B, T, C, NH);

        printf("> Compiling CUDA source file %s...\n", "train_gpt2_cuda.cu");

        byte[] ptx = CompileFromEmbeddedResource("LLM.dev.train_gpt2_cuda.cu");

        checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));

        checkCudaErrors(cuModuleGetFunction(out var attention_query_key_kernel, cuModule, "attention_query_key_kernel"));
        checkCudaErrors(cuModuleGetFunction(out var attention_softmax_kernel, cuModule, "attention_softmax_kernel"));
        checkCudaErrors(cuModuleGetFunction(out var attention_value_kernel, cuModule, "attention_value_kernel"));

        uint block_size = 256;

        attention_forward_1(B, T, C, NH, d_Out, d_preatt, d_att, d_inp, block_size, attention_query_key_kernel, attention_softmax_kernel, attention_value_kernel);

        checkCudaErrors(cuCtxSynchronize());

        float* p_p_d_Out = (float*)malloc(B * T * C * sizeof(float));
        checkCudaErrors(cuMemcpyDtoH_v2(p_p_d_Out, d_Out, (ulong)(B * T * C * sizeof(float))));
        bool ok = validate_results(p_p_d_Out, outp, B * T * C, "\nout");
        free(p_p_d_Out);
        
        float* p_p_d_preatt = (float*)malloc(B * NH * T * T * sizeof(float));
        checkCudaErrors(cuMemcpyDtoH_v2(p_p_d_preatt, d_preatt, (ulong)(B * NH * T * T * sizeof(float))));
        ok = validate_results(p_p_d_preatt, preatt, B * NH * T * T, "\npreatt");
        free(p_p_d_preatt);
        
        float* p_p_d_att = (float*)malloc(B * NH * T * T * sizeof(float));
        checkCudaErrors(cuMemcpyDtoH_v2(p_p_d_att, d_att, (ulong)(B * NH * T * T * sizeof(float))));
        ok = validate_results(p_p_d_att, att, B * NH * T * T, "\natt");
        free(p_p_d_att);

        Console.WriteLine();
        if (ok) {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"OK");
            Console.ResetColor();
        } else {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"FAILED.");
            Console.ResetColor();
        }

        Console.WriteLine();
        uint[] blocks = { 32, 64, 128, 256, 1024 };
        for (int block = 0; block < blocks.Length; block++) {
            block_size = blocks[block];
            timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            for (int i = 0; i < 2048; i++) {
                attention_forward_1(B, T, C, NH, d_Out, d_preatt, d_att, d_inp, block_size, attention_query_key_kernel, attention_softmax_kernel, attention_value_kernel);
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            Console.Write("attention_forward_1(block_size = {0}): {1} ms\n", block_size, time_elapsed_s * 1000);
        }

        checkCudaErrors(cuMemFree_v2(d_Out));
        checkCudaErrors(cuMemFree_v2(d_preatt));
        checkCudaErrors(cuMemFree_v2(d_att));
        checkCudaErrors(cuMemFree_v2(d_inp));

        free(outp);
        free(preatt);
        free(att);
        free(inp);

        checkCudaErrors(cuCtxDestroy_v2(ctx));

        Console.WriteLine();
        printf("Press [Enter] to continue...");
        Console.Out.Flush();
        Console.ReadKey();

        return 0;
    }
#endif
}