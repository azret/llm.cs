using System;

using static cuda;
using static nvrtc;
using static std;
using static common;
using static time;

unsafe static class matmul_forward {
    // Reference implementation
    public static void matmul_forward_cpu(float* _Out,
                                       float* _In,
                                       float* _Weight,
                                       float* _Bias,
                                       int B,
                                       int T,
                                       int C,
                                       int OC) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                for (int o = 0; o < OC; o++) {
                    float* x = _In + b * T * C + t * C;
                    float* y = _Out + b * T * OC + t * OC;
                    float* w = _Weight + o * C;
                    float sum = (_Bias != null) ? _Bias[o] : 0.0f;
                    for (int i = 0; i < C; i++) {
                        sum += x[i] * w[i];
                    }
                    y[o] = sum;
                }
            }
        }
    }

    private static unsafe void matmul_forward_1(int B, int T, int BT, int C, int OC, IntPtr d_Out, IntPtr d_In, IntPtr d_Weight,
        IntPtr d_Bias, IntPtr matmul_forward_kernel, uint block_size) {

        void*[] args = { &d_Out, &d_In, &d_Weight, &d_Bias, &BT, &C, &OC };

        checkCudaErrors(cuLaunchKernel(
            matmul_forward_kernel,
            CEIL_DIV((uint)B * (uint)T, block_size), CEIL_DIV((uint)OC, block_size), 1,
            block_size, block_size, 1,
            0,
            IntPtr.Zero,
            args,
            null));
    }

#if matmul_forward
    static unsafe int Main() {
        checkCudaErrors(cuInit());
        checkCudaErrors(cuDeviceGet(out var dev, 0));

        cuPrintDeviceInfo(dev);

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));
        checkCudaErrors(cuCtxSetCurrent(ctx));

        cuPrintCurrentContextInfo();

        ulong seed = 37;

        int B = 5;
        int T = 64;
        int BT = B * T;
        int C = 768;
        int OC = C * 4; // expansion of 4, e.g. in the MLP

        float* h_Out = (float*)malloc(B * T * OC * sizeof(float));
        float* h_In = (float*)malloc(B * T * C * sizeof(float));
        float* h_Weight = (float*)malloc(OC * C * sizeof(float));
        float* h_Bias = (float*)malloc(OC * sizeof(float));

        for (int i = 0; i < B * T * OC; i++) { h_Out[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * T * C; i++) { h_In[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < OC * C; i++) { h_Weight[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < OC; i++) { h_Bias[i] = randf(&seed) * 2.0f - 1.0f; }

        float* p_d_Out = null;
        float* p_d_In = null;
        float* p_d_Weight = null;
        float* p_d_Bias = null;

        IntPtr d_Out;
        IntPtr d_In;
        IntPtr d_Weight;
        IntPtr d_Bias;

        checkCudaErrors(cuMemAlloc_v2(out d_Out, (ulong)(B * T * OC * sizeof(float))));
        checkCudaErrors(cuMemAlloc_v2(out d_In, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemAlloc_v2(out d_Weight, (ulong)(OC * C * sizeof(float))));
        checkCudaErrors(cuMemAlloc_v2(out d_Bias, (ulong)(OC * sizeof(float))));

        checkCudaErrors(cuMemcpyHtoD_v2(d_Out, h_Out, (ulong)(B * T * OC * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_In, h_In, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_Weight, h_Weight, (ulong)(C * OC * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_Bias, h_Bias, (ulong)(OC * sizeof(float))));

        for (int i = 0; i < B * T * OC; i++) { h_Out[i] = randf(&seed) * 2.0f - 1.0f; }
        matmul_forward_cpu(
            h_Out,
            h_In, h_Weight, h_Bias,
            B, T, C, OC);

        IntPtr matmul_forward_kernel = IntPtr.Zero;

        printf("> Compiling CUDA source file %s...\n", "train_gpt2_cuda.cu");

        byte[] ptx = CompileFromEmbeddedResource("LLM.dev.train_gpt2_cuda.cu");

        checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));

        checkCudaErrors(cuModuleGetFunction(
            out matmul_forward_kernel,
            cuModule,
            nameof(matmul_forward_kernel)));

        Console.WriteLine();

        uint block_size = 4;

        matmul_forward_1(B, T, BT, C, OC, d_Out, d_In, d_Weight, d_Bias, matmul_forward_kernel, block_size);

        float* p_p_d_Out = (float*)malloc(B * T * C * sizeof(float));
        checkCudaErrors(cuMemcpyDtoH_v2(p_p_d_Out, d_Out, (ulong)(B * T * C * sizeof(float))));
        bool ok = validate_results(p_p_d_Out, h_Out, B * T * C);
        free(p_p_d_Out);

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
        uint[] blocks = { 4, 8, 16, 32 };
        for (int block = 0; block < blocks.Length; block++) {
            block_size = blocks[block];
            timespec start, end;
            clock_gettime(CLOCK_MONOTONIC, &start);
            for (int i = 0; i < 1024; i++) {
                matmul_forward_1(B, T, BT, C, OC, d_Out, d_In, d_Weight, d_Bias, matmul_forward_kernel, block_size);
            }
            clock_gettime(CLOCK_MONOTONIC, &end);
            double time_elapsed_s = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
            Console.Write("matmul_forward_1(block_size = {0}): {1} ms\n", block_size, time_elapsed_s * 1000);
            p_p_d_Out = (float*)malloc(B * T * C * sizeof(float));
            checkCudaErrors(cuMemcpyDtoH_v2(p_p_d_Out, d_Out, (ulong)(B * T * C * sizeof(float))));
            if (!validate_results(p_p_d_Out, h_Out, B * T * C, "matmul_forward_1", bPrint: false)) {
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"FAILED.");
                Console.ResetColor();
            }
            free(p_p_d_Out);
        }

        checkCudaErrors(cuCtxSynchronize());

        checkCudaErrors(cuMemFree_v2(d_In));
        checkCudaErrors(cuMemFree_v2(d_Out));
        checkCudaErrors(cuMemFree_v2(d_Weight));
        checkCudaErrors(cuMemFree_v2(d_Bias));

        free(h_Out);
        free(h_In);
        free(h_Weight);
        free(h_Bias);

        checkCudaErrors(cuCtxDestroy_v2(ctx));

        Console.WriteLine();
        printf("Press [Enter] to continue...");
        Console.Out.Flush();
        Console.ReadKey();

        return 0;
    }

#endif
}