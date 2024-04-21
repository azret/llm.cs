using System;
using System.IO;

using static cuda;
using static nvrtc;
using static std;

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

    static uint ceil_div(uint dividend, uint divisor) {
        return (dividend + divisor - 1) / divisor;
    }

#if matmul_forward
    static unsafe int Main() {
        checkCudaErrors(cuInit());
        checkCudaErrors(cuDeviceGet(out var dev, 0));

        cuPrintDeviceInfo(dev);

        string fileName = ".\\dev\\cuda\\matmul_forward.cu";

        printf("> Compiling CUDA source file %s...\n", fileName);

        byte[] ptx = compile(File.ReadAllText(fileName), "matmul_forward");

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));
        checkCudaErrors(cuCtxSetCurrent(ctx));

        checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));

        Console.WriteLine();

        ulong seed = 37;

        int B = 4;
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

        cuMemAlloc_v2(out var d_Out, (ulong)(B * T * OC * sizeof(float)));
        cuMemAlloc_v2(out var d_In, (ulong)(B * T * C * sizeof(float)));
        cuMemAlloc_v2(out var d_Weight, (ulong)(OC * C * sizeof(float)));
        cuMemAlloc_v2(out var d_Bias, (ulong)(OC * sizeof(float)));

        cuMemcpyHtoD_v2(d_Out, h_Out, (ulong)(B * T * OC * sizeof(float)));
        cuMemcpyHtoD_v2(d_In, h_In, (ulong)(B * T * C * sizeof(float)));
        cuMemcpyHtoD_v2(d_Weight, h_Weight, (ulong)(C * OC * sizeof(float)));
        cuMemcpyHtoD_v2(d_Bias, h_Bias, (ulong)(OC * sizeof(float)));

        for (int i = 0; i < B * T * OC; i++) { h_Out[i] = randf(&seed) * 2.0f - 1.0f; }
        matmul_forward_cpu(
            h_Out,
            h_In, h_Weight, h_Bias,
            B, T, C, OC);

        IntPtr matmul_forward_kernel = IntPtr.Zero;

        checkCudaErrors(cuModuleGetFunction(
            out matmul_forward_kernel,
            cuModule,
            nameof(matmul_forward_kernel)));

        uint sqrt_block_size = 4;

        void*[] args = { &d_Out, &d_In, &d_Weight, &d_Bias, &BT, &C, &OC };

        checkCudaErrors(cuLaunchKernel(
            matmul_forward_kernel,
            ceil_div((uint)B * (uint)T, sqrt_block_size), ceil_div((uint)OC, sqrt_block_size), 1,
            sqrt_block_size, sqrt_block_size, 1,
            0,
            IntPtr.Zero,
            args,
            null));

        checkCudaErrors(cuCtxSynchronize());

        bool ok = validate_results(d_Out, h_Out, B * T * C);

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
        printf("Press [Enter] to continue...");
        Console.Out.Flush();
        Console.ReadKey();

        return 0;
    }

    static bool validate_results(IntPtr d_Out, float* h_Out, int N) {
        bool ok = true;
        float* found = (float*)malloc(N * sizeof(float));
        checkCudaErrors(cuMemcpyDtoH_v2(found, d_Out, (ulong)(N * sizeof(float))));
        int faults = 0;
        int prints = 0;
        for (int i = 0; i < N; ++i) {
            if (Math.Abs(found[i] - h_Out[i]) > 1e-4f) {
                ok = false;
                if (faults < 7) {
                    Console.ForegroundColor = ConsoleColor.Red;
                    Console.WriteLine($"ERROR: CPU: {h_Out[i]} != GPU: {found[i]}");
                    Console.ResetColor();
                }
                faults++;
                break;
            } else {
                if (faults == 0 && prints < 5) Console.WriteLine($"OK: CPU: {h_Out[i]} == GPU: {found[i]}");
                prints++;
            }
        }
        free(found);
        return ok;
    }

#endif
}