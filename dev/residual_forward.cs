using System;

using static cuda;
using static nvrtc;
using static std;

public static unsafe class residual_forward {
    static unsafe void residual_forward_cpu(float* out_, float* inp1, float* inp2, int N) {
        for (int i = 0; i < N; i++) {
            out_[i] = inp1[i] + inp2[i];
        }
    }

    static string residual_forward_kernel = @"
extern ""C"" __global__  void residual_forward_kernel(float* out, const float* inp1, const float* inp2, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = __ldcs(&inp1[idx]) + __ldcs(&inp2[idx]);
    }
}
";

#if residual_forward
    static unsafe int Main() {
        checkCudaErrors(cuInit());
        checkCudaErrors(cuDeviceGet(out var dev, 0));

        cuPrintDeviceInfo(dev);

        printf("> Compiling CUDA source file %s...\n", "train_gpt2_cuda.cu");

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));
        checkCudaErrors(cuCtxSetCurrent(ctx));
        cuPrintCurrentContextInfo();

        Console.WriteLine();

        ulong seed = 37;

        int B = 4;
        int T = 64;
        int C = 768;

        float* h_Out = (float*)malloc(B * T * C * sizeof(float));
        float* h_In1 = (float*)malloc(B * T * C * sizeof(float));
        float* h_In2 = (float*)malloc(B * T * C * sizeof(float));

        for (int i = 0; i < B * T * C; i++) { h_Out[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * T * C; i++) { h_In1[i] = randf(&seed) * 2.0f - 1.0f; }
        for (int i = 0; i < B * T * C; i++) { h_In2[i] = randf(&seed) * 2.0f - 1.0f; }

        float* p_d_Out = null;
        float* p_d_In1 = null;
        float* p_d_In2 = null;

        IntPtr d_Out;
        IntPtr d_In1;
        IntPtr d_In2;

        checkCudaErrors(cuMemAllocHost_v2((void**)&p_d_Out, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemAllocHost_v2((void**)&p_d_In1, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemAllocHost_v2((void**)&p_d_In2, (ulong)(B * T * C * sizeof(float))));

        checkCudaErrors(cuMemHostGetDevicePointer_v2(out d_Out, (void*)p_d_Out, 0));
        checkCudaErrors(cuMemHostGetDevicePointer_v2(out d_In1, (void*)p_d_In1, 0));
        checkCudaErrors(cuMemHostGetDevicePointer_v2(out d_In2, (void*)p_d_In2, 0));

        checkCudaErrors(cuMemcpyHtoD_v2(d_Out, h_Out, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_In1, h_In1, (ulong)(B * T * C * sizeof(float))));
        checkCudaErrors(cuMemcpyHtoD_v2(d_In2, h_In2, (ulong)(B * T * C * sizeof(float))));

        for (int i = 0; i < B * T * C; i++) { h_Out[i] = randf(&seed) * 2.0f - 1.0f; }

        residual_forward_cpu(
            h_Out,
            h_In1, h_In2,
            B * T * C);

        byte[] ptx = CompileFromSourceCode(residual_forward_kernel, nameof(residual_forward_kernel));

        checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));

        IntPtr k_residual_forward_kernel = IntPtr.Zero;

        checkCudaErrors(cuModuleGetFunction(
            out k_residual_forward_kernel,
            cuModule,
            nameof(residual_forward_kernel)));

        uint block_size = 256;

        int d_N = B * T * C;

        void*[] args = { &d_Out, &d_In1, &d_In2, &d_N };

        checkCudaErrors(cuLaunchKernel(
            k_residual_forward_kernel,
            CEIL_DIV((uint)d_N, block_size), 1, 1,
            block_size, 1, 1,
            0,
            IntPtr.Zero,
            args,
            null));

        // checkCudaErrors(cuCtxSynchronize());

        float* p_p_d_Out = (float*)malloc(B * T * C * sizeof(float));
        checkCudaErrors(cuMemcpyDtoH_v2(p_p_d_Out, d_Out, (ulong)(B * T * C * sizeof(float))));
        bool ok = validate_results(p_p_d_Out, h_Out, B * T * C);
        free(p_p_d_Out);

        checkCudaErrors(cuMemFreeHost(p_d_Out));
        checkCudaErrors(cuMemFreeHost(p_d_In1));
        checkCudaErrors(cuMemFreeHost(p_d_In2));

        free(h_Out);
        free(h_In1);
        free(h_In2);

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

#endif
}