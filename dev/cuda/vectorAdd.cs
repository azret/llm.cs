using System;
using System.IO;

using static kernel32;
using static cuda;

unsafe static class vectorAdd {
#if vectorAdd
    static unsafe int Main() {
        checkCudaErrors(cuInit());

        checkCudaErrors(cuDeviceGet(out var device, 0));

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, device));

        checkCudaErrors(cuCtxSetCurrent(ctx));

        checkCudaErrors(cuModuleLoadData(out var cuModule, File.ReadAllBytes(".\\dev\\cuda\\vectorAdd_kernel64.fatbin")));

        checkCudaErrors(cuModuleGetFunction(out var vecAdd_kernel, cuModule, "VecAdd_kernel"));

        int N = 16;

        var h_A = (float*)malloc(N * sizeof(float));
        var h_B = (float*)malloc(N * sizeof(float));
        var h_C = (float*)malloc(N * sizeof(float));

        for (int i = 0; i < N; i++) {
            h_A[i] = 99; h_B[i] = 1; h_C[i] = 0;
        }

        checkCudaErrors(cuMemAlloc_v2(out var d_A, (ulong)N * sizeof(float)));
        checkCudaErrors(cuMemAlloc_v2(out var d_B, (ulong)N * sizeof(float)));
        checkCudaErrors(cuMemAlloc_v2(out var d_C, (ulong)N * sizeof(float)));

        checkCudaErrors(cuMemcpyHtoD_v2(d_A, h_A, (ulong)N * sizeof(float)));
        checkCudaErrors(cuMemcpyHtoD_v2(d_B, h_B, (ulong)N * sizeof(float)));

        uint threadsPerBlock     = (uint)256;
        uint blocksPerGrid       = (uint)((N + threadsPerBlock - 1) / threadsPerBlock);

        void*[] args = { &d_A, &d_B, &d_C, &N };

        // Launch the CUDA kernel
        checkCudaErrors(cuLaunchKernel(
            vecAdd_kernel,
            blocksPerGrid, 1, 1,
            threadsPerBlock, 1, 1, 0,
            CUstream.NULL,
            args,
            null));

        checkCudaErrors(cuCtxSynchronize());

        checkCudaErrors(cuMemcpyDtoH_v2(h_C, d_C, (ulong)N * sizeof(float)));

        // Verify result
        bool ok = true;

        for (int i = 0; i < N; ++i) {
            float sum = h_A[i] + h_B[i];
            if (Math.Abs(h_C[i] - sum) > 1e-7f) {
                ok = false;
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"ERROR: CPU: {sum} != GPU: {h_C[i]}.");
                Console.ResetColor();
                break;
            }
        }

        checkCudaErrors(cuMemFree_v2(d_A));
        checkCudaErrors(cuMemFree_v2(d_B));
        checkCudaErrors(cuMemFree_v2(d_C));

        free(h_A);
        free(h_B);
        free(h_C);

        checkCudaErrors(cuCtxDestroy_v2(ctx));

        if (ok) {
            Console.ForegroundColor = ConsoleColor.Green;
            Console.WriteLine($"OK.");
            Console.ResetColor();
        } else {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"FAILED.");
            Console.ResetColor();
        }

        Console.ReadKey();
        return 0;
    }

#endif
}