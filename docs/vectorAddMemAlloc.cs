using System;
using System.IO;
using System.Runtime.InteropServices;

using static cuda;
using static nvrtc;
using static std;

unsafe static class vectorAddMemAlloc {
#if vectorAddMemAlloc
    static unsafe int Main() {

        string fileName = ".\\dev\\cuda\\vectorAdd_kernel64.cu";

        byte[] ptx = compile(File.ReadAllText(fileName), "vectorAdd_kernel64");

        checkCudaErrors(cuInit());

        checkCudaErrors(cuDeviceGet(out var dev, 0));

        checkCudaErrors(cuDeviceGetAttribute(out int pi, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, dev));

        // get compute capabilities and the devicename
        checkCudaErrors(cuDeviceGetAttribute(
            out int major, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, dev));
        checkCudaErrors(cuDeviceGetAttribute(
            out int minor, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, dev));

        byte* devName = (byte*)malloc(256 + 1);

        checkCudaErrors(cuDeviceGetName(devName, 256, dev));

        devName[256] = (byte)'\0';

        printf("> GPU Device %s has SM %d.%d compute capability\n",
            Marshal.PtrToStringAnsi((IntPtr)devName),
            major,
            minor);

        free(devName);

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));

        checkCudaErrors(cuCtxSetCurrent(ctx));

        checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));

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

        uint threadsPerBlock = (uint)256;
        uint blocksPerGrid = (uint)((N + threadsPerBlock - 1) / threadsPerBlock);

        void*[] args = { &d_A, &d_B, &d_C, &N };

        // Launch the CUDA kernel
        checkCudaErrors(cuLaunchKernel(
            vecAdd_kernel,
            blocksPerGrid, 1, 1,
            threadsPerBlock, 1, 1, 0,
            IntPtr.Zero,
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