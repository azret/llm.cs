using System;
using System.IO;

using static kernel32;
using static cuda;
using static stdio;

unsafe static class vectorAddZeroCopy {

    // Macro to aligned up to the memory size in question
    const int MEMORY_ALIGNMENT = 4096;

    static void* MEMORY_ALIGN_UP(void* p, ulong size) {
        return (void*)(((ulong)p + (size - 1)) & (~(size - 1)));
    }

#if vectorAddZeroCopy
    static unsafe int Main() {
        checkCudaErrors(cuInit());

        checkCudaErrors(cuDeviceGet(out var d_Device, 0));

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, d_Device));

        checkCudaErrors(cuCtxSetCurrent(ctx));

        checkCudaErrors(cuDeviceGetAttribute(out int pi, CUdevice_attribute.CU_DEVICE_ATTRIBUTE_CAN_MAP_HOST_MEMORY, d_Device));

        if (pi != 1) { checkCudaErrors(CUresult.CUDA_ERROR_NOT_SUPPORTED); }

        checkCudaErrors(cuModuleLoadData(out var cuModule, File.ReadAllBytes(".\\dev\\cuda\\vectorAdd_kernel64.fatbin")));

        checkCudaErrors(cuModuleGetFunction(out var vecAdd_kernel, cuModule, "VecAdd_kernel"));

        int nelem = 256;

        float* a_UA = (float*)malloc(nelem * sizeof(float) + MEMORY_ALIGNMENT);
        float* b_UA = (float*)malloc(nelem * sizeof(float) + MEMORY_ALIGNMENT);
        float* c_UA = (float*)malloc(nelem * sizeof(float) + MEMORY_ALIGNMENT);

        // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)

        float* a = (float*)MEMORY_ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
        float* b = (float*)MEMORY_ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
        float* c = (float*)MEMORY_ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

        checkCudaErrors(cuMemHostRegister_v2(a, (ulong)(nelem * sizeof(float)), CU_MEMHOSTREGISTER_DEVICEMAP));
        checkCudaErrors(cuMemHostRegister_v2(b, (ulong)(nelem * sizeof(float)), CU_MEMHOSTREGISTER_DEVICEMAP));
        checkCudaErrors(cuMemHostRegister_v2(c, (ulong)(nelem * sizeof(float)), CU_MEMHOSTREGISTER_DEVICEMAP));

        checkCudaErrors(cuMemHostGetDevicePointer_v2(out var d_a, (void*)a, 0));
        checkCudaErrors(cuMemHostGetDevicePointer_v2(out var d_b, (void*)b, 0));
        checkCudaErrors(cuMemHostGetDevicePointer_v2(out var d_c, (void*)c, 0));

        uint threadsPerBlock     = (uint)256;
        uint blocksPerGrid       = (uint)((nelem + threadsPerBlock - 1) / threadsPerBlock);

        void*[] args = { &d_a, &d_b, &d_c, &nelem };

        for (int i = 0; i < nelem; i++) {
            a[i] = 99; b[i] = 1; c[i] = 0;
        }

        // Launch the CUDA kernel
        checkCudaErrors(cuLaunchKernel(
            vecAdd_kernel,
            blocksPerGrid, 1, 1,
            threadsPerBlock, 1, 1, 0,
            CUstream.NULL,
            args,
            null));

        checkCudaErrors(cuCtxSynchronize());

        // Verify result
        bool ok = true;

        for (int i = 0; i < nelem; ++i) {
            float sum = a[i] + b[i];
            if (Math.Abs(c[i] - sum) > 1e-7f) {
                ok = false;
                Console.ForegroundColor = ConsoleColor.Red;
                Console.WriteLine($"ERROR: CPU: {sum} != GPU: {c[i]}.");
                Console.ResetColor();
                break;
            }
        }

        checkCudaErrors(cuMemHostUnregister(a));
        checkCudaErrors(cuMemHostUnregister(b));
        checkCudaErrors(cuMemHostUnregister(c));

        free(a_UA);
        free(b_UA);
        free(c_UA);

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