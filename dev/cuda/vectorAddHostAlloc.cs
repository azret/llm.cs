using System;
using System.IO;

using static cuda;
using static stdio;

unsafe static class vectorAddHostAlloc {

#if vectorAddHostAlloc
    static unsafe int Main() {
        checkCudaErrors(cuInit());

        checkCudaErrors(cuDeviceGet(out var dev, 0));

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));

        checkCudaErrors(cuCtxSetCurrent(ctx));

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
            System.Text.Encoding.ASCII.GetString(devName, 256),
            major,
            minor);

        free(devName);

        if (pi != 1) { checkCudaErrors(CUresult.CUDA_ERROR_NOT_SUPPORTED); }

        checkCudaErrors(cuModuleLoadData(out var cuModule, File.ReadAllBytes(".\\dev\\cuda\\vectorAdd_kernel64.fatbin")));

        checkCudaErrors(cuModuleGetFunction(out var vecAdd_kernel, cuModule, "VecAdd_kernel"));

        bool bPinGenericMemory = false;

        int nelem = 256;

        float* a_UA = null;
        float* b_UA = null;
        float* c_UA = null;

        float* a = null;
        float* b = null;
        float* c = null;

        if (bPinGenericMemory) {

            a_UA = (float*)malloc(nelem * sizeof(float) + MEMORY_ALIGNMENT);
            b_UA = (float*)malloc(nelem * sizeof(float) + MEMORY_ALIGNMENT);
            c_UA = (float*)malloc(nelem * sizeof(float) + MEMORY_ALIGNMENT);

            // We need to ensure memory is aligned to 4K (so we will need to padd memory accordingly)

            a = (float*)MEMORY_ALIGN_UP(a_UA, MEMORY_ALIGNMENT);
            b = (float*)MEMORY_ALIGN_UP(b_UA, MEMORY_ALIGNMENT);
            c = (float*)MEMORY_ALIGN_UP(c_UA, MEMORY_ALIGNMENT);

            checkCudaErrors(cuMemHostRegister_v2(a, (ulong)(nelem * sizeof(float)), CU_MEMHOSTREGISTER_DEVICEMAP));
            checkCudaErrors(cuMemHostRegister_v2(b, (ulong)(nelem * sizeof(float)), CU_MEMHOSTREGISTER_DEVICEMAP));
            checkCudaErrors(cuMemHostRegister_v2(c, (ulong)(nelem * sizeof(float)), CU_MEMHOSTREGISTER_DEVICEMAP));

        } else {

            checkCudaErrors(cuMemAllocHost_v2((void**)&a, (ulong)(nelem * sizeof(float))));
            checkCudaErrors(cuMemAllocHost_v2((void**)&b, (ulong)(nelem * sizeof(float))));
            checkCudaErrors(cuMemAllocHost_v2((void**)&c, (ulong)(nelem * sizeof(float))));

        }

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
            IntPtr.Zero,
            args,
            null));

        checkCudaErrors(cuCtxSynchronize());

        /* Compare the results V1 */

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

        /* Compare the results V2 */

        printf("> Checking the results from vectorAddGPU() ...\n");
        float errorNorm = 0.0f;
        float refNorm = 0.0f;

        for (int n = 0; n < nelem; n++) {
            float cpu = a[n] + b[n]; // compute on CPU
            float diff = c[n] - cpu;
            errorNorm += diff * diff;
            refNorm += cpu * cpu;
        }

        errorNorm = (float)Math.Sqrt((double)errorNorm);
        refNorm = (float)Math.Sqrt((double)refNorm);
        
        if (errorNorm / refNorm > 1e-6f) {
            ok = false;
        }

        if (bPinGenericMemory) {

            checkCudaErrors(cuMemHostUnregister(a));
            checkCudaErrors(cuMemHostUnregister(b));
            checkCudaErrors(cuMemHostUnregister(c));

            free(a_UA);
            free(b_UA);
            free(c_UA);

        } else {

            checkCudaErrors(cuMemFreeHost(a));
            checkCudaErrors(cuMemFreeHost(b));
            checkCudaErrors(cuMemFreeHost(c));

        }

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