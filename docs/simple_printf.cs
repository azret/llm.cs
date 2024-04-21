using System;
using System.IO;
using System.Runtime.InteropServices;

using static cuda;
using static nvrtc;
using static std;

unsafe static class simple_printf {

    static uint ceil_div(uint dividend, uint divisor) {
        return (dividend + divisor - 1) / divisor;
    }

#if simple_printf
    static unsafe int Main() {
        checkCudaErrors(cuInit());
        checkCudaErrors(cuDeviceGet(out var dev, 0));

        cuPrintDeviceInfo(dev);

        string fileName = ".\\docs\\simple_printf.cu";

        printf("> Compiling CUDA source file %s...\n", fileName);

        byte[] ptx = compile(File.ReadAllText(fileName), "simple_printf");

        checkCudaErrors(cuCtxCreate_v2(out var ctx, CUctx_flags.CU_CTX_SCHED_AUTO, dev));
        checkCudaErrors(cuCtxSetCurrent(ctx));

        checkCudaErrors(cuModuleLoadData(out var cuModule, ptx));

        // Kernel configuration, where a two-dimensional grid and
        // three-dimensional blocks are configured.
        // dim3 dimGrid(2, 2);
        // dim3 dimBlock(2, 2, 2);
        // testKernel <<< 2, 3 >>> (10);

        IntPtr simple_printf_2;

        checkCudaErrors(cuModuleGetFunction(
            out simple_printf_2,
            cuModule,
            nameof(simple_printf_2)));

        Console.WriteLine();

        uint threadsPerBlock = 11;

        int B = 3;
        int T = 3;
        int C = 3;

        uint x = (uint)((B + threadsPerBlock - 1) / threadsPerBlock);
        uint y = (uint)((T + threadsPerBlock - 1) / threadsPerBlock);
        uint z = (uint)((C + threadsPerBlock - 1) / threadsPerBlock);

        checkCudaErrors(cuLaunchKernel(
            simple_printf_2,
            x, y, z,
            threadsPerBlock, 1, 1,
            0,
            IntPtr.Zero,
            null,
            null));

        checkCudaErrors(cuCtxDestroy_v2(ctx));

        Console.WriteLine();
        printf("Press [Enter] to continue...");
        Console.Out.Flush();
        Console.ReadKey();

        return 0;
    }

#endif
}