using System;

using static kernel32;
using static MathF;

public static unsafe class MatMul {

    /*  reference Implementation */

    static void matmul_forward_cpu_kernel(int b,
                                          int t,
                                          float* _Out,
                                          float* _In,
                                          float* _Weight,
                                          float* _Bias,
                                          int T,
                                          int C,
                                          int OC) {
        float* x = _In + b * T * C + t * C;
        float* y = _Out + b * T * OC + t * OC;

        for (int o = 0; o < OC; o++) {
            float val = (_Bias != null) ? _Bias[o] : 0.0f;
            float* w = _Weight + o * C;
            for (int i = 0; i < C; i++) {
                val += x[i] * w[i];
            }
            y[o] = val;
        }
    }

    public static void matmul_forward_cpu_0(float* _Out,
                                       float* _In,
                                       float* _Weight,
                                       float* _Bias,
                                       int B,
                                       int T,
                                       int C,
                                       int OC) {
        for (int b = 0; b < B; b++) {
            for (int t = 0; t < T; t++) {
                matmul_forward_cpu_kernel(
                    b,
                    t,
                    _Out,
                    _In,
                    _Weight,
                    _Bias,
                    T,
                    C,
                    OC);
            }
        }
    }

    /* indexing sanity check */

    public static void matmul_forward_cpu_1(float* _Out,
                               float* _In,
                               float* _Weight,
                               float* _Bias,
                               int B,
                               int T,
                               int C,
                               int OC) {
        for (int bt = 0; bt < B * T; bt++) {
            int b = bt / T;
            int t = bt % T;

            matmul_forward_cpu_kernel(
                    b,
                    t,
                    _Out,
                    _In,
                    _Weight,
                    _Bias,
                    T,
                    C,
                    OC);
        }
    }

    /* parallelize over B, T */

    public static void matmul_forward_cpu_2(float* _Out,
                           float* _In,
                           float* _Weight,
                           float* _Bias,
                           int B,
                           int T,
                           int C,
                           int OC) {

        System.Threading.Tasks.Parallel.For(0, B * T, (bt) => {
            int b = bt / T;
            int t = bt % T;

            matmul_forward_cpu_kernel(
                    b,
                    t,
                    _Out,
                    _In,
                    _Weight,
                    _Bias,
                    T,
                    C,
                    OC);
        });
    }

    static unsafe void validate_result(float* device_result, float* cpu_reference,
        string name, int num_elements, float tolerance = 1e-4f) {
        printf("%s:\n", name);
        for (int i = 0; i < num_elements; i++) {
            // print the first few comparisons
            if (i < 5) {
                printf("%f %f\n", cpu_reference[i], device_result[i]);
            }
            // ensure correctness for all elements
            if (Math.Abs(cpu_reference[i] - device_result[i]) > tolerance) {
                printf("Mismatch of %s at %d: %f vs %f\n", name, i, cpu_reference[i], device_result[i]);
                return;
            }
        }
        printf("OK\n");
    }

#if matmul_forward3
    static int Main() {
        ulong seed = 37;

        int B = 2;
        int T = 5;

        int C = 768;
        int OC = 768 * 4; // expansion of 4, e.g. in the MLP

        // // set up the device
        // int deviceIdx = 0;
        // cudaCheck(cudaSetDevice(deviceIdx));
        // cudaDeviceProp deviceProp;
        // cudaGetDeviceProperties(&deviceProp, deviceIdx);
        // printf("Device %d: %s\n", deviceIdx, deviceProp.name);
        // 
        // // setup cuBLAS and cuBLASLt
        // cublasCheck(cublasCreate(&cublas_handle));
        // cublasCheck(cublasLtCreate(&cublaslt_handle));
        // // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
        // int enable_tf32 = deviceProp.major >= 8 ? 1 : 0;
        // printf("enable_tf32: %d\n", enable_tf32);
        // cublas_compute_type = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;
        // cublasMath_t cublas_math_mode = enable_tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
        // cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));
        // // setup the (global) cuBLASLt workspace
        // cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

        // create host memory of random numbers

        float* h_Out = (float*)malloc(B * T * OC * sizeof(float));
        float* h_In = (float*)malloc(B * T * C * sizeof(float));
        float* h_Weight = (float*)malloc(OC * C * sizeof(float));
        float* h_Bias = (float*)malloc(OC * sizeof(float));

        for (int i = 0; i < B * T * OC; i++) { h_Out[i] = randf(&seed); }
        for (int i = 0; i < B * T * C; i++) { h_In[i] = randf(&seed); }
        for (int i = 0; i < OC * C; i++) { h_Weight[i] = randf(&seed); }
        for (int i = 0; i < OC; i++) { h_Bias[i] = randf(&seed); }

        float* d_Out = (float*)malloc(B * T * OC * sizeof(float));
        float* d_In = (float*)malloc(B * T * C * sizeof(float));
        float* d_Weight = (float*)malloc(OC * C * sizeof(float));
        float* d_Bias = (float*)malloc(OC * sizeof(float));

        CopyMemory(d_In, h_In, B * T * C * sizeof(float));
        CopyMemory(d_Weight, h_Weight, C * OC * sizeof(float));
        CopyMemory(d_Bias, h_Bias, OC * sizeof(float));

        // // move to GPU
        // float* d_out;
        // float* d_inp;
        // float* d_weight;
        // float* d_bias;
        // cudaCheck(cudaMalloc(&d_out, B * T * OC * sizeof(float)));
        // cudaCheck(cudaMalloc(&d_inp, B * T * C * sizeof(float)));
        // cudaCheck(cudaMalloc(&d_weight, C * OC * sizeof(float)));
        // cudaCheck(cudaMalloc(&d_bias, OC * sizeof(float)));
        // cudaCheck(cudaMemcpy(d_inp, inp, B * T * C * sizeof(float), cudaMemcpyHostToDevice));
        // cudaCheck(cudaMemcpy(d_weight, weight, C * OC * sizeof(float), cudaMemcpyHostToDevice));
        // cudaCheck(cudaMemcpy(d_bias, bias, OC * sizeof(float), cudaMemcpyHostToDevice));

        // read kernel_num from command line
        // int kernel_num = 1;
        // if (argc > 1) {
        //     kernel_num = atoi(argv[1]);
        // }
        // printf("Using kernel %d\n", kernel_num);

        // first check the correctness of the kernel

        matmul_forward_cpu_0(h_Out, h_In, h_Weight, h_Bias, B, T, C, OC);
        validate_result(h_Out, h_Out, "matmul_forward_cpu_0", B * T * OC, 1e-7f);

        for (int i = 0; i < B * T * OC; i++) { d_Out[i] = randf(&seed); }
        matmul_forward_cpu_1(d_Out, d_In, d_Weight, d_Bias, B, T, C, OC);
        validate_result(d_Out, h_Out, "matmul_forward_cpu_1", B * T * OC, 1e-7f);

        for (int i = 0; i < B * T * OC; i++) { d_Out[i] = randf(&seed); }
        matmul_forward_cpu_2(d_Out, d_In, d_Weight, d_Bias, B, T, C, OC);
        validate_result(d_Out, h_Out, "matmul_forward_cpu_2", B * T * OC, 1e-7f);

        // // time the kernel at different block sizes
        // int sqrt_block_sizes[] = { 4, 8, 16, 32 };
        // 
        // for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        //     int sqrt_block_size = sqrt_block_sizes[j];
        //     printf("Checking block size %d x %d.\n", sqrt_block_size, sqrt_block_size);
        //     matmul_forward(kernel_num, d_out, d_inp, d_weight, d_bias, B, T, C, OC, sqrt_block_size);
        //     validate_result(d_out, out, "out", B * T * OC, 1e-1f);
        // }
        // 
        // printf("All results match. Starting benchmarks.\n\n");
        // 
        // for (int j = 0; j < sizeof(sqrt_block_sizes) / sizeof(int); j++) {
        //     int sqrt_block_size = sqrt_block_sizes[j];
        // 
        //     int repeat_times = 100;
        //     float elapsed_time = benchmark_kernel(repeat_times, matmul_forward,
        //                                           kernel_num, d_out, d_inp, d_weight, d_bias,
        //                                           B, T, C, OC, sqrt_block_size);
        // 
        //     // napkin math: estimate the flops achieved
        //     // e.g. A100 40GB PCIe is advertised at 19.5 TFLOPS fp32
        //     float tflops = (float)B * T * C * OC * 2 / elapsed_time * 1e3f / 1e12f;
        //     printf("sqrt_block_size %4d | time %.4f ms | tflops %.2f\n", sqrt_block_size, elapsed_time, tflops);
        // }

        // free memory
        free(h_Out);
        free(h_In);
        free(h_Weight);
        free(h_Bias);

        free(d_Out);
        free(d_In);
        free(d_Weight);
        free(d_Bias);

        // cudaCheck(cudaFree(cublaslt_workspace));
        // cublasCheck(cublasDestroy(cublas_handle));
        // cublasCheck(cublasLtDestroy(cublaslt_handle));
        Console.ReadKey();
        return 0;
    }
    
#endif
}

    /*
    // ----------------------------------------------------------------------------
    // GPU kernels

    // kernel 1: naive kernel, every thread handles one output element, direct global memory access
    __global__ void matmul_forward_kernel1(float* out,
                                           const float* inp, const float* weight, const float* bias,
                                           int BT, int C, int OC) {
        // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        // in the naive kernel, every thread handles one element of out
        int bt = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y * blockDim.y + threadIdx.y;
    if (bt < BT && oc < OC) {
        int b = bt / BT;
        int t = bt % BT;
        float val = (bias != NULL) ? bias[oc] : 0.0f;
        const float* wrow = weight + oc * C;
        const float* inp_bt = inp + b * BT * C + t * C;
        for (int i = 0; i < C; i++) {
            val += inp_bt[i] * wrow[i];
        }
            out[bt * OC + oc] = val;
    }
    }

    // is there no better way other than just adding bias with a whole separate kernel?
    // this is a highly memory-bound operation, should be fused into the matmul kernel
    // but i can't seem to find a cuBLAS function that does this
    __global__ void add_bias(float* out, const float* bias, int B, int T, int OC) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (int i = idx; i < B * T * OC; i += stride) {
        int col = i % OC;
            out[i] += bias[col];
    }
    }

    // ----------------------------------------------------------------------------
    // kernel launcher

    // kernel 1 is the most naive matmul kernel
    void matmul_forward1(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC,
                         const int sqrt_block_size) {
        // out is (B,T,OC). OC is short for "output channels", e.g. OC = 4 * C
        // inp is (B,T,C), weight is (OC, C), bias is (OC)
        dim3 gridDim(ceil_div(B* T, sqrt_block_size), ceil_div(OC, sqrt_block_size));
        dim3 blockDim(sqrt_block_size, sqrt_block_size);
        matmul_forward_kernel1 <<< gridDim, blockDim >>> (out, inp, weight, bias, B* T, C, OC);
        cudaCheck(cudaGetLastError());
    }

    // kernel 2 calls cuBLAS, which should be very efficient
    void matmul_forward2(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC,
                         const int sqrt_block_size) {
        // for reference API is:
        // cublasStatus_t cublasSgemm(cublasHandle_t handle,
        //                        cublasOperation_t transa, cublasOperation_t transb,
        //                        int m, int n, int k,
        //                        const float           *alpha,
        //                        const float           *A, int lda,
        //                        const float           *B, int ldb,
        //                        const float           *beta,
        //                        float           *C, int ldc)
        // for us, inp is (B*T, C), weight is (OC, C), out is (B*T, OC)
        // cuBLAS does C = alpha * A * B + beta * C
        // where A is mxk, B is kxn, C is mxn
        // now, because we use row-major storage, cuBLAS (which is column-major) sees our matrices transposed.
        // algorithmically / in e.g. PyTorch we want to do: out = inp @ weight.T
        // but because cuBLAS is column-major, we actually want to get it to calculate out.T . Mathematically, this is:
        // out.T = weight @ inp.T
        // but again, our variables look transposed, so using the actual weight/inp we have here in this function, this becomes
        // out.T = weight.T @ inp
        // so we need to get cuBLAS to calculate weight.T @ inp (the variables here are the actual ones in this function)
        // => need to call cuBLAS with A = weight, B = inp
        // => need to call cuBLAS with transa = CUBLAS_OP_T, transb = CUBLAS_OP_N

        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasCheck(cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, OC, B * T, C, &alpha, weight, C, inp, C, &beta, out, OC));
        // and now we still have to add the bias... (ew)
        if (bias != NULL) {
            int block_size = sqrt_block_size * sqrt_block_size;
            int grid_size = ceil_div(OC * B * T, block_size);
            add_bias <<< grid_size, block_size >>> (out, bias, B, T, OC);
            cudaCheck(cudaGetLastError());
        }
    }

    // uses cublasLt to fuse the bias and gelu
    // https://docs.nvidia.com/cuda/cublas/#cublasltmatmul
    // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuBLASLt/LtSgemm/sample_cublasLt_LtSgemm.cu
    void matmul_forward3(float* out,
                         const float* inp, const float* weight, const float* bias,
                         int B, int T, int C, int OC) {
        int has_bias = (bias != NULL);
        int has_gelu = 0;

        // check bias alignment
        if (((uintptr_t)bias % 16) != 0) {
            printf("Bias pointer is not aligned (cuBLASLt requirement)!\n");
            exit(EXIT_FAILURE);
        }

        int returnedResults = 0;
        cublasLtMatmulDesc_t operationDesc;
        cublasLtMatmulPreference_t preference;
        cublasLtMatrixLayout_t weightLayout;
        cublasLtMatrixLayout_t inputLayout;
        cublasLtMatrixLayout_t outputLayout;
        cublasLtMatrixLayout_t biasLayout;
        cublasLtMatmulHeuristicResult_t heuristic;

        // create the operation descriptor
        cublasOperation_t opNoTranspose = CUBLAS_OP_N;
        cublasOperation_t opTranspose = CUBLAS_OP_T;
        cublasLtEpilogue_t epilogueBias = CUBLASLT_EPILOGUE_DEFAULT;
        if (has_bias && has_gelu) {
            epilogueBias = CUBLASLT_EPILOGUE_GELU_BIAS;
        } else if (has_bias) {
            epilogueBias = CUBLASLT_EPILOGUE_BIAS;
        } else if (has_gelu) {
            epilogueBias = CUBLASLT_EPILOGUE_GELU;
        }
        cublasCheck(cublasLtMatmulDescCreate(&operationDesc, cublas_compute_type, CUDA_R_32F));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opTranspose, sizeof(opTranspose)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opNoTranspose, sizeof(opNoTranspose)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogueBias, sizeof(epilogueBias)));
        cublasCheck(cublasLtMatmulDescSetAttribute(operationDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));

        // define matrix layouts
        cublasCheck(cublasLtMatrixLayoutCreate(&weightLayout, CUDA_R_32F, C, OC, C));
        cublasCheck(cublasLtMatrixLayoutCreate(&inputLayout, CUDA_R_32F, C, B * T, C));
        cublasCheck(cublasLtMatrixLayoutCreate(&outputLayout, CUDA_R_32F, OC, B * T, OC));
        cublasCheck(cublasLtMatrixLayoutCreate(&biasLayout, CUDA_R_32F, OC, 1, OC));

        // create a preference handle with specified max workspace
        cublasCheck(cublasLtMatmulPreferenceCreate(&preference));
        cublasCheck(cublasLtMatmulPreferenceSetAttribute(preference,
            CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            &cublaslt_workspace_size, sizeof(cublaslt_workspace_size)));

        // find a suitable algorithm
        cublasCheck(cublasLtMatmulAlgoGetHeuristic(cublaslt_handle, operationDesc,
            weightLayout, inputLayout, outputLayout, outputLayout,
            preference, 1, &heuristic, &returnedResults));
        if (returnedResults == 0) {
            printf("No cuBLASLt algorithm: B: %d, T: %d, C: %d, OC: %d, bias: %d, gelu: %d\n",
                B, T, C, OC, has_bias, has_gelu);
            exit(EXIT_FAILURE);
        }

        // call the matmul
        const float alpha = 1.0f, beta = 0.0f;
        cublasCheck(cublasLtMatmul(cublaslt_handle, operationDesc,
            &alpha, weight, weightLayout, inp, inputLayout, &beta,
            out, outputLayout, out, outputLayout, &heuristic.algo,
            cublaslt_workspace, cublaslt_workspace_size, 0));

        // cleanups
        cublasCheck(cublasLtMatmulPreferenceDestroy(preference));
        cublasCheck(cublasLtMatmulDescDestroy(operationDesc));
        cublasCheck(cublasLtMatrixLayoutDestroy(weightLayout));
        cublasCheck(cublasLtMatrixLayoutDestroy(inputLayout));
        cublasCheck(cublasLtMatrixLayoutDestroy(outputLayout));
        cublasCheck(cublasLtMatrixLayoutDestroy(biasLayout));
    }

    // kernel version dispatch
    void matmul_forward(int kernel_num,
                        float* out,
                        const float* inp, const float* weight, const float* bias,
                        int B, int T, int C, int OC,
                        const int sqrt_block_size) {
        switch (kernel_num) {
            case 1:
                matmul_forward1(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
                break;
            case 2:
                matmul_forward2(out, inp, weight, bias, B, T, C, OC, sqrt_block_size);
                break;
            case 3:
                matmul_forward3(out, inp, weight, bias, B, T, C, OC);
                break;
            default:
                printf("Invalid kernel number\n");
                exit(1);
        }
    }
    */

