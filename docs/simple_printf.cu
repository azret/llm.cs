extern "C" __global__ void simple_printf(int val) {
    printf("[%d, %d]:\t\tValue is:%d\n", blockIdx.y * gridDim.x + blockIdx.x,
        threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
        threadIdx.x,
        val);
}

extern "C" __global__ void simple_printf_2() {
    printf("[%d, %d, %d]\n",
        threadIdx.x,
        threadIdx.y,
        threadIdx.z);
}
