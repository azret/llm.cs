extern "C" __global__ void VecAdd_kernel(
    const float *A, const float *B, float *C, int N) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
        C[i] = A[i] + B[i];
  }
}