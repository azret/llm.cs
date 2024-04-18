// nvcc --keep -O3 --use_fast_math -Xcompiler -fopenmp dev\cuda\vectorAdd_kernel64.cu -o .\obj\vectorAdd_kernel64

//  -lcublas -lcublasLt

// Device code
extern "C" __global__ void VecAdd_kernel(
    const float *A, const float *B, float *C, int N) {

  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N) {
        C[i] = A[i] + B[i];
  }
}