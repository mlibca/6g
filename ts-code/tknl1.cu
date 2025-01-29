
#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>

#include <cufft.h>
#include <stdlib.h>
#include <iostream>

// CUDA kernel function (device code)
__global__ void cudakernel(int *a) {
    int idx = threadIdx.x;
    a[idx] = idx;
   

}

int main() {
    const int N = 2560;
    int h_a[N];

    // Allocate memory on the device
    int *d_a;
    cudaMalloc((void **)&d_a, N * sizeof(int));

    // Launch the kernel
    cudakernel<<<1, N>>>(d_a);

    // Copy result from device to host
    cudaMemcpy(h_a, d_a, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Print the result
    for (int i = 0; i < N; i++) {
        std::cout << h_a[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    cudaFree(d_a);

    return 0;
}
