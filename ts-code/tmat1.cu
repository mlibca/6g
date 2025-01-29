#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>


#include <cuda_runtime.h>



#define N 1024*100  // Size of the matrices

#define M  N*N/2  // cycles to monitor 


    clock_t start, end;
    double cpu_time_used;

  
   
   



__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
		
		
		if( row  == 101  )
		    printf("sum is : %f   | ", sum  ); 
    }
}





int tt() {
    clock_t start, end;
    double cpu_time_used;

    start = clock();
   
    end = clock();

    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken: %f seconds\n", cpu_time_used);

    return 0;
}



void matrixMul(float* A, float* B, float* C, int n) {
    float *d_A, *d_B, *d_C;
    size_t size = n * n * sizeof(float);

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);


    start = clock();
  
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
	
    end = clock();
  
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken for loadng data to device : %f seconds\n", cpu_time_used);


	
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

   start = clock();
   
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
	   end = clock();
  
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken for mat calc : %f seconds\n", cpu_time_used);


 start = clock();
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);
	 end = clock();
  
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken for load data from device to host : %f seconds\n", cpu_time_used);


    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main() {
    float *A, *B, *C;
    size_t size = N * N * sizeof(float);

    A = (float*)malloc(size);
    B = (float*)malloc(size);
    C = (float*)malloc(size);

    std::cout << "size :  " << size  << std::endl;

   // step 1 
	start = clock();

    for (long int i = 0; i < N * N; ++i) {
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;
		
		
		
    }

    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for init data : %f seconds\n", cpu_time_used);


   // step 2 

    matrixMul(A, B, C, N);

    std::cout << "Matrix multiplication completed." << std::endl;
	
	 start = clock();
	
	// check some rows 
    for (int i = 0; i < N * N; ++i) 
	 {
	  if( (i ==1024 ) ) 
	    printf(" C[%d]= %f  | ", i, C[i] ); 
     }

     end = clock();
  
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;

    printf("Time taken for loop data : %f seconds\n", cpu_time_used);


    free(A);
    free(B);
    free(C);

    return 0;
}
