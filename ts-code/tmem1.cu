#include <iostream>
#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>


#include <cuda_runtime.h>



#include <unistd.h>
#include <termios.h>
#include <fcntl.h>


#define N 1024*5  // Size of the matrices

#define M  N*N/2  // cycles to monitor 


#define og 1024ULL * 1024 * 1024 

    clock_t start, end;
    double cpu_time_used;

  
   
    size_t  g_size; 






__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
	
	int idx = row*n + col ; 
    
    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
		
		
		if( (row %1024)  == 2  )
		    printf("sum is : %f   | ", sum  ); 
    }
}




__global__ void packetKernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
		
		
		if( (row %10) == 2  )
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
	
	// for testing purpose only  
	size = g_size; 

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


void pauseit()
{
    printf("Press any key to continue...\n");
    getchar(); // Wait for a key press
    printf("Key pressed. Continuing execution...\n");


}


int kbhit(void) {
    struct termios oldt, newt;
    int ch;
    int oldf;

    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF) {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}






int main() {
    float *A, *B, *C;
    size_t size = N * N * sizeof(float);


    // 
	
	//size = size *4*5; 
	
	g_size = size; 

    A = (float*)malloc(g_size);
    B = (float*)malloc(g_size);
    C = (float*)malloc(g_size);

   // std::cout << "size :  " << g_size  << std::endl;
   printf("size(g) : %f \n", (float) g_size/ ((float) og) );

   // step 1 
	start = clock();

    for (long int i = 0; i < N * N; ++i)
   
 //  for (long int i = 0; i < g_size/sizeof(float); ++i)
	{
        A[i] = static_cast<float>(rand()) / RAND_MAX;
        B[i] = static_cast<float>(rand()) / RAND_MAX;		
    }

    printf("Press any key to stop the loop...\n");

    while ( 1 )
	{
	
	if ( kbhit()) { printf("Key pressed. Exiting loop...\n"); break; }
	
#if 0 	
	 for (long int i = 0; i < N * N; ++i)
	  {
	   A[i] = B[i]*  (rand()) / RAND_MAX;
	  
	   B[i] = A[i]*  (rand()) / RAND_MAX; 
      }	  
#endif 	

	}




    end = clock();
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken for init data : %f seconds\n", cpu_time_used);

    pauseit(); 
    
    //return 0 ; 
	
	
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
