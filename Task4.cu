#include <iostream>
#include <cuda_runtime.h>

#define TILE_SIZE 16 // Define the size of the shared memory tile

// CUDA kernel for optimized matrix multiplication
__global__ void optimizedMatrixMul(float* A, float* B, float* C, int N) {
    // Shared memory for tiles of A and B
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    // Calculate row and column indices of the element to compute
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    float value = 0.0f;

    // Loop over tiles of A and B in steps of TILE_SIZE
    for (int t = 0; t < (N + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load elements of A and B into shared memory
        if (row < N && t * TILE_SIZE + threadIdx.x < N) {
            tileA[threadIdx.y][threadIdx.x] = A[row * N + t * TILE_SIZE + threadIdx.x];
        }
        else {
            tileA[threadIdx.y][threadIdx.x] = 0.0f;
        }

        if (col < N && t * TILE_SIZE + threadIdx.y < N) {
            tileB[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        }
        else {
            tileB[threadIdx.y][threadIdx.x] = 0.0f;
        }

        __syncthreads(); // Synchronize threads to ensure tiles are fully loaded

        // Perform partial computation for the tile using loop unrolling
#pragma unroll
        for (int i = 0; i < TILE_SIZE; i++) {
            value += tileA[threadIdx.y][i] * tileB[i][threadIdx.x];
        }

        __syncthreads(); // Synchronize threads before loading the next tile
    }

    // Write the computed value to the output matrix
    if (row < N && col < N) {
        C[row * N + col] = value;
    }
}

// Host function
int main() {
    const int N = 64; // Matrix size (N x N)
    const int SIZE = N * N * sizeof(float);

    // Host memory allocation
    float* h_A = new float[N * N];
    float* h_B = new float[N * N];
    float* h_C = new float[N * N];

    // Initialize matrices A and B
    for (int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f; // Example value
        h_B[i] = 2.0f; // Example value
    }

    // Device memory allocation
    float* d_A, * d_B, * d_C;
    cudaMalloc((void**)&d_A, SIZE);
    cudaMalloc((void**)&d_B, SIZE);
    cudaMalloc((void**)&d_C, SIZE);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, SIZE, cudaMemcpyHostToDevice);

    // Define thread block and grid dimensions
    dim3 blockDim(TILE_SIZE, TILE_SIZE);
    dim3 gridDim((N + TILE_SIZE - 1) / TILE_SIZE, (N + TILE_SIZE - 1) / TILE_SIZE);

    // Launch the kernel
    optimizedMatrixMul << <gridDim, blockDim >> > (d_A, d_B, d_C, N);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, SIZE, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Result matrix C:" << std::endl;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            std::cout << h_C[i * N + j] << " ";
        }
        std::cout << std::endl;
    }

    // Free memory
    delete[] h_A;
    delete[] h_B;
    delete[] h_C;
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    return 0;
}
