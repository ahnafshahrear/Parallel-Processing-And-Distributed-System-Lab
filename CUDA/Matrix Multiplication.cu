#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

const int K = 9; //... Total number of matrix
const int M = 5; //... Row of 1st matrix
const int N = 3; //... Column of 1st matrix & row of 2nd matrix
const int P = 4; //... Column of 2nd matrix

__global__ void multiplyMatrix(int *x, int *y, int *z, int M, int N, int P)
{
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;

    if (r < M and c < P)
    {
        int ans = 0;
        for (int k = 0; k < N; k++)
        {
            ans += x[r * N + k] * y[k * P + c];
        }
        z[r * P + c] = ans;
    }
}

void generateRandomMatrix(int *matrix, int M, int N)
{
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            matrix[r * N + c] = rand() % 10;
        }
    }
}

void printMatrix(int *matrix, int M, int N, int id)
{
    cout << "Result " << id << " :\n";
    for (int r = 0; r < M; r++)
    {
        for (int c = 0; c < N; c++)
        {
            cout << matrix[r * N + c] << "\t";
        }
        cout << "\n";
    }
    cout << "\n";
}

int main()
{
    int *X; //... Host 1st matrix
    int *Y; //... Host 2nd matrix
    int *Z; //... Host ans matrix

    //... Host memoty allocation
    X = (int *)malloc(K * M * N * sizeof(int));
    Y = (int *)malloc(K * N * P * sizeof(int));
    Z = (int *)malloc(K * M * P * sizeof(int));

    srand(time(nullptr));

    //... 1st & 2nd matrix generation
    for (int k = 0; k < K; k++)
    {
        generateRandomMatrix(X + k * M * N, M, N);
        generateRandomMatrix(Y + k * N * P, N, P);
    }

    int *x; //... Device 1st matrix
    int *y; //... Device 2nd matrix
    int *z; //... Device ans matrix

    //... Device memory allocation
    cudaMalloc((void **)&x, K * M * N * sizeof(int));
    cudaMalloc((void **)&y, K * N * P * sizeof(int));
    cudaMalloc((void **)&z, K * M * P * sizeof(int));

    cudaEvent_t start; //... Start time
    cudaEventCreate(&start); 

    cudaEvent_t end; //... End time
    cudaEventCreate(&end);

    //... Copy data from host to device
    cudaMemcpy(x, X, K * M * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(y, Y, K * N * P * sizeof(int), cudaMemcpyHostToDevice);

    //... Define grid dimension & block dimension of thread
    dim3 blockDim(16, 16);
    dim3 gridDim((P + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    cudaEventRecord(start);

    //... Launch the matrix multiplication kernel for each pair of matrices
    for (int k = 0; k < K; k++)
    {
        multiplyMatrix<<<gridDim, blockDim>>>(x + k * M * N, y + k * N * P, z + k * M * P, M, N, P);
    }

    cudaEventRecord(end);
    cudaEventSynchronize(end);

    //... Copy the result back to the host
    cudaMemcpy(Z, z, K * M * P * sizeof(int), cudaMemcpyDeviceToHost);

    float time_taken = 0;
    cudaEventElapsedTime(&time_taken, start, end);
    cout << "Time taken to execute: " << time_taken << " ms\n\n";

    //... Print the result
    for (int k = 0; k < K; k++)
    {
        printMatrix(Z + k * M * P, M, P, k);
    }

    //... Free device and host memory
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    free(X);
    free(Y);
    free(Z);

    return 0;
}
