#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

//... To compile: mpic++ matrix-multiplication.cpp -o matrix-multiplication
//... To run: mpirun -n 3 ./matrix-multiplication  

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double start_time = MPI_Wtime();

    srand(time(nullptr));

    const int K = 9; //... Total Number of matrix
    const int M = 4; //... Row of 1st matrix
    const int N = 3; //... Column of 1st matrix & Row of 2nd matrix
    const int P = 5; //... Column of 2nd matrix

    if (K % world_size != 0)
    {
        cout << "Number of matrices should be divisible by number of Process\n";
        MPI_Finalize();
        return 0;
    }

    int X[K][M][N]; //... Array of 1st matrix
    int Y[K][N][P]; //... Array of 2nd matrix    
    int Z[K][M][P]; //... Array of ans matrix   

    if (!world_rank) //... Rank 0 process will create the matrices
    {
        for (int k = 0; k < K; k++)
        {
            //... Generating 1st matrix with random numbers
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    X[k][r][c] = rand() % 100;
                }
            }
            //... Generating 2nd matrix with random numbers
            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < P; c++)
                {
                    Y[k][r][c] = rand() % 100;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int size_per_process = K / world_size;
    int x[size_per_process][M][N]; //... Local array of 1st matrix
    int y[size_per_process][N][P]; //... Local array of 2nd matrix
    int z[size_per_process][M][P]; //... Local array of ans matrix

    MPI_Scatter(X, size_per_process * M * N, MPI_INT, x, size_per_process * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(Y, size_per_process * N * P, MPI_INT, y, size_per_process * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    //... Performing matrix multiplication by each process
    for (int n = 0; n < size_per_process; n++)
    {
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < P; c++)
            {
                z[n][r][c] = 0;
                for (int k = 0; k < N; k++)
                {
                    z[n][r][c] += x[n][r][k] * y[n][k][c];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(z, size_per_process * M * P, MPI_INT, Z, size_per_process * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    if (!world_rank) //... Rank 0 process will output the results
    {
        for (int k = 0; k < K; k++)
        {
            cout << "Result of the matrix multiplication " << k + 1 << ":\n\n";
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < P; c++)
                {
                    cout << Z[k][r][c] << " ";
                }
                cout << "\n";
            }
            cout << "\n";
        }
    }

    double finish_time = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d took %f seconds.\n", world_rank, finish_time - start_time);

    MPI_Finalize();

    return 0;
}
