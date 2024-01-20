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

    int M1[K][M][N]; //... Array of 1st matrix
    int M2[K][N][P]; //... Array of 2nd matrix    
    int M3[K][M][P]; //... Array of ans matrix   

    if (!world_rank) //... Rank 0 process will create the matrices
    {
        for (int k = 0; k < K; k++)
        {
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < N; c++)
                {
                    M1[k][r][c] = rand() % 10;
                }
            }
            for (int r = 0; r < N; r++)
            {
                for (int c = 0; c < P; c++)
                {
                    M2[k][r][c] = rand() % 10;
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int size_per_process = K / world_size;
    int m1[size_per_process][M][N]; //... Local array of 1st matrix
    int m2[size_per_process][N][P]; //... Local array of 2nd matrix
    int m3[size_per_process][M][P]; //... Local array of ans matrix

    MPI_Scatter(M1, size_per_process * M * N, MPI_INT, m1, size_per_process * M * N, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(M2, size_per_process * N * P, MPI_INT, m2, size_per_process * N * P, MPI_INT, 0, MPI_COMM_WORLD);

    //... Performing matrix multiplication 
    for (int n = 0; n < size_per_process; n++)
    {
        for (int r = 0; r < M; r++)
        {
            for (int c = 0; c < P; c++)
            {
                m3[n][r][c] = 0;
                for (int k = 0; k < N; k++)
                {
                    m3[n][r][c] += m1[n][r][k] * m2[n][k][c];
                }
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Gather(m3, size_per_process * M * P, MPI_INT, M3, size_per_process * M * P, MPI_INT, 0, MPI_COMM_WORLD);

    if (!world_rank) //... Rank 0 process will output the result
    {
        for (int k = 0; k < K; k++)
        {
            cout << "Result " << k << ":\n";
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < P; c++)
                {
                    cout << M3[k][r][c] << " ";
                }
                cout << "\n";
            }
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d took %f seconds.\n", world_rank, MPI_Wtime() - start_time);

    MPI_Finalize();

    return 0;
}
