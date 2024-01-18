#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

//... To compile: mpic++ matrix-multiplication.cpp -o matrix-multiplication
//... To run: mpirun -n 4 ./matrix-multiplication  

void generate_matrix(int row, int column, vector<vector<int>> &x)
{
    x.resize(row, vector<int>(column));
    for (int r = 0; r < row; r++)
    {
        for (int c = 0; c < column; c++)
        {
            x[r][c] = rand() % 10;
        }
    }
}

void multiply_matrix(vector<vector<int>> &x, vector<vector<int>> &y, vector<vector<int>> &z)
{
    int m = x.size(), n = x[0].size(), p = y[0].size();
    z.resize(m, vector<int>(p, 0));

    for (int r = 0; r < m; r++)
    {
        for (int c = 0; c < p; c++)
        {
            for (int k = 0; k < n; k++)
            {
                z[r][c] += x[r][k] * y[k][c];
            }
        }
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    srand(time(nullptr));

    const int K = 9; 
    const int M = 4;  
    const int N = 3; 
    const int P = 5;

    vector<vector<int>> x, y, z;

    double start_time, end_time;

    for (int k = 0; k < K; k++)
    {
        if (k % world_size == world_rank)
        {
            start_time = MPI_Wtime();
            generate_matrix(M, N, x);
            generate_matrix(N, P, y);
            multiply_matrix(x, y, z);

            MPI_Barrier(MPI_COMM_WORLD);

            cout << "Process " << world_rank << " & Result:" << "\n";
            for (int r = 0; r < M; r++)
            {
                for (int c = 0; c < P; c++)
                {
                    cout << z[r][c] << " ";
                }
                cout << "\n";
            }

            end_time = MPI_Wtime();
            cout << "Process " << world_rank << " took " << end_time - start_time << " seconds\n";
        }
        else MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();

    return 0;
}
