#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define N 100000 //... Size of the array

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2)
    {
        if (rank == 0)
        {
            printf("This program requires exactly 2 process\n");
        }
        MPI_Finalize();
        return 0;
    }

    int *array = NULL;
    if (rank == 0)
    {
        array = (int *)malloc(N * sizeof(int));
        for (int i = 0; i < N; i++)
        {
            array[i] = i;
        }
    }

    double start_time = MPI_Wtime();

    //... Scatter the array to both processes
    int local_size = N / 2;
    int *local_array = (int *)malloc(local_size * sizeof(int));
    MPI_Scatter(array, local_size, MPI_INT, local_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    //... Perform local sum
    int local_sum = 0;
    for (int i = 0; i < local_size; i++)
    {
        local_sum += local_array[i];
    }

    //... Reduce the local sums to get the global sum
    int global_sum;
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();

    if (rank == 0)
    {
        printf("Global sum: %d\n", global_sum);
        printf("Time taken by process #0: %f seconds\n", end_time - start_time);
    }

    if (rank == 0) 
    {
        free(array);
    }
    free(local_array);
    
    MPI_Finalize();
    return 0;
}
