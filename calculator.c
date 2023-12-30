#include <stdio.h>
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int number1 = 147, number2 = 93;

    if (rank == 1)
    {
        printf("I'm Rank %d & addition = %d", rank, number1 + number2);
    }
    else if (rank == 2)
    {
        printf("I'm Rank %d & subtraction = %d", rank, number1 - number2);
    }
    else if (rank == 3)
    {
        printf("I'm Rank %d & multiplication = %d", rank, number1 * number2);
    }
    else if (rank == 4)
    {
        printf("I'm Rank %d & division = %d", rank, number1 / number2);
    }
    else printf("I'm Rank %d & I've nothing to do!", rank);

    MPI_Finalize();
    return 0;
}
