#include <stdio.h>
#include <string.h>
#include <mpi.h>

#define WORD_LENGTH 100

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

    if (rank == 0)
    {
        char word[WORD_LENGTH];
        scanf("%s", word);

        //... Send the word to the receiver
        MPI_Ssend(word, WORD_LENGTH, MPI_CHAR, 1, 0, MPI_COMM_WORLD);

        //... Receive the word from the sender
        MPI_Recv(word, WORD_LENGTH, MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Sender (Rank #0) received modified word: %s\n", word);
    }
    else if (rank == 1)
    {
        char received_word[WORD_LENGTH];

        //... Receive the word from the sender
        MPI_Recv(received_word, WORD_LENGTH, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Receiver (Rank #1) received: %s\n", received_word);

        //... Toggle each letter of the word
        for (int i = 0; i < strlen(received_word); i++) 
        {
            received_word[i] ^= 32;
        }

        //... Send the modified word back to the sender
        MPI_Ssend(received_word, strlen(received_word) + 1, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
