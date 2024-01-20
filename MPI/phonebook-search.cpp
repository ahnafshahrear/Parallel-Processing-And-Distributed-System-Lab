#include <bits/stdc++.h>
#include <mpi.h>

using namespace std;

//... To compile: mpic++ phonebook-search.cpp -o phonebook-search
//... To run: mpirun -n 4 ./phonebook-search phonebook1.txt phonebook1.txt

void send_int(int number, int receiver)
{
    MPI_Send(&number, 1, MPI_INT, receiver, 1, MPI_COMM_WORLD);
}

int receive_int(int sender)
{
    int number;
    MPI_Recv(&number, 1, MPI_INT, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return number;
}

void send_string(string text, int receiver)
{
    int length = (int)text.size() + 1;
    send_int(length, receiver);
    MPI_Send(&text[0], length, MPI_CHAR, receiver, 1, MPI_COMM_WORLD);
}

string receive_string(int sender)
{
    int length = receive_int(sender);
    char *text = new char[length];
    MPI_Recv(text, length, MPI_CHAR, sender, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    return string(text);
}

string vector_to_string(vector<string> &words, int start, int end)
{
    string text = "";
    for (int i = start; i < min((int)words.size(), end); i++)
    {
        text += words[i] + "\n";
    }
    return text;
}

vector<string> string_to_vector(string text)
{
    stringstream x(text);
    vector<string> words;
    string word;
    while (x >> word)
    {
        words.push_back(word);
    }
    return words;
}

void check(string name, string phone, string search_name, int rank)
{
    if (name.size() != search_name.size()) 
    {
        return;
    }
    for (int i = 0; i < search_name.size(); i++)
    {
        if (name[i] != search_name[i])
        {
            return;
        }
    }
    printf("%s %s found by process %d.\n", name.c_str(), phone.c_str(), rank);
}

void read_phonebook(vector<string> &file_names, vector<string> &names, vector<string> &phone_numbers)
{
    for (auto file_name: file_names)
    {
        ifstream file(file_name);
        string name, number;
        while (file >> name >> number)
        {
            names.push_back(name);
            phone_numbers.push_back(number);
        }
        file.close();
    }
}

int main(int argc, char** argv)
{
    MPI_Init(&argc, &argv);

    int world_size, world_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    double start_time = MPI_Wtime();

    if (!world_rank)
    {
        vector<string> names, phone_numbers;
        vector<string> file_names(argv + 1, argv + argc);
        read_phonebook(file_names, names, phone_numbers);
        int segment = names.size() / world_size + 1;

        for (int i = 1; i < world_size; i++)
        {
            int start = i * segment, end = start + segment;
            string names_string = vector_to_string(names, start, end);
            send_string(names_string, i);
            string phone_numbers_string = vector_to_string(phone_numbers, start, end);
            send_string(phone_numbers_string, i);
        }

        string name = "Henry";
        for (int i = 0; i < segment; i++)
        {
            check(names[i], phone_numbers[i], name, world_rank);
        }
    }
    else 
    {
        string names_string = receive_string(0);
        vector<string> names = string_to_vector(names_string);
        string phone_numbers_string = receive_string(0);
        vector<string> phone_numbers = string_to_vector(phone_numbers_string);

        string name = "Jackson";
        for (int i = 0; i < names.size(); i++)
        {
            check(names[i], phone_numbers[i], name, world_rank);
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);

    printf("Process %d took %f seconds.\n", world_rank, MPI_Wtime() - start_time);

    MPI_Finalize();

    return 0;
}
