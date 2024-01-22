#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

struct Contact
{
    char name[256];
    char phone_number[256];
};

__device__ bool check(char* name, char *search_name, int length)
{
    for (int i = 0; i < length; i++)
    {
        if (name [i] != search_name[i])
        {
            return false;
        }
    }
    return true;
}

__global__ void searchPhonebook(Contact* phonebook, int size, char* search_name, int length)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < size)
    {
        if (check(phonebook[index].name, search_name, length))
        {
            printf("Name: %s, Phone: %s\n", phonebook[index].name, phonebook[index].phone_number);
        }
    }
}

int main()
{
    vector<string> file_names = {"phonebook1.txt", "phonebook2.txt"};
    vector<Contact> phonebook;

    for (auto file_name: file_names)
    {
        ifstream file(file_name);
        Contact contact;
        while (file >> contact.name >> contact.phone_number)
        {
            phonebook.push_back(contact);
        }
        file.close();
    }

    int size = phonebook.size();

    Contact* device_phonebook;
    cudaMalloc((void **)&device_phonebook, sizeof(Contact) * size);

    cudaMemcpy(device_phonebook, phonebook.data(), sizeof(Contact) * size, cudaMemcpyHostToDevice);

    string search_name = "Sophie";
    int name_length = search_name.size() + 1;
    char* device_search_name;
    cudaMalloc((void**)&device_search_name, name_length);

    cudaMemcpy(device_search_name, search_name.c_str(), name_length, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize = (size + blockSize - 1) / blockSize;

    searchPhonebook<<<gridSize, blockSize>>>(device_phonebook, size, device_search_name, name_length);

    cudaDeviceSynchronize();

    cudaFree(device_phonebook);
    cudaFree(device_search_name);
    
    return 0;
}
