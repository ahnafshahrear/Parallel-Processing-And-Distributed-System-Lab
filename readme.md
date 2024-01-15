## Parallel Processing & Distributed System Lab
- **[Ahnaf Shahrear Khan](https://github.com/ahnafshahrear)**
- **Computer Science & Engineering, University of Rajshahi**
- **Code:** `CSE4112`

### Installation Guide (MPI)
- **To install MPI, see this [video](https://www.youtube.com/watch?v=bkfCrj-rBjU) & follow every step carefully. If you face the issue "sal.h dependency not found" or something like that, uninstall the MinGW compiler & follow this [video](https://www.youtube.com/watch?v=_-O94qsnOLk)**
### MPI Installation Through "Windows Subsystem for Linux"
- **Go to `Turn Windows features on or off` & turn on the `Windows Subsystem for Linux`**


### Run Command 
- **First go to `Terminal` > `Run Build Task...` in the VS Code & then run the following command**
<pre>
<b>mpiexec -n number_of_processors file_name_without_extension</b>
</pre>

### Lab Tasks
- **Write an MPI program to multiply two matrices of size MxN and NxP**
- **Write an MPI program to simulate a simple calculator. Perform each operation using a different process in parallel ✓**
- **Write an MPI program to count the words in a file & sort it in descending order of frequency of words that is, the highest occurring word must come first & the least occurring word must come last**
- **Write a nMPI program using synchronous send. The sender process sends a word to the receiver. The second process receives the word, toggles each letter of the word and sends it back to the first process. Both processes use synchronous send operations ✓**
- **Write an MPI program to add an array of size N using two processes. Print the result in the root process. Investigate the amount of time taken by each process ✓**
- **Write a Cuda program for matrix multiplication**
- **Write a Cuda program to find out the maximum common subsequence**
- **Given a paragraph & a pattern like %x%. Now write a Cuda program to find out the line number where %x% this pattern exists in the given paragraph**

### Basic Instructions of MPI
- **To configure the program to run an MPI, initialize all the data structures -**
<pre>
<b>int MPI_Init(int *argc, char ***argv)</b>
</pre>
- **To stop the process & turn of any communication -**
<pre>
<b>int MPI_Finalize()</b>
</pre>
- **To get the number of processes -**
<pre>
<b>int MPI_Comm_size(MPI_Comm comm, int *size)</b>
</pre>
- **To get the local processes index -**
<pre>
<b>int MPI_Comm_rank(MPI_Comm comm, int *rank)</b>
</pre>
- **To send information -**
<pre>
<b>int MPI_Send(void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm)</b>
</pre>
- **To receive information -**
<pre>
<b>int MPI_Recv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Status *status)</b>
</pre>
