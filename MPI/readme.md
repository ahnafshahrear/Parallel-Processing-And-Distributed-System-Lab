## Basic Instructions of MPI
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
