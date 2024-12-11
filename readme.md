# Parallel Processing & Distributed System Lab
**[Ahnaf Shahrear Khan](https://github.com/ahnafshahrear) | ahnafshahrearkhan@gmail.com | [LinkedIn](https://www.linkedin.com/in/ahnafshahrearkhan/) | [Facebook](https://www.facebook.com/ahnaf.shahrear.khan)**
- **Computer Science & Engineering, University of Rajshahi**
- **Course Code: `CSE-4112`**


## Installation of MPI in Windows Subsystem for Linux
- **Go to `Turn Windows features on or off` & turn on the `Windows Subsystem for Linux`**
- **Now go to windows Terminal & run the 1st command (if doesn't work run the 2nd command)**
<pre>
<b>wsl.exe --install</b>
<b>wsl --install -d Ubuntu-22.04</b>
</pre>
- **If any error occurs go to the provided link & download the latest package from `Step 4 - Download the Linux kernel update package`**
- **If further error occurs with error code `0x80370102` go to Terminal & run the following command**
<pre>
<b>wsl --update</b>
<b>wsl --set-default-version 1</b>
</pre>
- **Now go to `Ubunto` & Setup with Username & Password**
- **Then run the following commands on `Ubunto`**
<pre>
<b>sudo apt-get update</b>
<b>sudo apt-get install mpich</b>
</pre>
- **Once after every session you have to mount your drive using the command on `Ubuntu`**
<pre>
<b>cd /mnt/(C/D/E/F)</b>
</pre>


### Run Command on Ubuntu
- **To run a C Program, go to `Ubuntu` & then run the following commands**
<pre>
<b>mpicc name.c -o name</b>
<b>mpirun -n number_of_processors ./name</b>  
</pre>
- **To run a C++ Program, go to `Ubuntu` & then run the following commands**
<pre>
<b>mpic++ name.cpp -o name</b>
<b>mpirun -n number_of_processors ./name</b>  
</pre>


## Installation of MPI in Visual Studio Code
- **To install MPI, see this [video](https://www.youtube.com/watch?v=bkfCrj-rBjU) & follow every step carefully. If you face the issue "sal.h dependency not found" or something like that, uninstall the MinGW compiler & follow this [video](https://www.youtube.com/watch?v=_-O94qsnOLk)**


### Run Command on Visual Studio Code
- **To run a C Program, go to `Terminal` > `Run Build Task...` in the VS Code & then run the following command**
<pre>
<b>mpiexec -n number_of_processors file_name_without_extension</b>
</pre>



## Lab Tasks
- **Write an MPI program to multiply two matrices of size MxN & NxP ✓**
- **Given a list of names & phone numbers as a phonebook. Write an MPI program to search for a name & phone number from the phonebook ✓**
- **Write an MPI program to simulate a simple calculator using a different process in parallel ✓**
- **Write an MPI program to count the words in a file & sort it in descending order of frequency of words that is, the highest occurring word must come first & the least occurring word must come last**
- **Write an MPI program using synchronous send. The sender process sends a word to the receiver. The second process receives the word, toggles each letter of the word and sends it back to the first process. Both processes use synchronous send operations ✓**
- **Write an MPI program to add an array of size N using two processes. Print the result in the root process. Investigate the amount of time taken by each process ✓**
- **Write a CUDA program for matrix multiplication ✓**
- **Given a list of names & phone numbers as a phonebook. Write a CUDA program to search for a name & phone number from a phonebook ✓**
- **Given a paragraph & a pattern like %x%. Now write a CUDA program to find out the line number where %x% this pattern exists in the given paragraph**
