## Parallel Processing & Distributed System Lab
- **[Ahnaf Shahrear Khan](https://github.com/ahnafshahrear)**
- **Computer Science & Engineering, University of Rajshahi**
- **Code:** `CSE4112`

### Installation Guide (MPI)
- **To install MPI, see this [video](https://www.youtube.com/watch?v=bkfCrj-rBjU) & follow every step carefully. If you face the issue "sal.h dependency not found" or something like that, uninstall the MinGW compiler & follow this [video](https://www.youtube.com/watch?v=_-O94qsnOLk)**


### Run Command 
- **First go to `Terminal` > `Run Build Task...` in the VS Code & then run the following command**
```
mpiexec -n number_of_processors file_name_without_extension
```

### Lab Tasks
- **Write a simple C++ program in MPI to multiply two matrices of size MxN and NxP**
- **Write a program in MPI to simulate a simple calculator. Perform each operation using a different process in parallel ✓**
- **Write a program in C++ to count the words in a file and sort it in descending order of frequency of words that is, the highest occurring word must come first and least occurring word must come last**
- **Write a MPI program using synchronous send. The sender process sends a word to the receiver. The second process receives the word, toggles each letter of the word and sends it back to the first process. Both processes use synchronous send operations**
- **Write a MPI program to add an array of size N using two processes. Print the result in the root process. Investigate the amount of time taken by each process**
- **Write a Cuda program for matrix multiplication**
- **Write a Cuda program to find out maximum common subsequence**
- **Given a paragraph and a pattern like %x%. Now write a Cuda program to find out the line number where %x% this pattern exists in the given paragraph**
