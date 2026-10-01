#include <mpi.h>
#include <stdio.h>
#include <string.h>


int main(int argc, char *argv[])
{
	int myid, npes, size;
	char message[10];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	if (myid == 0) 
		strcpy(message, "Hello");
	else  
		strcpy(message, "Nothing");

	printf("Before Broadcast: I am the node %d and my message is %s\n", myid, message);
	size = 10;
	MPI_Bcast(message, size, MPI_CHAR, 0, MPI_COMM_WORLD);
	printf("After Broadcast: I am the node %d and my message is %s\n", myid, message);

	MPI_Finalize();
}
