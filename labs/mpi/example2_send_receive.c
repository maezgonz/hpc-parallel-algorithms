#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
	int myid, npes, size, tag = 666, i;
	char message[100];

	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
	if(myid==0) {
		strcpy(message, "salam"); 
		size = strlen(message)+1;
	}
	else
	{
		strcpy(message, "empty"); 
		size = strlen(message)+10;
	}

	printf("Before: I am the node %d and I have the messages %s\n", myid, message);

	if(myid == 0)
	 	for(i=1; i<npes;i++)
	    	MPI_Send(message, size, MPI_CHAR, i, tag, MPI_COMM_WORLD);
    else 
		MPI_Recv(message, size, MPI_CHAR, 0, tag, MPI_COMM_WORLD, &status);


	printf("After: I am the node %d and I have the message %s \n", myid, message);
	MPI_Finalize();
}
