#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


int main(int argc, char *argv[])
{
	int myid, npes, size,i;
	char message[10];
	int *vector;
	int vector_rec[1];

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

	if (myid == 0) {
	   vector=(int *) malloc(npes*sizeof(int));
	   for(i=0;i<npes;i++)
		vector[i]=i*100+1;
	}
	else 
	   vector=(int *) malloc(1*sizeof(int));

	MPI_Scatter(vector,1,MPI_INT,vector_rec,1,MPI_INT,0,MPI_COMM_WORLD);
	
	printf("I am  %d vector_rec:%d \n", myid, vector_rec[0]);

	MPI_Finalize();
}


