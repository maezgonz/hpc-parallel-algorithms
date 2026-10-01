// Use only two processes
// Example. Proccess 0 sends the second row of its A matrix to proccess 1 


#include "mpi.h"
#include <stdio.h>
#include <string.h>
#define N 10

int main(int argc,char *argv[])
{

    MPI_Status info;
    int myid, npes, i, j;
    int tag=9;
    float A[N][N];

    MPI_Init(&argc,&argv);
    MPI_Comm_size(MPI_COMM_WORLD,&npes);
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);

    if (myid==0)
    	for (i = 0; i < N; i++)
	        for (j = 0; j < N; j++)
	            A[i][j] = (float) 100*i + j;
    else 
    	for (i = 0; i < N; i++)
	        for (j = 0; j < N; j++)
	            A[i][j] = (float) -99;

    if (myid==1) {
        printf("\n Proccess 1. Original matrix\n");
	    for (i = 0; i < N; i++){
	        for (j = 0; j < N; j++) 
 	    	    printf("%5.f", A[i][j]);
	    printf("\n");
    	}
    }

    if (myid==0)
	    MPI_Send(&A[2][0], 10, MPI_FLOAT, 1, tag, MPI_COMM_WORLD);
    else if (myid==1)
	    MPI_Recv(&A[2][0], 10, MPI_FLOAT, 0, tag, MPI_COMM_WORLD, &info);

    if (myid==1) {
        printf("\n Proccess 1. New matrix\n");
	    for (i = 0; i < N; i++){
	        for (j = 0; j < N; j++) 
 	    	    printf("%5.f", A[i][j]);
	        printf("\n");
   	    }
    }

   MPI_Finalize();
}

