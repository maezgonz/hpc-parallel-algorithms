#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
 
int main(int argc, char *argv[])
{
int err = 0;
int *sendbuf, recvbuf, *recvcounts;
int npes, myid, i, sumval;
 
MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &npes);	
MPI_Comm_rank(MPI_COMM_WORLD, &myid);

sendbuf = (int *) malloc( npes * sizeof(int) );

for (i=0; i<npes; i++) 
        sendbuf[i] = myid + i;

recvcounts = (int *)malloc( npes * sizeof(int) );
for (i=0; i<npes; i++) 
       recvcounts[i] = 1;
 
MPI_Reduce_scatter(sendbuf, &recvbuf, recvcounts, MPI_INT, MPI_SUM, MPI_COMM_WORLD );
 
sumval = npes * myid + ((npes - 1) * npes)/2;

printf("\n P(%d) got %d expected %d\n", myid, recvbuf, sumval );fflush(stdout);

MPI_Finalize( );

}
