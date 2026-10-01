#include <mpi.h>
#include <stdio.h>

int main(int argc, char *argv[])
{
  int myid, npes;
  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &npes);	
  MPI_Comm_rank(MPI_COMM_WORLD, &myid);

   if (myid == 0)   
   {
     printf ("\n I am process %i out of %i: Hello world!\n", 
             myid, npes);
   }
   else
   {
     printf("\n I am process %i out of %i \n", myid, npes);
   }

	MPI_Finalize();
}