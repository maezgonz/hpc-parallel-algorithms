#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[])
{
	int myid, npes, size, tag = 666, i;

    int message_recv;

	MPI_Status status;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

if (myid == 0) {
        MPI_Send(&myid, 1, MPI_INT, 1, tag, MPI_COMM_WORLD);
    } 
    else if (myid == 1) {
        MPI_Recv(&message_recv, 1, MPI_INT, 0, tag, MPI_COMM_WORLD, &status);
        printf("I am process %d and I received the message: %d\n", myid, message_recv);
        MPI_Send(&myid, 1, MPI_INT, 2, tag, MPI_COMM_WORLD);
    } 
    else if (myid == 2) {
        MPI_Recv(&message_recv, 1, MPI_INT, 1, tag, MPI_COMM_WORLD, &status);
        printf("I am process %d and I received the message: %d\n", myid, message_recv);
        MPI_Send(&myid, 1, MPI_INT, 3, tag, MPI_COMM_WORLD);
    } 
    else if (myid == 3) {
        MPI_Recv(&message_recv, 1, MPI_INT, 2, tag, MPI_COMM_WORLD, &status);
        printf("I am process %d and I received the message: %d\n", myid, message_recv);
    }

	MPI_Finalize();
}
