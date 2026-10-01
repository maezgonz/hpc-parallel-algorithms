#include <mpi.h>
#include <stdio.h>
#include <string.h>

#define V_dim 100000

int main(int argc, char *argv[])
{
	int myid, npes, size, tag = 666, i;
	int src,dest;
	double Vector_send[V_dim];
	double Vector_recv[V_dim];

	MPI_Status status;
	MPI_Request request_send,request_recv;

    double T_ini_send, T_end_send;
    double T_ini_recv, T_end_recv;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);
	
	size=V_dim;
	
    for (i = 0; i < V_dim; i++)
		Vector_send[i] = (float) 100*myid + 1;
	
    src = myid-1;
    dest = myid+1;
    if (src < 0)
		src += npes;
    if (dest >= npes)
		dest -= npes;

	T_ini_send=MPI_Wtime();
    MPI_Isend(Vector_send, size, MPI_DOUBLE, dest, tag, MPI_COMM_WORLD,&request_send);
	T_end_send=MPI_Wtime();
	T_ini_recv=MPI_Wtime();
	MPI_Recv(Vector_recv, size, MPI_DOUBLE, src, tag, MPI_COMM_WORLD, &status);

	T_end_recv=MPI_Wtime();

	printf("I am the node %d and I have the message from %d Time_send:%lf Time_recv:%lf \n", myid, status.MPI_SOURCE,T_end_send-T_ini_send,T_end_recv-T_ini_recv);

	MPI_Finalize();
}
