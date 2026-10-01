#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[])
{
int myid, npes, size;
double pdf_local;
double cfg_cum;

MPI_Init(&argc, &argv);
MPI_Comm_size(MPI_COMM_WORLD, &npes);	
MPI_Comm_rank(MPI_COMM_WORLD, &myid);

/*local probability in each process*/
pdf_local = 0.5*exp(-0.5*(npes-myid) /2.0);

/* obtain the cumulative frequency histogram in cfg_cum in each process*/
MPI_Scan(&pdf_local, &cfg_cum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

printf("\n P(%d) Local_probability:%lf cumulative sum = %lf \n",myid, pdf_local, cfg_cum);

MPI_Finalize();
}

