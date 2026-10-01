#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char *argv[])
{
  int myid, npes, size, i;
  int N=100000;
  float *x, *y;
  float dot,total_dot;
  double time_st, time_end, mytime;    
	double maxtime,mintime,totaltime,avgtime;

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &npes);	
	MPI_Comm_rank(MPI_COMM_WORLD, &myid);

  /* Memory for vector*/
  if((x = (float *) malloc(N*sizeof(float))) == NULL)
      printf("Error en malloc x[%d]\n",N);
  if((y = (float *) malloc(N*sizeof(float))) == NULL)
        printf("Error en malloc y[%d]\n",N);

  /* Vector inicialization in each proccess*/
  for(i=0; i<N; i++){
		x[i] = log10((N - i))/(myid+1);
		y[i] = 0.0001*i*(myid+1);
  }

  /* Dot product in each proccess*/
  time_st = MPI_Wtime();
  dot = 0.0;
  for(i=0; i<N; i++)
		  dot += x[i] * y[i];
  
  time_end = MPI_Wtime();
  mytime = time_end-time_st;

  printf("Process:%d Local dot product = %lf in (sec):%lf\n", myid,dot,mytime);

  MPI_Reduce(&dot, &total_dot, 1, MPI_FLOAT,MPI_SUM, 0, MPI_COMM_WORLD);

  /*compute max, min, and average timing statistics*/
  MPI_Reduce(&mytime, &maxtime, 1, MPI_DOUBLE,MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&mytime, &mintime, 1, MPI_DOUBLE, MPI_MIN, 0,MPI_COMM_WORLD);
  MPI_Reduce(&mytime, &totaltime, 1, MPI_DOUBLE, MPI_SUM, 0,MPI_COMM_WORLD);
  if (myid == 0) {
    avgtime = totaltime/npes;
    printf("Root process: Results:%g Min_Time: %lf Max_Time:%lf Avg_Time:%lf\n", total_dot,mintime, maxtime,avgtime);
  }

   MPI_Finalize();
}


