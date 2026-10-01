/* Obtain the PI value using the numerical integration of 4/(1+x*x) between 0 and 1.
PI/4=arctan(1)=integral 1/(1+x*x) , between 0 and 1.
The numerical integration is calculated with n rectangular intervals (area=(1/n)*4/(1+x*x))) and adding the area of all these rectangle intervals  
Execution time: around 250 seconds */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(argc,argv)
int argc;
char *argv[];
{
    long int n=80000000000; /* ONLY IN ROOT PROCESS*/ 
    long int i;
    double PI25DT = 3.141592653589793238462643;
    double pi, h, sum, x;
    
    clock_t begin = clock();

    h   = 1.0 / (double) n;  //wide of the rectangle
    sum = 0.0;
    for (i = 0; i < n; i ++) {
	x = h * ((double)i + 0.5);   //height of the rectangle
        sum += 4.0 / (1.0 + x*x);
    }

    pi = h * sum;

    clock_t end = clock();

    printf("The obtained Pi value is: %.16f, the error is: %.16f\n", pi, fabs(pi - PI25DT));

    printf("Execution time: %f seconds\n", (double)(end - begin)/CLOCKS_PER_SEC);

}
