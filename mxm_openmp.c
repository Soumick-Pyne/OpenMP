# include <stdlib.h>
# include <stdio.h>
# include <math.h>
# include <time.h>
# include <omp.h>

int main ( void );
void timestamp ( void );

/******************************************************************************/

int main ( void )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for MXM_OPENMP.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    13 October 2011

  Author:

    John Burkardt
*/
/*
The code below solves matrix matrix multiplication problem using
OpenMP for parallel execution
*/
{
  double a[500][500];
  double angle;
  double b[500][500];
  double c[500][500];
  int i;
  int j;
  int k;
  int n = 500;
  double pi = 3.141592653589793;
  double s;
  int thread_num;
  double wtime;

  timestamp ( );

  printf ( "\n" );
  printf ( "MXM_OPENMP:\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  Compute matrix product C = A * B.\n" );

  //assigning thread number to be the maximum possible during runtime
  //More threads means less iterations per thread => lower runtime
  thread_num = omp_get_max_threads ( );

  printf ( "\n" );
  //The integer returned by this function may be less than the total no. of 
  //physical processors in the multiprocessor depending on how the run-tme system
  //gives access to processors
  printf ( "  The number of processors available = %d\n", omp_get_num_procs ( ) );
  printf ( "  The number of threads available    = %d\n", thread_num );

  printf ( "  The matrix order N                 = %d\n", n );

  s = 1.0 / sqrt ( ( double ) ( n ) );

  //The omp_get_wtime routine returns elapsed wall clock time in seconds
  wtime = omp_get_wtime ( );

//inside this block every time the program forks all threads will have their own copy
//of variables declared private by the directive below. Rest of the shared variables 
//accessible by all threads at a common memory address.
# pragma omp parallel shared ( a, b, c, n, pi, s ) private ( angle, i, j, k )
{ 
/*
  Loop 1: Assign value to matrix A.
*/
  /*
  Here we parallelize the outer loop and not the inner loop.
  Reason: If we parallelize the inner loop, then the program will fork 
  and join threads for every iteration of the outer
  loop. The fork/join overhead may very well be greater than the time saved by
  dividing the execution of the n iterations of the inner loop among multiple threads.
  On the other hand, if we parallelize the outer loop, the program only incurs the
  fork/join overhead once
  */
  # pragma omp for
  for ( i = 0; i < n; i++ )
  {
    for ( j = 0; j < n; j++ )
    {
      angle = 2.0 * pi * i * j / ( double ) n;
      a[i][j] = s * ( sin ( angle ) + cos ( angle ) );
    }
  }
/*
  Loop 2: Copy A into B.
  Here we parallelize the outer loop and not the inner loop.
  Reason mentioned in Loop 1 comment.
*/
  # pragma omp for
  for ( i = 0; i < n; i++ )
  {
    for ( j = 0; j < n; j++ )
    {
      b[i][j] = a[i][j];
    }
  }
/*
  Loop 3: Compute C = A * B.
  Here we parallelize the outer loop and not the inner loops.
  Reason mentioned in Loop 1 comment.
*/
  /*
  Here by structure no 2 threads accesses the same index of the 3 matrices 
  so no cache coherence protocols required seperately
  */
  # pragma omp for
  for ( i = 0; i < n; i++ )
  { 
    //Traversing along row i of A
    for ( j = 0; j < n; j++ )
    {
      //Traversing along column j of B
      c[i][j] = 0.0;
      for ( k = 0; k < n; k++ )
      {
        //updating value of 1 element of C
        c[i][j] = c[i][j] + a[i][k] * b[k][j];
      }
    }
  }

}
//Calculating time taken by the code from the point A was initialised to the point
//where C was calculated and all threads joined
  wtime = omp_get_wtime ( ) - wtime;
  printf ( "  Elapsed seconds = %g\n", wtime );
  printf ( "  C(100,100)  = %g\n", c[99][99] );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "MXM_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );

  printf ( "\n" );
  timestamp ( );

  return 0;
}
/******************************************************************************/

void timestamp ( void )

/******************************************************************************/
/*
  Purpose:

    TIMESTAMP prints the current YMDHMS date as a time stamp.

  Example:

    31 May 2001 09:45:54 AM

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    24 September 2003

  Author:

    John Burkardt

  Parameters:

    None
*/
{
# define TIME_SIZE 40

  static char time_buffer[TIME_SIZE];
  const struct tm *tm;
  time_t now;

  now = time ( NULL );
  tm = localtime ( &now );

  strftime ( time_buffer, TIME_SIZE, "%d %B %Y %I:%M:%S %p", tm );

  printf ( "%s\n", time_buffer );

  return;
# undef TIME_SIZE
}
