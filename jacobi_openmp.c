# include <math.h>
# include <stdio.h>
# include <stdlib.h>

int main ( );

/******************************************************************************/

int main ( )

/******************************************************************************/
/*
  Purpose:

    MAIN is the main program for JACOBI_OPENMP.

  Discussion:

    JACOBI_OPENMP carries out a Jacobi iteration with OpenMP.

  Licensing:

    This code is distributed under the GNU LGPL license. 

  Modified:

    31 January 2017

  Author:

    John Burkardt
*/
{
  double *b;
  double d;
  int i;
  int it;
  int m;
  int n;
  double r;
  double t;
  double *x;
  double *xnew;

  m = 5000;
  n = 50000;

  b = ( double * ) malloc ( n * sizeof ( double ) );
  x = ( double * ) malloc ( n * sizeof ( double ) );
  xnew = ( double * ) malloc ( n * sizeof ( double ) );

  printf ( "\n" );
  printf ( "JACOBI_OPENMP:\n" );
  printf ( "  C/OpenMP version\n" );
  printf ( "  Jacobi iteration to solve A*x=b.\n" );
  printf ( "\n" );
  printf ( "  Number of variables  N = %d\n", n );
  printf ( "  Number of iterations M = %d\n", m );

  printf ( "\n" );
  printf ( "  IT     l2(dX)    l2(resid)\n" );
  printf ( "\n" );
  /*
  Each thread in the loop inside the block below will have it's own copy of i
  All other variables in the loops will be shared by default because of the type 
  of abstaction on which OpenMP is buit.
  */
# pragma omp parallel private ( i )
  {
/*
  Set up the right hand side for Ax=b
  _
*/
# pragma omp for
    for ( i = 0; i < n; i++ )
    {
      b[i] = 0.0;
    }

    b[n-1] = ( double ) ( n + 1 );
/*
  Initialize the solution estimate to 0.
  Exact solution is (1,2,3,...,N).
*/
# pragma omp for
    for ( i = 0; i < n; i++ )
    {
      x[i] = 0.0;
    }

  }
/*
  Iterate M times.
  This outermost loop cannot be parallelised in a true sense since each iteration
  is dependent upon values from the previous iteration so threads corresponding
  to differnt iterations cannot run concurrently.

*/
  for ( it = 0; it < m; it++ )
  {
/*The private directive declares data to have a separate copy in the memory of each thread. 
  Such private variables are initialized as they would be in a main program.
  Any computed value goes away at the end of the parallel region
*/
# pragma omp parallel private ( i, t )
    {
/*
  Jacobi update.
  _ A matrix : The main diagonal elements are all 2. 
  			   The elements of the diagonals adjacent to the main diagonal 
  			   on either side are all -1.
  			   Rest all elements are zero
  The for loop below has a cannonical shape so it can be multithreaded by the complier.
  Each thread in the loop below modifies a different index of the array xnew[]
  so there is no requirement of snooping of any kind.
*/
# pragma omp for
      for ( i = 0; i < n; i++ )
      {
        xnew[i] = b[i];
        if ( 0 < i )
        {
          xnew[i] = xnew[i] + x[i-1];
        }
        if ( i < n - 1 )
        {
          xnew[i] = xnew[i] + x[i+1];
        }
        xnew[i] = xnew[i] / 2.0;
      }
/*
  Difference.
  d below is the reduction variable that will hold the result of the summation
  of values across threads.
*/
      d = 0.0;
# pragma omp for reduction ( + : d )
      for ( i = 0; i < n; i++ )
      {
        d = d + pow ( x[i] - xnew[i], 2 );
      }
/*
  Overwrite old solution.
  This is an "embarassingly parallel" block
*/
# pragma omp for
      for ( i = 0; i < n; i++ )
      {
        x[i] = xnew[i];
      }
/*
  Residual.
  r below is the reduction variable that will hold the result of the summation 
  of values across threads. OpenMP will take care of details like storing partial
  sums in private variables and then adding the partial sums to the shared variable.
*/
      r = 0.0;
# pragma omp for reduction ( + : r )
      for ( i = 0; i < n; i++ )
      {
        t = b[i] - 2.0 * x[i];
        if ( 0 < i )
        {
          t = t + x[i-1];
        }
        if ( i < n - 1 )
        {
          t = t + x[i+1];
        }
        r = r + t * t;
      }
/*This just prints the iteration no. and L2 norm values
The omp master directive means that the section of code that follows
must be run only by the master thread
*/
# pragma omp master
      {
        if ( it < 10 || m - 10 < it )
        {
          printf ( "  %8d  %14.6g  %14.6g\n", it, sqrt ( d ), sqrt ( r ) );
        }
        if ( it == 9 )
        {
          printf ( "  Omitting intermediate results.\n" );
        }
      }

    }

  }
/*
  Write part of final estimate. 
  No parallel directive below
*/
  printf ( "\n" );
  printf ( "  Part of final solution estimate:\n" );
  printf ( "\n" );
  for ( i = 0; i < 10; i++ )
  {
    printf ( "  %8d  %14.6g\n", i, x[i] );
  }
  printf ( "...\n" );
  for ( i = n - 11; i < n; i++ )
  {
    printf ( "  %8d  %14.6g\n", i, x[i] );
  }
/*
  Free memory to avoid creating orphaned blocks in memory created by dynamic allocation.
*/
  free ( b );
  free ( x );
  free ( xnew );
/*
  Terminate.
*/
  printf ( "\n" );
  printf ( "JACOBI_OPENMP:\n" );
  printf ( "  Normal end of execution.\n" );

  return 0;
}

