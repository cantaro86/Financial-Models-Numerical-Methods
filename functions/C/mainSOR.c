#include "SOR.h"



/*  
  GENERAL MATRIX (DIAGONAL DOMINANT)
  
  double A[4][4] = { {10, 5, 2, 1},
		     {2, 15, 2, 3},
		     {1, 8, 13, 1},
		     {2, 3, 1, 8} };
  double b[4] = {30, 50, 60, 43}; 
  double* x =  SOR(A,b, w,eps, N_max);

  TRIDIAGONAL MATRIX
  double A[4][4] = { {10, 5, 0, 0},
		     {2, 15, 2, 0},
		     {0, 8, 13, 1},
		     {0, 0, 1, 8} };
  double b[4] = {20, 38, 59, 35};
  double* x = SOR_trid(A, b, w, eps, N_max);
*/



int main()
{
  // TRIDIAGONAL WITH CONSTANT aa,bb,cc
  double A[4][4] = { {10, 5, 0, 0},
		     {2, 10, 5, 0},
		     {0, 2, 10, 5},
		     {0, 0, 2, 10} };

  double b[4] = {20, 37, 54, 46};
  
  double aa=2, bb=10, cc=5;
  
 
  const double w = 1;
  const double eps = 1e-10;
  const int N_max = 100;

  
  printf("Matrix A: \n");
  print_matrix(A);
  printf("Vector b: \n");
  print_array(b);

  //  double* x = SOR_abc(aa, bb, cc, b, N, w, eps, N_max);

  double* x0 = calloc(N, sizeof(double) );
  double* x1 = calloc(N, sizeof(double) );
  double* x = SOR_aabbcc(aa, bb, cc, b, x0, x1, N, w, eps, N_max);

  printf("Solution x: \n");
  print_array(x);

  free(x0);
  free(x1);
  
  return 0;
}
