#ifndef SOR_H
#define SOR_H


#include <stdio.h>
#include <string.h>
#include <stdlib.h>


static const int N = 4;

void print_matrix(double arr[N][N]);
void print_array(double* arr);

double* SOR(double A[N][N], double* b, double w, double eps, int N_max);
double* SOR_trid(double A[N][N], double* b, double w, double eps, int N_max);
double* SOR_abc(double aa, double bb, double cc, double* b, int NN, double w, double eps, int N_max);
double* SOR_aabbcc(double aa, double bb, double cc, double *b, double *x0, double *x1,
		   int NN, double w, double eps, int N_max);


double dist_square(double* a, double* b);
double dist_square2(double* a, double* b, int NN);
void print_array_2(double* arr, int NN);

#endif
