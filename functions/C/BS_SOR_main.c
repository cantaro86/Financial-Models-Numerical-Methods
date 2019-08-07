#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SOR.h"
#include "PDE_solver.h"
#include <math.h>




int main()
{

  double r = 0.1;
  double sig = 0.2;
  double S = 100.0;
  double K = 100.0;
  double T = 1.;

  int Nspace = 3000;    // space steps
  int Ntime = 2000;     // time steps   

  double w = 1.68;       // relaxation parameter
  
  printf("The price is: %f \n ", PDE_SOR(Nspace,Ntime,S,K,T,sig,r,w) );

  return 0;
}


