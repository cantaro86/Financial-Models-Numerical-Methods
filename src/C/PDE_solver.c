#include "PDE_solver.h"




double PDE_SOR(int Ns, int Nt, double S, double K, double T, double sig, double r, double w)
{

  const double eps = 1e-10;
  const int N_max = 600;
  
  double S_max = 3*K;                
  double S_min = K/3;
  double x_max = log(S_max);
  double x_min = log(S_min);

  double dx = (x_max - x_min)/(Ns-1);
  double dt = T/Nt;

  double sig2 = sig*sig; 
  double dxx = dx * dx;
  double aa = ( (dt/2) * ( (r-0.5*sig2)/dx - sig2/dxx ) );
  double bb = ( 1 + dt * ( sig2/dxx + r ) );
  double cc = (-(dt/2) * ( (r-0.5*sig2)/dx + sig2/dxx ) );


  // array allocations  
  double *x = calloc(Ns,  sizeof(double) );  
  double *x_old = calloc(Ns-2,  sizeof(double) );
  double *x_new = calloc(Ns-2,  sizeof(double) );
  double *help_ptr = calloc(Ns-2,  sizeof(double) );
  double *temp;

  for (unsigned int i=0; i<Ns; ++i)  // price vector
    x[i] = exp(x_min + i * dx);
  
  for (unsigned int i=0; i<Ns-2; ++i)  // payoff
    x_old[i] = fmax( x[i+1] - K, 0 );

  
  // Backward iteration
  for (int k=Nt-1; k>=0; --k)
    {
      x_old[Ns-3] -= cc * ( S_max - K * exp( -r*(T-k*dt) ) );  // offset
      x_new = SOR_aabbcc(aa, bb, cc, x_old, help_ptr, x_new, Ns-2, w, eps, N_max);  //SOR solver
      // x_new = SOR_abc(aa, bb, cc, x_old, Ns-2, w, eps, N_max);  //SOR solver
      
      if (k != 0)  // swap the pointers (we don't need to allocate new memory) 
	{
	  temp = x_old;    
	  x_old = x_new;
	  x_new = temp;    
	}
    }
  free(help_ptr);
  free(x_old);

  // x_new is the solution!! 

  // binary search:  Search for the points for the interpolation  

  int low = 1;
  int high = Ns-2;
  int mid;
  double result = -1;

  if (S > x[high] || S < x[low])
    {
      printf("error: Price S out of grid.\n");
      free(x_new);
      free(x);
      return result;
    }
  
  while ( (low+1) != high)
    {

      mid = (low + high) / 2;
      
      if ( fabs(x[mid]-S)< 1e-10 )
	{
	  result = x_new[mid-1];
	  free(x_new);
	  free(x);
	  return result;
	}
      else if ( x[mid] < S)
	{
	  low = mid;
	}
      else
	{
	  high = mid;
	}
    }

  // linear interpolation
  result = x_new[low-1] + (S - x[low]) * (x_new[high-1] - x_new[low-1]) / (x[high] - x[low])  ;
  free(x_new);
  free(x);
  return result;
}  
