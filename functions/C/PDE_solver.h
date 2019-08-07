#ifndef PDE_SOLVER_H
#define PDE_SOLVER_H


#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "SOR.h"
#include <math.h>

double PDE_SOR(int Ns, int Nt, double S, double K, double T, double sig, double r, double w);


#endif
