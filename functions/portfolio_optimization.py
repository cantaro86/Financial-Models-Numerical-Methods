import numpy as np
from scipy.optimize import minimize, Bounds, LinearConstraint



def optimal_weights(MU, COV, Rf=0, w_max=1, desired_mean=None, desired_std=None):
    """
    Compute the optimal weights for a portfolio containing a risk free asset and several stocks.
    MU = vector of mean
    COV = covariance matrix
    Rf = risk free return
    w_max = maximum weight bound for the stock portfolio
    desired_mean = desired mean of the portfolio
    desired_std = desired standard deviation of the portfolio
    """
    
    if (desired_mean!=None) and (desired_std!=None):
        raise ValueError("One among desired_mean and desired_std must be None")
    if ((desired_mean!=None) or (desired_std!=None)) and Rf==0 :
        raise ValueError("We just optimize the Sharpe ratio, no computation of efficient frontier")
    
    N = len(MU)
    bounds = Bounds(0, w_max)
    linear_constraint = LinearConstraint( np.ones(N, dtype=int),1,1)
    weights = np.ones(N)
    x0 = weights/np.sum(weights)      # initial guess

    sharpe_fun = lambda w:  -(MU @ w - Rf) / np.sqrt(w.T @ COV @ w) 
    res = minimize(sharpe_fun, x0=x0, method='trust-constr', constraints=linear_constraint, bounds=bounds)
    print(res.message + '\n')
    w_sr = res.x
    std_stock_portf = np.sqrt(w_sr@COV@w_sr)
    mean_stock_portf = MU@w_sr
    stock_port_results = {"Sharpe Ratio" : -sharpe_fun(w_sr), "stock weights" : w_sr.round(4),
           "stock portfolio" : {"std": std_stock_portf.round(6), "mean" : mean_stock_portf.round(6)} }
    
    if (desired_mean==None) and (desired_std==None):
        return stock_port_results
    
    elif (desired_mean==None) and (desired_std!=None):
        w_stock = desired_std/std_stock_portf
        if desired_std>std_stock_portf:
            print("The risk you take is higher than the tangency portfolio risk ==> SHORT POSTION")
        tot_port_mean = Rf + w_stock *(mean_stock_portf-Rf) 
        return {**stock_port_results, "Bond + Stock weights" : {"Bond": (1-w_stock).round(4), 
                                                                "Stock": w_stock.round(4) },
                "Total portfolio":{"std": desired_std, "mean": tot_port_mean.round(6)} }
    
    elif (desired_mean!=None) and (desired_std==None):
        w_stock = (desired_mean-Rf)/(mean_stock_portf-Rf)
        if desired_mean>mean_stock_portf:
            print("The return you want is higher than the tangency portfolio return ==> SHORT POSTION")
        tot_port_std = w_stock * std_stock_portf
        return {**stock_port_results, "Bond + Stock weights" : {"Bond": (1-w_stock).round(4), 
                                                                "Stock": w_stock.round(4)},
               "Total portfolio":{"std": tot_port_std.round(6), "mean": desired_mean} }
