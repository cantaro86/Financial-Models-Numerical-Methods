Financial-Models-Numerical-Methods 
==================================


This is a collection of [Jupyter notebooks](https://jupyter.org/) based on different topics in the area of quantitative finance.


### Is this a tutorial?

Almost! :) 

This is just a collection of topics and algorithms that in my opinion are interesting.     

It contains several topics that are not so popular nowadays, but that can be very powerful. 
Usually, topics such as PDE methods, Lévy processes, Fourier methods or Kalman filter are not very popular among practitioners, who prefers to work with more standard tools.     
The aim of these notebooks is to present these interesting topics, by showing their practical application through an interactive python implementation.


### Who are these notebooks for?

Not for absolute beginners. 

These topics require a basic knowledge in stochastic calculus, financial mathematics and statistics. A basic knowledge of python programming is also necessary.

In these notebooks I will not explain what is a call option, or what is a stochastic process, or a partial differential equation.     
However, every time I will introduce a concept, I will also add a link to the corresponding wiki page or to a reference manual.
In this way, the reader will be able to immediately understand what I am talking about. 

These notes are for students in science, economics or finance who have followed at least one undergraduate course in financial mathematics and statistics.       
Self-taught students or practicioners should have read at least an introductiory book on financial mathematics. 


### Why is it worth to read these notes?  

First of all, this is not a book!      
Every notebook is (almost) independent from the others. Therefore you can select only the notebook you are interested in!

```diff
- Every notebook, contains python code ready to use!     
```

It is not easy to find on internet examples of financial models implemented in python which are ready to use and well documented.    
I think that beginners in quantitative finance will find these notebooks very useful!  

Moreover, Jupyter notebooks are interactive i.e. you can run the code inside the notebook. 
This is probably the best way to study!

If you open a notebook with Github or NBviewer, sometimes mathematical formulas are not displayed correctly. 
For this reason, I suggest you to clone/download the repository. 


### Is this series of notebooks complete?

**No!**    
I will upload more notebooks from time to time. 

At the moment, I'm interested in the areas of stochastic processes, Kalman Filter, statistics and much more. I will add more interesting notebooks on these topics in the future. 

If you have any kind of questions, or if you find some errors, or you have suggestions for improvements, feel free to contact me.      
This is my [linkedin](https://www.linkedin.com/in/nicolacantarutti) page.



### Contents

1.1) **Black-Scholes numerical methods** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.1%20Black-Scholes%20numerical%20methods.ipynb) *(lognormal distribution, change of measure, Monte Carlo, Binomial method)*.

1.2) **SDE simulation and statistics** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.2%20SDE%20simulations%20and%20statistics.ipynb)
*(paths generation, Confidence intervals, Hypothesys testing, Geometric Brownian motion, Cox-Ingersoll-Ross process, Euler Maruyama method, parameters estimation)*

1.3) **Fourier inversion methods** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.3%20Fourier%20transform%20methods.ipynb)
*(derivation of inversion formula, numerical inversion, option pricing)*

1.4) **SDE, Heston model** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.4%20SDE%20-%20Heston%20model.ipynb)
*(correlated Brownian motions, Heston paths, Heston distribution, characteristic function, option pricing)*

1.5) **SDE, Lévy processes** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/1.5%20SDE%20-%20L%C3%A9vy%20processes.ipynb)
*(Merton, Variance Gamma, NIG, path generation, parameter estimation)*

2.1) **The Black-Scholes PDE** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/2.1%20Black-Scholes%20PDE%20and%20sparse%20matrices.ipynb)
*(PDE discretization, Implicit method, sparse matrix tutorial)*

2.2) **Exotic options** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/2.2%20Exotic%20options.ipynb)
*(Binary options, Barrier options)*

2.3) **American options** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/2.3%20American%20Options.ipynb)
*(PDE, Binomial method, Longstaff-Schwartz)*

3.1) **Merton Jump-Diffusion PIDE** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/3.1%20Merton%20jump-diffusion%2C%20PIDE%20method.ipynb)
*(Implicit-Explicit discretization, discrete convolution, model limitations, Monte Carlo, Fourier inversion, semi-closed formula )*

3.2) **Variance Gamma PIDE** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/3.2%20Variance%20Gamma%20model%2C%20PIDE%20method.ipynb)
*(approximated jump-diffusion PIDE, Monte Carlo, Fourier inversion, Comparison with Black-Scholes)*

3.3) **Normal Inverse Gaussian PIDE** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/3.3%20Pricing%20with%20the%20NIG%20Process.ipynb)
*(approximated jump-diffusion PIDE, Monte Carlo, Fourier inversion, properties of the Lévy measure)*

4.1) **Pricing with transaction costs** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/4.1%20Option%20pricing%20with%20transaction%20costs.ipynb)
*(Davis-Panas-Zariphopoulou model, singular control problem, HJB variational inequality, indifference pricing, binomial tree, performances)*

5.1) **Linear regression and Kalman filter** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/5.1%20Linear%20regression%20-%20Kalman%20filter.ipynb)
*(market data cleaning, Linear regression methods, Kalman filter design, choice of parameters)*

A.1) **Appendix: Linear equations** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/A.1%20Solution%20of%20linear%20equations.ipynb)
*(LU, Jacobi, Gauss-Seidel, SOR, Thomas)*
  
A.2) **Appendix: Code optimization** [nbviewer](https://nbviewer.ipython.org/github/cantaro86/Financial-Models-Numerical-Methods/blob/master/A.2%20Optimize%20and%20speed%20up%20the%20code.%20%28SOR%20algorithm%2C%20Cython%20and%20C%29.ipynb)
*(cython, C code)*

A.3) **Appendix: Review of Lévy processes theory** [github](https://github.com/cantaro86/Financial-Models-Numerical-Methods/blob/master/A.3%20Introduction%20to%20L%C3%A9vy%20processes%20and%20PIDEs.pdf)
*(basic and important definitions, derivation of the pricing PIDE)*



## How to run the notebooks 

You have two options:

1) Install [docker](https://www.docker.com/) following the instructions in [install link](https://docs.docker.com/install/) 

At this point, you just need to run the script ```docker_start_notebook.py``` and you are done.     
This script will download the data-science docker image [scipy-notebook](https://hub.docker.com/r/jupyter/scipy-notebook), that will be used every time you run the script (the script will take about 10-15 minutes to download the image, ONLY the first time). You can also download a different image by modifying the script. For a list of images see [here](https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html).

2) Clone the repository and open the notebooks using `jupyter-notebook`. 
If you are using an old version of python there can be compatibility problems.

```diff
- Cython code needs to be compiled!
```

If you are using the data science image, you can open the shell in the notebooks directory, and run the script 
```bash
python docker_start_notebook.py
```

after that, copy-paste the following code into the shell:

```bash 
docker exec -it Numeric_Finance bash
cd work/functions/cython
python setup.py build_ext --inplace
exit
``` 
(`Numeric_Finance` is the name of the docker container)

If you are not using docker, just copy in the shell the following:

```bash 
cd functions/cython
python setup.py build_ext --inplace
``` 


### Enjoy!