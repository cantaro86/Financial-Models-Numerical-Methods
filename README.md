Financial-Models-Numerical-Methods
==================================


## DO NOT READ IT!! DO NOT USE IT!!
## It is not ready yet!


But... if you still want to have a look, notebooks 2.1, A.1 and A.2 are complete. A.3 is almost complete.   

You just need to install docker and run the script ```docker_start_notebook.py```.
If you open the notebook from another docker jupyther image, or from you local jupyter, be careful that probably some Cython code in the notebook A.2 will not work.



```bash 
docker exec -it Numeric_Finance bash
cd work/functions/cython
python setup.py build_ext --inplace
``` 



