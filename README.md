# PDE Estimation

The code in this repository describes the procedure for estimating the form of a PDE that generates a set of data. 

To run a parameter estimation, choose a PDE and run `python3 run_inv.py $EQN` where `$EQN` is the equation of interest. For example, to run the wave equation run `python3 run_inv.py wave`.

#### Adding new equations

To define new equations, define a new dictionary with the following format:  

```
eqn = {'eqn_type':equation name,
        'fcn':exact function,
        'domain':dictionary with keys of variables and values of lists with intervals,
        'dictionary':string of dictionary functions,
        'err_vec': vector to determine accuracy of estimation}
```

For more information on the algorithms described, please check the following paper:

```
Hasan, A., Pereira, J. M., Ravier, R., Farsiu, S., & Tarokh, V. (2019). 
Learning Partial Differential Equations from Data Using Neural Networks. 
arXiv preprint arXiv:1910.10262.
```
