import numpy as np
import train as train
import pickle
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('equation', metavar='eqn', type=str,
                    help='which equation to run')
args = parser.parse_args()

params = {'n_epochs':50001,
          'n_points':1000,
          'n_grad_points':1000,
          'lambda_u':10,
          'lambda_f':1,
          'svd_init':500,
          'lambda_norm':1e-3,
          'width':50,
          'layers':4,
          'lr':0.002,
          'noise_levels': range(0,5),
          'n_repetitions':3}

wave = {'eqn_type':'wave',
        'fcn':'0.5*(exp(-(-t+x)**2)+exp(-(t+x)**2))+t',
        'domain':{'x':[-3,3],'t':[-3,3]},
        'dictionary':'(u_tt,u_xx,u_t,u_x,u,u**2,u*u_x)',#
        'err_vec':[1,-1,0,0,0,0,0]}#

burgers_inviscid = {'eqn_type':'burgers_i',
        'fcn':'(1+2*x*(t+1)+(1+4*x*(t+1))**.5)/(2*(t+1)**2)',
        'domain':{'x':[0,1],'t':[0,1]},
        'dictionary':'(u_tt,u_xx,u_t,u_x,u,u**2,u*u_x,u_xx**2)',
        'err_vec':[0,0,1,0,0,0,1,0]}

helmholtz = {'eqn_type':'helmholtz',
        'fcn':'sin(x)*y-sin(y)*x',
        'domain':{'x':[-np.pi,np.pi],'y':[-np.pi,np.pi]},
        'dictionary':'(u_yy,u_xx,u_yx,u_y,u_x,u,u**2,u*u_x,u*u_y)',
        'err_vec':[1,1,0,0,0,1,0,0,0]}

vortex = {'eqn_type':'vortex',
        'fcn':'exp(-1/2*(x-cos(t))**2-1/2*(y-sin(t))**2)',
        'domain':{'x':[-2,2],'y':[-2,2],'t':[0,2]},
        'dictionary':'(u_t,u_x,u_y,x*u_x,y*u_x,x*u_y,y*u_y,u)',
        'err_vec':[-1,0,0,0,1,-1,0,0]}

oldkdv = {'eqn_type':'KdV',
        'fcn':'(1/2)*(1/(cosh(1/2*(x-1*t))**2))+(5/2)*(1/(cosh(sqrt(5)/2*(x-5*t))**2))+4*(1/(cosh(sqrt(8)/2*(x-8*t))**2))',
        'domain':{'x':[5,10],'t':[0,10]},
        'dictionary':'(u_ttt,u_xxx,u_tt,u_xx,u_t,u_x,u,u**2,u_t*u,u_x*u)',
        'err_vec':[0,1,0,0,1,0,0,0,0,6]}

kdv = {'eqn_type':'KdV',
        'fcn':'(5/2)*(1/(cosh(sqrt(5)/2*(x+9-5*t))**2))+(8/2)*(1/(cosh(sqrt(8)/2*(x+2-8*t))**2))',
        'domain':{'x':[0,4],'t':[0,3]},
        'dictionary':'(u_xxx,u_tt,u_xx,u_t,u_x,u,u*u_x,u_x**2)',
        'err_vec':[1,0,0,1,0,0,6,0]}

hjb = {'eqn_type':'hjb',
        'fcn':'x**2/(1+4*t)+1/2*log(1+4*t)',
        'domain':{'x':[-2,2],'t':[0,2]},
        'dictionary':'(u_tt,u_xx,u_t,u_x,u,u**2,u*u_x,u_x**2)',
        'err_vec':[0,-1,1,0,0,0,0,1]}

ode = {'eqn_type':'ode',
        'fcn':'exp(x)',
        'domain':{'x':[0,1]},
        'dictionary':'(u_x,u)',
        'err_vec':[1,-1]}

exec("eqn = {}".format(args.equation))

name = eqn['eqn_type'] + '_' +str(params['noise_levels'][1])+'_multiple_'+str(params['n_repetitions'])+'.pkl'
stats = train.train(**eqn,**params)
out_dict = {'all_stats':stats,'eqn_type':eqn['eqn_type'],'noise_levels':params['noise_levels'],'params':params,'eqn':eqn}

with open(name,'wb') as f:
    pickle.dump(stats,f)
