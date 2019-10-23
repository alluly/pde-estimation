import math
import itertools
import pickle

import torch
from torch.nn import init
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
from torch.autograd import Variable
import torch.optim as optim
import torch.distributions as tdist

import numpy as np

from nets import MLP

import sympy

def torchvec2str(vec):
    out = '['
    for v in vec.tolist():
        out = out + '%1.4f, '%v
    out = out + '\b\b]'
    return out

def getdictionary(func,vardict,dictionary):
# This function interprets a dictionary string with derivatives defined
# by subscripts (for example u_x is derivative of u in x) and returns the
# dictionary matrix
    
    u = func(torch.cat(list(vardict.values()),dim=1))
    fundict = vardict.copy()
    fundict['u'] = u
    
    fundict_= {}
    
    flag = True
    while flag:
        try:
            M =  torch.cat(eval(dictionary,fundict),dim=1)
            flag = False
        except NameError as error:
            key = str(error).split("'")[1]
            for i in range(2,len(key)):
                keyi = key[:i+1]
                if not(keyi in fundict.keys()):
                    var = vardict[keyi[i]]
                    if i==2:
                        fun = u
                        fundict_[keyi] = grad(fun,var,create_graph=True,\
                                   grad_outputs=torch.ones_like(fun))
                        fundict[keyi] = fundict_[keyi][0]
                    else:
                        fun = fundict_[keyi[:-1]]
                        fundict_[keyi] = grad(fun,var,create_graph=True,\
                                   grad_outputs=torch.ones_like(fun[0]))
                        fundict[keyi] = fundict_[keyi][0]
    
    return M


def train(eqn_type,fcn,domain,dictionary,err_vec,**params):
    
    n_epochs = params.setdefault('n_epochs',10000)
    n_points = params.setdefault('n_points',1000)
    n_grad_points = params.setdefault('n_grad_points',1000)
    lambda_u = params.setdefault('lambda_u',1)
    lambda_f_ = params.setdefault('lambda_f',1)
    lambda_norm_ = params.setdefault('lambda_norm',5e-6)
    svd_init = params.setdefault('svd_init',500)
    width = params.setdefault('width',50)
    layers = params.setdefault('layers',4)
    lr = params.setdefault('lr',0.002)
    noise_levels = params.setdefault('noise_levels', list(range(10,0,-1)))
    n_repetitions = params.setdefault('n_repetitions',4)
    
    normal = tdist.Normal(0,1)

    all_stats = []

    for _ in range(n_repetitions):
        for idx_noise in noise_levels:
            stat_dict = {'noise':idx_noise,'epoch':[],'coeff_err':[],'loss_u':[],'loss_f':[],'norm_loss':[]}
            
            lambda_f = 0
            lambda_norm = 0

            in_size = len(domain)
            n_params = len(err_vec)
            u_exact = sympy.sympify(fcn)

            grad_inputs = []
            sampled_inputs = []

            x = sympy.symbols([x for x in domain.keys()])

            for key in domain.keys():
                min_d = domain[key][0]
                max_d = domain[key][1]
                sampled_inputs.append(((max_d-min_d)*torch.rand(n_points)+min_d).double().cuda(0).unsqueeze(1))
                grad_inputs.append(Variable(((max_d-min_d)*torch.rand(n_grad_points)+min_d).double().cuda(0).unsqueeze(1),requires_grad=True))

            u_exact = sympy.lambdify(x,sympy.sympify(fcn),'numpy')
            u = u_exact(*[i.cpu() for i in sampled_inputs])

            c_truth = torch.from_numpy(np.array(err_vec)).double().cuda(0)

            print('-------------')
            u_hat = MLP(in_size,width,layers,1).cuda(0)
            u_hat.train()
            u_hat.double()

            sv_test = torch.Tensor()
            
            noise = 10**-(idx_noise/2)

            sampled_inputs = [si.cuda(0) for si in sampled_inputs]
            
            print('% noise:',noise)

            u = u + noise*torch.sqrt(torch.var(u))*normal.sample(u.size()).to(u.device).double()
            u = u.cuda(0)

            print('width ' + str(width))
            print('depth ' + str(layers))

            total_params = sum(p.numel() for p in u_hat.parameters())
            print('total number of parameters: ' + str(total_params))

            comb = nn.Parameter(torch.randn(n_params,requires_grad=True,device="cuda",dtype=torch.double))
            params = [{'params': u_hat.parameters(), 'lr': lr},
                      {'params': comb, 'lr': .02}]
            
            optimizer = optim.Adam(params)
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,.9998)
            
            # Run optimization procedure
            for epoch in range(n_epochs):

                if epoch == svd_init:
                    lambda_f = lambda_f_
                    lambda_norm = lambda_norm_

                if epoch == 2*svd_init:
                    diff_svd = torch.sqrt(1-torch.abs(torch.dot(comb,sv_test)/torch.norm(sv_test)/torch.norm(comb)))
                    if diff_svd > 0.5:
                        print('TERMINATING SIMULATION DUE TO DIVERGENCE. ERROR {:.4f}'.format(diff_svd))
                        break
        
                def closure():
                    optimizer.zero_grad()

                    out_uhat = u_hat(torch.cat(sampled_inputs,dim=1).double())
                    true_loss_u = F.mse_loss(out_uhat,u.double()).cuda(0)
                    loss_u = lambda_u*(true_loss_u+1e-4)**(.5)
                    
                    vardict = dict(zip(domain.keys(),grad_inputs))
                    M = getdictionary(u_hat, vardict, dictionary)
                    
                    if epoch == svd_init:
                        su,s,sv = torch.svd(M)
                        comb.data = sv[:,-1].data
                    
                    comb_norm = comb/torch.norm(comb)

                    comb.data = comb_norm.data
                    
                    lf = torch.sum(M*comb_norm,dim=1)
 
                    true_loss_f = torch.norm(lf)**2 / n_grad_points
                    loss_f = lambda_f*true_loss_f
                    loss_norm = lambda_norm*torch.norm(comb_norm,p=1)
                    loss = loss_u*(1 + loss_f + loss_norm) 
                    loss.backward(retain_graph=False)

                    if epoch//2 % 50 ==0: 
                        
                        su,s,sv = torch.svd(M)
                        sv_test.data = sv[:,-1].data


                        err_vec = torch.sqrt(1-torch.abs(torch.dot(comb_norm,c_truth)/torch.norm(c_truth)/torch.norm(comb_norm))) 
                        err_svd = torch.sqrt(1-torch.abs(torch.dot(sv[:,-1],c_truth)/torch.norm(c_truth)/torch.norm(sv[:,-1])))
                        err_svd2 = torch.sqrt(1-torch.abs(torch.dot(sv[:,-2],c_truth)/torch.norm(c_truth)/torch.norm(sv[:,-2])))
                        err_law  = err_vec
                        if eqn_type == 'ode':
                            err_law = (comb_norm[0]/comb_norm[1] + 1)**2

                        print('\n###### Epoch {:.0f} ######'.format(epoch))
                        print('Error in law (svd) {:e}'.format(err_svd.item()))
                        print(torchvec2str(sv[:,-1]))
                        print('Error in law (svd2) {:e}'.format(err_svd2.item()))
                        print(torchvec2str(sv[:,-2]))

                        print('Eigenvalues ' + torchvec2str(s))
                        for pg in optimizer.param_groups:
                            print('Learning rate:',pg['lr'])

                        print('Loss u ' + str(loss_u.item()))
                        print('Loss f ' + str(loss_f.item()))
                        print('Error in law ' + str(err_law.item()))
                        
                        if epoch >= svd_init:
                            print('Error in law (vec) {:e}'.format(err_vec.item()))
                            print(torchvec2str(comb_norm))
                            print('Loss f ' + str(true_loss_f.item()))
                            print('Loss norm ' + str(loss_norm.item()))

                        stat_dict['epoch'].append(epoch)
                        stat_dict['coeff_err'].append(err_law.item())
                        stat_dict['loss_u'].append(loss_u.item())
                        stat_dict['loss_f'].append(loss_f.item())
                    return loss
                optimizer.step(closure)
                scheduler.step()
            all_stats.append(stat_dict)
            print('----------')
    return all_stats

