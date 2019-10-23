import torch
import torch.nn as nn
import torch.nn.functional as F

#activation functions
class quadratic(nn.Module):
    def __init__(self):
        super(quadratic,self).__init__()

    def forward(self,x):
        return x**2

class cos(nn.Module):
    def __init__(self):
        super(cos,self).__init__()

    def forward(self,x):
        return torch.cos(x)

class relu2(nn.Module):
    def __init__(self):
        super(relu2,self).__init__()

    def forward(self,x):
        return F.relu(x)**2

class mod_softplus(nn.Module):
    def __init__(self):
        super(mod_softplus,self).__init__()

    def forward(self,x):
        return F.softplus(x) + x/2 - torch.log(torch.ones(1)*2).to(device=x.device)

class mod_softplus2(nn.Module):
    def __init__(self):
        super(mod_softplus2,self).__init__()

    def forward(self,x,d):
        return d*(1+d)*(2*F.softplus(x) - x  - 2*torch.log(torch.ones(1)*2).to(device=x.device))

class mod_softplus3(nn.Module):
    def __init__(self):
        super(mod_softplus3,self).__init__()

    def forward(self,x):
        return F.relu(x) + F.softplus(-torch.abs(x)) 

class Shallow(nn.Module):
    def __init__(self,input_size,out_size):
        super(Shallow, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_size,input_size),quadratic(),nn.Linear(input_size,out_size))

    def forward(self,x):
        return self.net(x)


class SMLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size,hidden_size)
        self.tanh = nn.Tanh()
        mid_list = [nn.Linear(hidden_size,hidden_size),nn.Tanh()]
        for i in range(layers-1):
           mid_list += [nn.Linear(hidden_size,hidden_size),nn.Tanh()]
        self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self,x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.mid(out)
        out = self.out(out)
        return out

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, layers, out_size, n=10, proj=False, bn=False):
        super(MLP, self).__init__()
        fn = nn.Softplus(beta=1)

        self.a = nn.Parameter(torch.ones(1),requires_grad=True)
        self.lips = nn.Parameter(torch.ones(1),requires_grad=True)
        self.fc1 = ScaledLinear(input_size,hidden_size,self.a,sigma=fn,bias=True,proj=proj,lips=self.lips,bn=bn)#nn.Linear(input_size,hidden_size)
        self.layers = layers
        if layers > 0:
            mid_list = [ScaledLinear(hidden_size,hidden_size,self.a,sigma=fn,bias=False,proj=proj,lips=self.lips,bn=bn) for _ in range(layers)]
            self.mid = nn.Sequential(*mid_list)
        self.out = nn.Linear(hidden_size, out_size,bias=True)
        

    def forward(self,x):
        out = self.fc1(x)
        if self.layers > 0:
            out = self.mid(out)
        out = self.out(out)
        return out

class L2Proj(nn.Module):
    def __init__ (self):
        super(L2Proj, self).__init__()
    def forward(self, x):
        if torch.norm(x) > 1:
            return x/torch.norm(x)
        else:
            return x

class ScaledLinear(nn.Module):
    def __init__(self, input_size, output_size,a,n=10,sigma=nn.Tanh(),bias=False,proj=False,lips=0,bn=False):
        super(ScaledLinear, self).__init__()
        self.n = n
        self.fc = nn.Linear(input_size,output_size,bias=bias)
        self.a = a
        self.sigma = sigma
        self.proj = proj
        self.l2proj = L2Proj()
        self.lips = lips
        self.bn = bn 
        self.batch_norm = nn.BatchNorm1d(output_size)

    def forward(self,x):
        out = self.fc(x)
        if type(self.sigma) == mod_softplus2:
            out = self.sigma(out,self.a)
        else:
            out = self.sigma(out)
        if self.proj:
            out = self.lips*self.l2proj(out)
        if self.bn:
            out = self.batch_norm(out)
        return out

class MLPPartial(nn.Module):
    def __init__(self,initval):
        super(MLPPartial, self).__init__()
        self.net = nn.Sequential(PartialActivationLayer(2,4,2),PartialActivationLayer(4,4,2),PartialActivationLayer(4,1,0,initval))

    def forward(self,x):
        return self.net(x)

def init_weights(m):
    if type(m) == nn.Linear:
        for i in m.parameters():
            if i.size(0) == 4 and i.size(1) == 2:
                i[2,0] = nn.Parameter(torch.ones(1),requires_grad=True)
                i[3,0] = nn.Parameter(torch.zeros(1))

                i[2,1] = nn.Parameter(torch.zeros(1))
                i[3,1] = nn.Parameter(torch.ones(1))
            elif i.size(0) == 4 and i.size(1) == 4:
                for x in range(i.size(0)):
                    for y in range(i.size(1)):
                        if x < 2:
                            i[x,y] = nn.Parameter(torch.zeros(1))
                        elif x > 1:
                            if x == y:
                                i[x,y] = nn.Parameter(torch.ones(1))
            elif i.size(0) == 4 and i.size(1) == 1: 
                print(i)
                i[2,0] = nn.Parameter(torch.ones(1)) 
                i[3,0] = nn.Parameter(torch.ones(1))
            weight = i
        m.weight = nn.Parameter(weight)

class PartialActivationLayer(nn.Module):
    def __init__(self, input_size, output_size,num_active,initval=None,sigma=nn.Softplus()):
        super(PartialActivationLayer,self).__init__()
        self.sigma = sigma
        self.num_active = num_active
        self.fc = nn.Linear(input_size, output_size,bias=False)

        #initialization
        for i in self.fc.parameters(): 
            if i.size(0) == 4 and i.size(1) == 2:
                i[2,0] = nn.Parameter(torch.ones(1),requires_grad=True)
                i[3,0] = nn.Parameter(torch.zeros(1))

                i[2,1] = nn.Parameter(torch.zeros(1))
                i[3,1] = nn.Parameter(torch.ones(1))
            elif i.size(0) == 4 and i.size(1) == 4:
                for x in range(i.size(0)):
                    for y in range(i.size(1)):
                        if x < 2:
                            i[x,y] = nn.Parameter(torch.zeros(1))
                        elif x > 1:
                            if x == y:
                                i[x,y] = nn.Parameter(torch.ones(1))
            elif i.size(0) == 4 and i.size(1) == 1: 
                if initval is not None:
                    i[2,0] = initval[0] 
                    i[3,0] = initval[1]  
            weight = i
        self.fc.weight = nn.Parameter(weight)

    def forward(self,x):
        out = self.fc(x)
        if self.num_active > 0:
            out = torch.cat((out[:,self.num_active:],self.sigma(out[:,:self.num_active])),dim=1)
        return out
