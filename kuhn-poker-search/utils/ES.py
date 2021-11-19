import numpy as np
np.set_printoptions(suppress=True)
import random
import os
import copy
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
torch.set_printoptions(sci_mode=False)
import multiprocessing as mp
from collections import Counter
from sklearn import metrics
import scipy.linalg as la
from numpy.linalg import eig

class DPP():
    """
    Attributes
    ----------
    A : PSD/Symmetric Kernel
    Usage:
    ------
    >>> from pydpp.dpp import DPP
    >>> import numpy as np
    >>> X = np.random.random((10,10))
    >>> dpp = DPP(X)
    >>> dpp.compute_kernel(kernel_type='rbf', sigma=0.4)
    >>> samples = dpp.sample()
    >>> ksamples = dpp.sample_k(5)
    """

    def __init__(self, X=None,  A=None, **kwargs):
        self.X = X
        if A:
            self.A = A

    def compute_kernel(self, kernel_type='cos-sim', kernel_func=None, *args, **kwargs):
        if kernel_func == None:
            if kernel_type == 'cos-sim':
                self.A = cosine_similarity(self.X )
            elif kernel_type == 'rbf':
                self.A = metrics.pairwise.rbf_kernel(self.X)
        else:
            self.A = kernel_func(self.X, **kwargs)


    def sample(self):

        if not hasattr(self,'A'):
            self.compute_kernel(kernel_type='cos-sim')

        eigen_vals, eigen_vec = eig(self.A)
        eigen_vals =np.real(eigen_vals)
        eigen_vec =np.real(eigen_vec)
        eigen_vec = eigen_vec.T
        N = self.A.shape[0]
        Z= list(range(N))

        probs = eigen_vals/(eigen_vals+1)
        jidx = np.array(np.random.rand(N)<=probs)    # set j in paper


        V = eigen_vec[jidx]           # Set of vectors V in paper
        num_v = len(V)


        Y = []
        while num_v>0:
            Pr = np.sum(V**2, 0)/np.sum(V**2)
            y_i=np.argmax(np.array(np.random.rand() <= np.cumsum(Pr), np.int32))

            # pdb.set_trace()
            Y.append(y_i)
            V =V.T
            ri = np.argmax(np.abs(V[y_i]) >0)
            V_r = V[:,ri]


            if num_v>0:
                try:
                    V = la.orth(V- np.outer(V_r, (V[y_i,:]/V_r[y_i]) ))
                except:
                    pdb.set_trace()

            V= V.T

            num_v-=1

        Y.sort()

        out = np.array(Y)

        return out

    def sample_k(self, k=5):

        if not hasattr(self,'A'):
            self.compute_kernel(kernel_type='cos-sim')

        eigen_vals, eigen_vec = eig(self.A)
        eigen_vals =np.real(eigen_vals)
        eigen_vec =np.real(eigen_vec)
        eigen_vec = eigen_vec.T
        N =self.A.shape[0]
        Z= list(range(N))

        if k==-1:
            probs = eigen_vals/(eigen_vals+1)
            jidx = np.array(np.random.rand(N)<=probs)    # set j in paper

        else:
            jidx = sample_k_eigenvecs(eigen_vals, k)

        V = eigen_vec[jidx]           # Set of vectors V in paper
        num_v = len(V)

        Y = []
        while num_v>0:
            Pr = np.sum(V**2, 0)/np.sum(V**2)
            y_i=np.argmax(np.array(np.random.rand() <= np.cumsum(Pr), np.int32))

            # pdb.set_trace()
            Y.append(y_i)
            # Z.remove(Z[y_i])
            V =V.T
            try:
                ri = np.argmax(np.abs(V[y_i]) >0)
            except:
                print("Error: Check: Matrix PSD/Sym")
                exit()
            V_r = V[:,ri]
            # nidx = list(range(ri)) + list(range(ri+1, len(V)))
            # V = V[nidx]

            if num_v>0:
                try:
                    V = la.orth(V- np.outer(V_r, (V[y_i,:]/V_r[y_i]) ))
                except:
                    print("Error in Orthogonalization: Check: Matrix PSD/Sym")
                    pdb.set_trace()

            V= V.T

            num_v-=1

        Y.sort()
        out = np.array(Y)

        return out

    def return_kernel(self):
      
        self.compute_kernel(kernel_type='rbf')
        return self.A

def elem_sympoly(lmbda, k):
    N = len(lmbda)
    E= np.zeros((k+1,N+1))
    E[0,:] =1
    for l in range(1,(k+1)):
        for n in range(1,(N+1)):
            E[l,n] = E[l,n-1] + lmbda[n-1]*E[l-1,n-1]
    return E

def sample_k_eigenvecs(lmbda, k):
    E = elem_sympoly(lmbda, k)
    i = len(lmbda)
    rem = k
    S = []
    while rem>0:
        if i==rem:
            marg = 1
        else:
            marg= lmbda[i-1] * E[rem-1,i-1]/E[rem,i]

        if np.random.rand()<marg: 
            S.append(i-1)
            rem-=1
        i-=1
    S= np.array(S)
    return S

def gen_dpp_noise(nb_workers, num_samples):
  A = np.random.normal(size=(num_samples, 10640))

  dpp = DPP(A)
  dpp.compute_kernel(kernel_type='rbf')
  idx = dpp.sample_k(nb_workers)

  A = A[idx]

  return A

def evo_update_batch(original_model, evo_std, evo_workers, num_samples, dpp=False):

  new_evo_models = []
  evo_noise = []
  if dpp == True:
    noise_block = gen_dpp_noise(evo_workers, num_samples)

  for i in range(evo_workers):
    interim_model = copy.deepcopy(original_model)

    interim_noise_dict = {}
    interim_model_dict = {}

    if dpp == False:
        for var_name, w in original_model.state_dict().items():
            interim_noise_dict[var_name] = np.random.randn(*w.shape)
            interim_model_dict[var_name] = w.clone() + evo_std*interim_noise_dict[var_name]
        
        for name, params in interim_model.named_parameters():
            params.data.copy_(torch.nn.Parameter(interim_model_dict[name])) 
    
    else:
        tmp_total = 0
        for var_name, w in original_model.state_dict().items():
            tmp_shape = w.shape
            tmp_num = w.numel()
            tmp_noise = noise_block[i][tmp_total:tmp_total + tmp_num]
            tmp_noise = tmp_noise.reshape(*tmp_shape)
            interim_noise_dict[var_name] = tmp_noise
            interim_model_dict[var_name] = w.clone() + evo_std*interim_noise_dict[var_name]
            tmp_total += tmp_num

        for name, params in interim_model.named_parameters():
            params.data.copy_(torch.nn.Parameter(interim_model_dict[name])) 
        
    new_evo_models.append(interim_model)
    evo_noise.append(interim_noise_dict)
  
  return new_evo_models, evo_noise

def evo_update_es(original_model, evo_std, sample_seed):

    r = np.random.RandomState(sample_seed)

    new_evo_models = []
    evo_noise = []

    interim_model = copy.deepcopy(original_model)

    interim_noise_dict = {}
    interim_model_dict = {}

    for var_name, w in original_model.state_dict().items():
        interim_noise_dict[var_name] = r.randn(*w.shape)
        interim_model_dict[var_name] = w.clone() + evo_std*interim_noise_dict[var_name]

    for name, params in interim_model.named_parameters():
        params.data.copy_(torch.nn.Parameter(interim_model_dict[name])) 
        
    new_evo_models.append(interim_model)
    evo_noise.append(interim_noise_dict)
  
    return new_evo_models, evo_noise


def evo_gradient_single(final_score, evo_noise):

  evo_noise.update((x, y*final_score) for x, y in evo_noise.items())
  return evo_noise
  
def evo_gradient_full(final_scores, evo_noise, evo_std):

  grads_list = []
  noise = copy.deepcopy(evo_noise)

  #for i, noise in enumerate(evo_noise):
  #  print(final_scores[i])
  #  noise.update((x, y*final_scores[i]) for x, y in noise.items())
  #  grads_list.append(noise)

  for i in range(len(evo_noise)):
    interim_grad = evo_gradient_single(final_scores[i], noise[i])
    grads_list.append(interim_grad)

  n = len(grads_list)
  gradient_sum = Counter()

  for grads in grads_list:
    gradient_sum.update(grads)

  gradient_sum = dict(gradient_sum)
  scaling_factor = 1 / (n*evo_std)

  gradient_sum.update((x, y*scaling_factor) for x,y in gradient_sum.items())

  return gradient_sum
