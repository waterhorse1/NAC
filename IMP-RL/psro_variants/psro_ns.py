import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import time
import os
from scipy import stats
import pickle
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(0)
from numpy.random import RandomState
import scipy.linalg as la

import copy
from scipy.linalg import circulant
torch.set_printoptions(sci_mode=False)
from utils.utils import fictitious_play

device = 'cpu'

def run_psro(torch_pop, environment, method, model=None, iters=5, lr=.2, train_iters=10):
    print(method)

    if method == 'nash':
        meta_game = torch_pop.get_metagame(numpy=True)
        print(meta_game.shape)
        mn1, mn2 = fictitious_play(payoffs=meta_game)
        meta_nash1 = mn1[-1]
        meta_nash2 = mn2[-1]
        print(meta_nash1.shape)
        meta_nash1 = torch.Tensor(meta_nash1).to(device)
        meta_nash2 = torch.Tensor(meta_nash2).to(device)
        exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()
        exps = []
        exps.append(exp)
        
        for psro_iters in range(iters):
            print(psro_iters)
            torch_pop.psro_popn_update(meta_nash1, meta_nash2, 10, 10)
            torch_pop.pop1[-1].end_train()
            torch_pop.pop2[-1].end_train()
            meta_game = torch_pop.get_metagame(numpy=True)
            mn1, mn2 = fictitious_play(payoffs=meta_game)
            meta_nash1 = mn1[-1]
            meta_nash2 = mn2[-1]
            meta_nash1 = torch.tensor(meta_nash1).to(device)
            meta_nash2 = torch.tensor(meta_nash2).to(device)
            exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()
            exps.append(exp)

    elif method == 'auto': 
        model.to(device)
        meta_game = torch_pop.get_metagame().to(device)
        meta_nash1 = model(meta_game)[0]
        meta_nash2 = model(-torch.transpose(meta_game, -2, -1))[0]
        exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()#direct exploitability can get
        exps = []
        exps.append(exp)

        for psro_iters in range(iters):
            print(psro_iters)
            torch_pop.psro_popn_update(meta_nash1, meta_nash2, 10, 10)
            torch_pop.pop1[-1].end_train()
            torch_pop.pop2[-1].end_train()
            meta_game = torch_pop.get_metagame().to(device)
            meta_nash1 = model(meta_game)[0]
            meta_nash2 = model(-torch.transpose(meta_game, -2, -1))[0]
            exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()#direct exploitability can get
            exps.append(exp)
    
    elif method == 'uniform':
        meta_game = torch_pop.get_metagame(numpy=True)
        meta_nash1 = torch.ones(meta_game.shape[-1]) / meta_game.shape[-1]
        meta_nash2 = torch.ones(meta_game.shape[-1]) / meta_game.shape[-1]
        meta_nash1 = torch.Tensor(meta_nash1).to(device)
        meta_nash2 = torch.Tensor(meta_nash2).to(device)
        exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()
        exps = []
        exps.append(exp)
        
        for psro_iters in range(iters):
            print(psro_iters)
            torch_pop.psro_popn_update(meta_nash1, meta_nash2, 10, 10)
            torch_pop.pop1[-1].end_train()
            torch_pop.pop2[-1].end_train()
            meta_game = torch_pop.get_metagame(numpy=True)
            meta_nash1 = torch.ones(meta_game.shape[-1]) / meta_game.shape[-1]
            meta_nash2 = torch.ones(meta_game.shape[-1]) / meta_game.shape[-1]
            meta_nash1 = torch.tensor(meta_nash1).to(device)
            meta_nash2 = torch.tensor(meta_nash2).to(device)
            exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()
            exps.append(exp)
            
    elif method == 'self-play':
        exps = []
        meta_game = torch_pop.get_metagame().to(device)
        meta_nash1 = np.zeros(meta_game[0][0].shape[0])
        meta_nash1[-1] = 1.
        meta_nash2 = np.zeros(meta_game[0][0].shape[0])
        meta_nash2[-1] = 1.
        meta_nash1 = torch.tensor(meta_nash1).to(device)
        meta_nash2 = torch.tensor(meta_nash2).to(device)
        exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()
        exps.append(exp)

        for psro_iter in range(iters):
            print(f'psro iter {psro_iter}')
            torch_pop.psro_popn_update(meta_nash1, meta_nash2, 10, 10)
            meta_game = torch_pop.get_metagame().to(device)
            meta_nash1 = np.zeros(meta_game[0][0].shape[0])
            meta_nash1[-1] = 1.
            meta_nash2 = np.zeros(meta_game[0][0].shape[0])
            meta_nash2[-1] = 1.   
            meta_nash1 = torch.tensor(meta_nash1).to(device)
            meta_nash2 = torch.tensor(meta_nash2).to(device)
            exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, 20, 10).item()
            exps.append(exp)
            

    return exps