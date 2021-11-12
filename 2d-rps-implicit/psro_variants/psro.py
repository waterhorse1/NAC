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
from environments.gos import torchAgent, TorchPop_gos, sample_gos_games, gen_gos_payoffs
from environments.gmm_rps import GMMAgent, TorchPop_rps, sample_rps_games
from environments.blotto import DLottoAgent, TorchPop_blotto, sample_blotto_games
from utils.utils import fictitious_play

device = 'cuda:0'

def run_psro(torch_pop, environment, model=None, iters=5, lr=.2, train_iters=10, seed=0):

    if model == None:
        device = 'cuda:0'
        meta_game = torch_pop.get_metagame(numpy=True)
        meta_nash = fictitious_play(payoffs=meta_game, iters=1000)[0][-1]
        meta_nash = torch.Tensor(meta_nash).to(device)
        exp = torch_pop.get_exploitability(meta_nash,  100, 10.0, test=True).item()
        exps = []
        exps.append(exp)
        
        for psro_iters in range(iters):
            torch_pop.psro_popn_update(meta_nash, train_iters, lr)
            torch_pop.pop[-1].end_train()
            meta_game = torch_pop.get_metagame(numpy=True)
            meta_nash = fictitious_play(payoffs=meta_game, iters=1000)[0][-1]
            meta_nash = torch.Tensor(meta_nash).to(device)
            exp = torch_pop.get_exploitability(meta_nash, 100, 10.0, test=True).item()
            exps.append(exp)

    else: 
        device = 'cuda:0'
        model.to(device)
        meta_game = torch_pop.get_metagame().to(device)
        meta_nash = model(meta_game)[0]
        exp = torch_pop.get_exploitability(meta_nash, 100, 10.0, test=True).item()#direct exploitability can get
        exps = []
        exps.append(exp)

        for psro_iters in range(iters):
            torch_pop.psro_popn_update(meta_nash, train_iters, lr)
            torch_pop.pop[-1].end_train()
            meta_game = torch_pop.get_metagame().to(device)
            meta_nash = model(meta_game)[0]
            exp = torch_pop.get_exploitability(meta_nash, 100, 10.0, test=True).item()#direct exploitability can get
            exps.append(exp)

    return exps