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

device = 'cuda:0'

def run_psro(torch_pop, environment, method, model=None, iters=10, inner_lr=0.75, inner_iters=100, exp_inner_lr=0.75, exp_iters=100, implicit=True):
    print(method)

    if method == 'nash':
        meta_game = torch_pop.get_metagame(numpy=True)
        meta_nash = fictitious_play(payoffs=meta_game, iters=1000)[0][-1]
        meta_nash = torch.Tensor(meta_nash).to(device)
        exp = torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
        exps = []
        exps.append(exp)
        
        for psro_iters in range(iters):
            torch_pop.psro_popn_update(meta_nash, inner_iters, inner_lr, lam=0, norm_break_val=0.001, implicit=implicit)
            torch_pop.pop[-1].end_train()
            meta_game = torch_pop.get_metagame(numpy=True)
            meta_nash = fictitious_play(payoffs=meta_game, iters=1000)[0][-1]
            meta_nash = torch.Tensor(meta_nash).to(device)
            exp = torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
            exps.append(exp)

    elif method == 'auto': 
        model.to(device)
        meta_game = torch_pop.get_metagame().to(device)
        meta_nash = model(meta_game)[0]
        exp = torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()#direct exploitability can get
        exps = []
        exps.append(exp)

        for psro_iters in range(iters):
            torch_pop.psro_popn_update(meta_nash, inner_iters, inner_lr, lam=0, norm_break_val=0.001, implicit=implicit)
            torch_pop.pop[-1].end_train()
            meta_game = torch_pop.get_metagame().to(device)
            meta_nash = model(meta_game)[0]
            exp = torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()#direct exploitability can get
            exps.append(exp)
    
    elif method == 'uniform':
        meta_game = torch_pop.get_metagame(numpy=True)
        meta_nash = torch.ones(meta_game.shape[-1]) / meta_game.shape[-1]
        meta_nash = torch.Tensor(meta_nash).to(device)
        exp =torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
        exps = []
        exps.append(exp)
        
        for psro_iters in range(iters):
            torch_pop.psro_popn_update(meta_nash, inner_iters, inner_lr, lam=0, norm_break_val=0.001, implicit=implicit)
            torch_pop.pop[-1].end_train()
            meta_game = torch_pop.get_metagame(numpy=True)
            meta_nash = torch.ones(meta_game.shape[-1]) / meta_game.shape[-1]
            meta_nash = torch.Tensor(meta_nash).to(device)
            exp = torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
            exps.append(exp)
      
            
    elif method == 'self-play':
        meta_game = torch_pop.get_metagame(numpy=True)
        meta_nash = torch.zeros(meta_game.shape[-1])
        meta_nash[-1] = 1.
        meta_nash = torch.Tensor(meta_nash).to(device)
        exp =torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
        exps = []
        exps.append(exp)
        
        for psro_iters in range(iters):
            torch_pop.psro_popn_update(meta_nash, inner_iters, inner_lr, lam=0, norm_break_val=0.001, implicit=implicit)
            torch_pop.pop[-1].end_train()
            meta_game = torch_pop.get_metagame(numpy=True)
            meta_nash = torch.zeros(meta_game.shape[-1])
            meta_nash[-1] = 1.
            exp = torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
            exps.append(exp)
            
    elif method == 'psro_rn':
        meta_game = torch_pop.get_metagame(numpy=True)
        meta_nash = fictitious_play(payoffs=meta_game, iters=1000)[0][-1]
        exp = torch_pop.get_exploitability(torch.Tensor(meta_nash).to(device), exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
        exps = []
        exps.append(exp)

        for psro_iter in range(iters):
            print(psro_iter)
            iter_pop_size = torch_pop.pop_size
            for agent in range(iter_pop_size):
                if meta_nash[agent] > 0.05:
                    agent_payoffs = meta_game[agent]
                    agent_payoffs[agent_payoffs > 0] = 1
                    agent_payoffs[agent_payoffs < 0] = 0
                    weights = np.multiply(agent_payoffs, meta_nash)
                    weights /= weights.sum()       
                    torch_pop.psro_popn_update(torch.Tensor(weights).to(device), inner_iters, inner_lr, lam=0, norm_break_val=0.001, implicit=implicit)
            meta_game = torch_pop.get_metagame(numpy=True)
            meta_nash = fictitious_play(payoffs=meta_game, iters=1000)[0][-1]
        for psro_iter in range(iters):
            meta_game = torch_pop.get_metagame(numpy=True)
            meta_game = meta_game[:(psro_iter+1), :(psro_iter+1)]
            meta_nash = fictitious_play(payoffs=meta_game, iters=1000)[0][-1]
            meta_nash = torch.Tensor(meta_nash).to(device)
            exp = torch_pop.get_exploitability(meta_nash, exp_iters, norm_break_val=0.001, lr=exp_inner_lr, test=True, implicit=implicit).item()
            exps.append(exp)
    
    return exps