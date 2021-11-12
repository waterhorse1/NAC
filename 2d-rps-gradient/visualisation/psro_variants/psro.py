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
from environments.rps_enivronment import GMMAgent_nash, TorchPop_nash
from utils.utils import fictitious_play

device = 'cpu'

def gradient_loss_update(torch_pop, k, train_iters=10, lr=0.1):

    # We compute metagame M and then L in a differentiable way
    # We compute expected payoff of agent k-1 against aggregated strat

    # Make strategy k trainable
    torch_pop.pop[k].x.requires_grad = True

    # Optimiser
    optimiser = optim.SGD(torch_pop.pop[k].parameters(), lr=lr)

    for iters in range(train_iters):

        # Get metagame and metastrat
        M = torch_pop.get_metagame(k=k+1)
        meta_nash = fictitious_play(payoffs=M.detach().cpu().clone().numpy()[:k, :k], iters=1000)[0][-1]

        # Compute the expected return given that enemy plays agg_strat (using :k first strats)
        exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[k], meta_nash, k)
        #print(exp_payoff)
        #print(f'On iter {iter} the expected payoff is {exp_payoff}')

        # Loss
        loss = -(exp_payoff)

        # Optimise !
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

    torch_pop.pop_hist[k].append(torch_pop.pop[k].x.detach().cpu().clone().numpy())
    # Make strategy k non-trainable
    torch_pop.pop[k].x.detach()
    return exp_payoff.item()

def run_psro(torch_pop, model=None, iters=5, num_mode=3, lr=.2, train_iters=10, seed=0):

    #if model == None:
    #    device = 'cpu'
        # Compute initial exploitability and init stuff
    #    metagame = torch_pop.get_metagame(numpy=True)
    #    metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
    #    exps = []

     #   for i in range(iters):
     #       k = torch_pop.pop_size - 1

            # Diverse PSRO
     #       exp_payoff = gradient_loss_update(torch_pop, k, train_iters=train_iters, lr=lr)

    #        metagame = torch_pop.get_metagame(numpy=True)
     #       metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
     #       exp = torch_pop.get_exploitability(metanash, 1.0, nb_iters=train_iters)
     #       exps.append(exp)

     #       torch_pop.add_new()
    metanash_list = []
    if model == None:
        device = 'cuda:0'
        # Compute initial exploitability and init stuff
        metagame = torch_pop.get_metagame(numpy=True)
        metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
        metanash_list.append(metanash)
        metanash = torch.Tensor(metanash)
        torch_pop.add_new()
        torch_pop.pop[-1].x.requires_grad = True
        exps = []

        for psro_iters in range(iters):

            for i in range(train_iters):

                # Diverse PSRO
                exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], metanash, len(metanash))
                loss = -(exp_payoff)

                psro_grad = torch.autograd.grad(loss, torch_pop.pop[-1].x, create_graph=False)
                torch_pop.pop[-1].x = torch_pop.pop[-1].x - lr * psro_grad[0]

                torch_pop.pop[-1].x.detach()

            metagame = torch_pop.get_metagame(numpy=True)
            metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
            metanash_list.append(metanash)
            metanash = torch.Tensor(metanash).to(device)
            exp = torch_pop.get_exploitability_test(metanash, 50, 1.0).item()
            exps.append(exp)

            torch_pop.add_new()
            torch_pop.pop[-1].x.requires_grad = True

    else: 
        device = 'cuda:0'
        # Compute initial exploitability and init stuff
        model.to(device)
        metagame = torch_pop.get_metagame()
        metagame = metagame[None,None,].to(device)
        metanash = model(metagame)[0]
        metanash_list.append(metanash.detach().cpu().numpy())
        exps = []
        torch_pop.add_new()
        torch_pop.pop[-1].x.requires_grad = True

        for psro_iters in range(iters):

            for i in range(train_iters):

                # Diverse PSRO
                exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], metanash, len(metanash))
                loss = -(exp_payoff)

                psro_grad = torch.autograd.grad(loss, torch_pop.pop[-1].x, create_graph=False)
                torch_pop.pop[-1].x = torch_pop.pop[-1].x - lr * psro_grad[0]

            torch_pop.pop[-1].x.detach()

            metagame = torch_pop.get_metagame()
            #print(metagame)
            metagame = metagame[None,None,].to(device)
            metanash = model(metagame)[0]
            metanash_list.append(metanash.detach().cpu().numpy())
            exp = torch_pop.get_exploitability_test(metanash, 100, 0.1).item()
            exps.append(exp)

            torch_pop.add_new()
            torch_pop.pop[-1].x.requires_grad = True

    return torch_pop, metanash_list#exps