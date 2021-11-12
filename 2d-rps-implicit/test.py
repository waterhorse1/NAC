import numpy as np
np.set_printoptions(suppress=True)
import gym
from typing import List, Dict
from gym.spaces import Discrete, Box
import random
import itertools
import argparse
from scipy.stats import entropy
import matplotlib
import matplotlib.pyplot as plt
import math
import time
import os
from scipy import stats
import glob
import copy
from tqdm import tqdm
from scipy.special import softmax
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
torch.set_printoptions(sci_mode=False)
import multiprocessing as mp
from collections import Counter
import pickle
import json
import scipy.linalg as la
from numpy.linalg import eig
import pickle

from environments.gmm_rps import GMMAgent, TorchPop_rps, sample_rps_games
from utils.meta_solver import meta_solver_small, meta_solver_large, meta_solver_gru, meta_solver_gru_temp, meta_solver_mlp_large, meta_solver_gru_large
from utils.utils import load_trained_model, gen_labels, fictitious_play, get_agent_nn_total_size
from psro_variants.psro_ns import run_psro

parser = argparse.ArgumentParser(description='gos')
parser.add_argument('--environment', type=str, default='rps')
parser.add_argument('--job_number', type=int, default=10)
parser.add_argument('--num_test_games', type=int, default=20)
parser.add_argument('--psro_iters', type=int, default=10)
parser.add_argument('--gos_dim', type=int, default=200)
parser.add_argument('--num_customer', type=int, default=9)
parser.add_argument('--num_position', type=int, default=500)
parser.add_argument('--num_mode', type=int, default=7)
parser.add_argument('--type_ag', type=str, default='logits')
parser.add_argument('--agent_nn_input_size', type=int, default=2)

args = parser.parse_args(args=[])

PATH_RESULTS = os.path.join('results' , args.environment + '_job_' + str(args.job_number))
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)

device = 'cpu'
 

def run_experiment(mod_list, method_list, unique_params, num_unique, num_test_games, psro_iters):

    num_mod = len(mod_list)
    set_seed = 123123

    game_args_dict = {
        'gos_dim' : args.gos_dim,
        'num_mode' : args.num_mode,
        'nb_customers' : args.num_customer,
        'nb_positions' : args.num_position,
        'type_ag' : args.type_ag, 
        'nn_input_size' : args.agent_nn_input_size,
        'nn_total_size' : get_agent_nn_total_size(args.agent_nn_input_size),
        'testing' : False,
        'nf_payoffs' : None
    }

    POPN = eval(f'TorchPop_{args.environment}')
    
    torch.manual_seed(set_seed)
    np.random.seed(set_seed)
    
    game_list = [[POPN(game_args_dict, device, seed=(set_seed + i), test=True) for i in range(num_test_games)] for _ in range(num_mod)]

    model_exps = [[] for _ in range(num_mod)]

    for i, mod in enumerate(mod_list):
        print('model {}'.format(i))
        for j in tqdm(range(num_test_games)):
            game = game_list[i][j]
            exps = run_psro(game, args.environment, method=method_list[i], model=mod, iters=10, inner_lr=0.75, inner_iters=100, exp_inner_lr=0.75, exp_iters=100, implicit=True)
            model_exps[i].append(exps)
    
    d = model_exps
    tmp_label = 'test_result' + str(method_list[0])
    PATH_DATA = os.path.join(PATH_RESULTS, 'test_data')
    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)
    pickle.dump(d, open(os.path.join(PATH_DATA, tmp_label + '_data.p'), 'wb'))


def plot_error(data, label=''):
    alpha = .4
    data_mean = np.mean(np.array(data), axis=0)
    error_bars = stats.sem(np.array(data), axis=0)
    plt.plot(data_mean, label=label)
    plt.fill_between([i for i in range(data_mean.size)],
                        np.squeeze(data_mean - error_bars),
                        np.squeeze(data_mean + error_bars), alpha=alpha)
    
    
model_list = []
# For testing NAC, load the model here

# m = meta_solver_small().to(device)
# m.load_state_dict(torch.load('./pretrain_model/model.pt'))
# model_list = [m]
# method_list = ['auto']

# For testing Nash-PSRO/Uniform/Self-play/PSRO-RN,
# model_list = [None]
# method_list = ['nash']
# method_list = ['uniform']
# method_list = ['self-play']
# method_list = ['psro_rn']

run_experiment(model_list, method_list, 0, 0, args.num_test_games, args.psro_iters)