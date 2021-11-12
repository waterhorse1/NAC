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
from sklearn import metrics
import scipy.linalg as la
from numpy.linalg import eig
import pickle

from environments.IMP_new import Agent, TorchPop_imp, sample_imp_games, gen_imp_payoffs
from utils.meta_solver import meta_solver_small, meta_solver_large, meta_solver_gru, meta_solver_gru_temp, meta_solver_mlp_large, meta_solver_gru_large, meta_solver_mlp_small
from utils.utils import load_trained_model, gen_labels, fictitious_play, get_agent_nn_total_size
from psro_variants.psro_ns import run_psro



parser = argparse.ArgumentParser(description='gos')
parser.add_argument('--environment', type=str, default='imp')
parser.add_argument('--job_number', type=int, default=111)
parser.add_argument('--num_test_games', type=int, default=20)
parser.add_argument('--psro_iters', type=int, default=9)

args = parser.parse_args(args=[])

PATH_RESULTS = os.path.join('rl_results' , args.environment + '_job_' + str(args.job_number))
if not os.path.exists(PATH_RESULTS):
    os.makedirs(PATH_RESULTS)

device = 'cpu'
 

def run_experiment(mod_list, method_list, unique_params, num_unique, num_test_games, train_iters):

    num_mod = len(mod_list)
    set_seed = 123123

    game_args_dict = {
        'rl_batch_size' : 32,
        'gamma' : 1,
        'use_baseline' : True,
        'len_rollout': 50,
        'batch_size': 1
    }

    POPN = eval(f'TorchPop_{args.environment}')
    
    torch.manual_seed(set_seed)
    np.random.seed(set_seed)
    
    payoffs = [gen_imp_payoffs(seed=(set_seed + i), test=True) for i in range(num_test_games)]
    game_list = [[POPN(game_args_dict, payoffs[i], device, seed=(set_seed + i), test=True) for i in range(num_test_games)] for _ in range(num_mod)]

    nash_game = [POPN(game_args_dict, payoffs[i], device, seed=(set_seed + i), test=True) for i in range(num_test_games)]

    model_exps = [[] for _ in range(num_mod)]

    for i, mod in enumerate(mod_list):
        print('model {}'.format(i))
        for j in tqdm(range(num_test_games)):
            game = game_list[i][j]
            exps = run_psro(game, args.environment, method=method_list[i], model=mod, iters=train_iters, lr=-1, train_iters=-1)
            model_exps[i].append(exps)
    
    d = model_exps
    tmp_label = 'test_result' + str(method_list[0])
    PATH_DATA = os.path.join(PATH_RESULTS, 'test_data')
    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)
    pickle.dump(d, open(os.path.join(PATH_DATA, tmp_label + '_data.p'), 'wb'))
    
model_list = []
# For testing NAC, load the model here

# m = meta_solver_gru().to(device)
# m.load_state_dict(torch.load('./pretrain_model/model.pt'))
# model_list = [m]
# method_list = ['auto']

# For testing Nash-PSRO/Uniform/Self-play,
# model_list = [None]
# method_list = ['nash']
# method_list = ['uniform']
# method_list = ['self-play']

run_experiment(model_list,method_list, 0, 0, args.num_test_games, args.psro_iters)