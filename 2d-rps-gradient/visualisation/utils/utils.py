import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import os

from utils.meta_solver import meta_solver_with_invariant_large, meta_solver_with_invariant_small, meta_solver_gru


# This is how to load the model
def load_trained_model(job, exp, PATH):

    MODEL_PATH = os.path.join(PATH, 'exp_' + str(exp) + '_model.pth')
    possible_models = [meta_solver_with_invariant_small(), meta_solver_with_invariant_large(), meta_solver_gru()]
    load_success = False
    for mod in possible_models:
        try:
            model = mod
            with open(MODEL_PATH, 'rb') as f:
                model.load_state_dict(torch.load(f))
                break
        except:
            continue
    model.eval()

    return model

def gen_labels(nb_exps):
    mod_exps = np.arange(1, nb_exps+1)
    labels = []
    for j in mod_exps:
        labels.append('exp_' + str(j))
        #labels.append('exp_' + str(j) + '_dpp')
    labels.append('uniform')
    labels.append('nash')
    return labels

def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0,1,(1,dim))
    pop = pop/pop.sum(axis=1)[:,None]
    averages = pop
    exps = []
    for i in range(iters):
        average = np.average(pop, axis=0)
        br = get_br_to_strat(average, payoffs=payoffs)
        exp1 = average@payoffs@br.T
        exp2 = br@payoffs@average.T
        exps.append(exp2-exp1)
        # if verbose:
        #     print(exp, "exploitability")
        averages = np.vstack((averages, average))
        pop = np.vstack((pop, br))
    return averages, exps

def get_br_to_strat(strat, payoffs=None, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br