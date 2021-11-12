import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import os
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
import random

from utils.meta_solver import meta_solver_large, meta_solver_small, meta_solver_gru, meta_solver_gru_temp

def set_seed(seed, cudnn=True):
    """
    Seed everything we can!
    Note that gym environments might need additional seeding (env.seed(seed)),
    and num_workers needs to be set to 1.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # note: the below slows down the code but makes it reproducible
    torch.cuda.manual_seed_all(seed) # Sets the seed for generating random numbers on all GPUs. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# This is how to load the model
def load_trained_model(job, exp, MODEL_PATH, PARAMS_PATH):

    MODEL_PATH = os.path.join(MODEL_PATH, 'exp_' + str(exp) + '_model.pth')
    PARAM_PATH = os.path.join(PARAMS_PATH, 'exp_' + str(exp) + '_params.json')
    
    with open(PARAM_PATH) as json_file:
        params = json.load(json_file)
    
    model_type = params['model_size']
    
    model = eval(f'meta_solver_{model_type}')()
    with open(MODEL_PATH, 'rb') as f:
        model.load_state_dict(torch.load(f))

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

#Search over the pure strategies to find the BR to a strategy
def get_br_to_strat(strat, payoffs, row=True):
    if not row:
        row_weighted_payouts = strat@payoffs
    else:
        row_weighted_payouts = payoffs@strat
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmax(row_weighted_payouts)] = 1
    return br

def fictitious_play(iters=2000, payoffs=None, verbose=False):
    dim = payoffs.shape[0]
    pop = np.random.uniform(0,1,(1,dim))
    pop1 = pop/pop.sum(axis=1)[:,None]
    pop = np.random.uniform(0,1,(1,dim))
    pop2 = pop/pop.sum(axis=1)[:,None]
    averages1 = pop1
    averages2 = pop2
    for i in range(iters):
        average1 = np.average(pop1, axis=0)
        br2 = get_br_to_strat(average1, payoffs=-payoffs, row=False)#x^T-AY
        average2 = np.average(pop2, axis=0)
        br1 = get_br_to_strat(average2, payoffs=payoffs)#x^TAY
        averages1 = np.vstack((averages1, average1))
        averages2 = np.vstack((averages2, average2))
        pop1 = np.vstack((pop1, br1))
        pop2 = np.vstack((pop2, br2))
    return averages1, averages2

def get_agent_nn_total_size(input_size):
    size2 = input_size * 2
    size4 = (input_size * size2) + size2
    return (size4 + (2 * size2)) + 2

def get_agent_nn_input_size(total_size):
    if total_size == 22:
        return 2
    elif total_size == 58:
        return 4
    else:
        return 8

def meta_grad_init(model):
    mg_init = [0 for _ in range(len(model.state_dict()))]
    i = 0
    for param in model.parameters():
        mg_init[i] = torch.zeros(param.size())
        mg_init[i].share_memory_()
        i += 1
    return mg_init

def record_res(exps, args, best_model, PATH_RESULTS):

    params = vars(args)

    exps_train = []
    exps_train.append(exps[0])        

    PATH_BEST_MODELS = os.path.join(PATH_RESULTS, 'best_models')
    if not os.path.exists(PATH_BEST_MODELS):
        os.makedirs(PATH_BEST_MODELS)
    PATH_PARAMS = os.path.join(PATH_RESULTS, 'params')
    if not os.path.exists(PATH_PARAMS):
        os.makedirs(PATH_PARAMS)
    PATH_DATA = os.path.join(PATH_RESULTS, 'train_data')
    if not os.path.exists(PATH_DATA):
        os.makedirs(PATH_DATA)

    torch.save(best_model.state_dict(), os.path.join(PATH_BEST_MODELS, 'exp_' + str(args.exp_number) + '_model.pth'))

    with open(os.path.join(PATH_PARAMS, 'exp_' + str(args.exp_number) + '_params.json'), 'w', encoding='utf-8') as json_file:
        json.dump(params, json_file, indent=4)

    d = {
        'train_exploitabilities' : exps_train
    }

    pickle.dump(d, open(os.path.join(PATH_DATA, 'exp_' + str(args.exp_number) + '_data.p'), 'wb'))

    def plot_error(data, label=''):
        data_mean = np.mean(np.array(data), axis=1)
        error_bars = stats.sem(np.array(data), axis=1)
        plt.plot(data_mean, label=label)
        plt.fill_between([i for i in range(data_mean.size)],
                            np.squeeze(data_mean - error_bars),
                            np.squeeze(data_mean + error_bars), alpha=alpha)

    alpha = .4
    fig_handle = plt.figure()
    plot_error(exps[0], label='Train')
    plt.legend(loc="upper left")
    plt.title('Experiment')

    PATH_PLOTS= os.path.join(PATH_RESULTS, 'training_plots')
    if not os.path.exists(PATH_PLOTS):
        os.makedirs(PATH_PLOTS)

    plt.savefig(os.path.join(PATH_PLOTS, 'training_curves_exp_' + str(args.exp_number) + '.pdf'))
    plt.close()