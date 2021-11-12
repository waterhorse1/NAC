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