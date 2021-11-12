import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.utils as u
np.random.seed(0)
from numpy.random import RandomState
import time

import copy
torch.set_printoptions(sci_mode=False)

from environments.gos import torchAgent, TorchPop_gos, sample_gos_games
from environments.gmm_rps import GMMAgent, TorchPop_rps, sample_rps_games
from environments.blotto import DLottoAgent, TorchPop_blotto, sample_blotto_games
from utils.meta_solver import meta_solver_large, meta_solver_small, meta_solver_gru, meta_solver_gru_temp
from utils.utils import get_agent_nn_total_size

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def run(args):

    args_dict = vars(args)

    if args_dict['gradient_type'] == 'implicit':
        args_dict['implicit'] = True
    else:
        args_dict['implicit'] = False

    game_args_dict = {
        'batch_size' : args_dict['batch_size'],
        'gos_dim' : args_dict['gos_dim'],
        'num_mode' : args_dict['num_mode'],
        'type_ag' : args_dict['type_ag'], 
        'nb_customers' : args_dict['num_customer'],
        'nb_positions' : args_dict['num_position'],
        'nn_input_size' : args_dict['agent_nn_input_size'],
        'nn_total_size' : get_agent_nn_total_size(args_dict['agent_nn_input_size']),
        'testing' : False,
        'nf_payoffs' : None
    }
    
    #note to self, shouldn't really use eval so change this when have some time
    game_environment = args_dict['environment']
    model_type = args_dict['model_size']
    game_sampler = eval(f'sample_{game_environment}_games')
    model = eval(f'meta_solver_{model_type}')().to(device)
        
    meta_optimiser = torch.optim.Adam(model.parameters(), lr=args_dict['outer_lr'])
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    best_exp = 100
    assert args_dict['nb_psro_iters'] % args_dict['window_size'] == 0

    for overall_iter in range(args_dict['nb_total_iters']):
        st_time = time.time()
        print(f'The overall iteration is {overall_iter}')
        
        #gos_payoffs will only be relevant for games of skill environment
        game, gos_payoffs = game_sampler(game_args_dict)
        meta_grad = copy.deepcopy(meta_grad_init)
        inner_loop_loss = 0
        batch_losses = []

        for ng, g in enumerate(game):
            print(f'Training on game {ng}')
            if args_dict['environment'] == 'gos':
                payoffs = gos_payoffs[ng]
                game_args_dict['nf_payoffs'] = payoffs
            else:
                payoffs = None
            torch_pop = g
            torch_pop.nf_payoffs = payoffs
            num_stop = 2

            for psro_iter in range(args_dict['nb_psro_iters']):
                payoff = torch_pop.get_metagame().to(device)
                meta_nash = model(payoff)[0]
                torch_pop.psro_popn_update(meta_nash, args_dict['nb_inner_train_iters'], args_dict['inner_lr'], args_dict['lam'], implicit=args_dict['implicit'])

                if (psro_iter + 1) % args_dict['window_size'] == 0:
                    print('Getting window gradient')
                    final_payoffs = torch_pop.get_metagame().to(device)
                    meta_nash = model(final_payoffs)[0]
                    final_exp = torch_pop.get_exploitability(meta_nash, args_dict['nb_exp_iters'], args_dict['test_lr'], test=False, implicit=args_dict['implicit'])
                    batch_loss = final_exp
                    if (psro_iter + 1) % args_dict['nb_psro_iters'] == 0:
                        batch_losses.append(batch_loss.item())
                    batch_grad = torch.autograd.grad(batch_loss, model.parameters())

                    for gr in range(len(batch_grad)):
                        if torch.norm(batch_grad[gr]) > args_dict['grad_clip_val']:
                            meta_grad[gr] += ((batch_grad[gr] / torch.norm(batch_grad[gr])) * 0.1).detach()
                        else:
                            meta_grad[gr] += batch_grad[gr].detach()

                    torch_pop.detach_window(num_stop, args_dict['window_size'])

                    num_stop += args_dict['window_size']

        if np.mean(batch_losses) < best_exp:
            best_exp = np.mean(batch_losses)
            best_model = copy.deepcopy(model)
            print('NEW BEST MODEL')

        meta_optimiser.zero_grad()

        for c, param in enumerate(model.parameters()):
            param.grad = meta_grad[c] / float(args_dict['batch_size'] * (args_dict['nb_psro_iters'] // args_dict['window_size']))

        meta_optimiser.step()

        print(f'time for 1 iteration: {time.time() - st_time}')

    return best_model