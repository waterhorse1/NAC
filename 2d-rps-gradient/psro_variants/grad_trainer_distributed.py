import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.utils as u
import torch.multiprocessing as tmp
import multiprocessing as mp
from numpy.random import RandomState
import time
import wandb

import copy
torch.set_printoptions(sci_mode=False)

from environments.gmm_rps import GMMAgent, TorchPop_rps, sample_rps_games
from utils.meta_solver import meta_solver_large, meta_solver_small, meta_solver_gru, meta_solver_gru_temp
from utils.utils import get_agent_nn_total_size, set_seed

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

def psro_subloop(seed, proc, config, game_args_dict, model, game_sampler, meta_grad, stop_value, start_event, end_event, end_count, batch_losses, true_losses, batch_norms):
    set_seed(seed + proc)
    while stop_value == 0:

        print('waiting here')
        print(start_event.is_set())
        start_event.wait()
        print('iteration is startin')

        if stop_value == 1:
            return
    
        game, gos_payoffs = game_sampler(game_args_dict)
        num_stop = 2
        game_loss = []
        
        if config['environment'] == 'rps':
            exp_inner_lr = 2.0
        else:
            exp_inner_lr = config['inner_lr']

        if config['environment'] == 'gos':
            payoffs = gos_payoffs[0]
        else:
            payoffs = None
        torch_pop = game[0]
        torch_pop.nf_payoffs = payoffs

        for psro_iter in range(config['nb_psro_iters']):
            payoff = torch_pop.get_metagame().to(device)
            meta_nash = model(payoff)[0]
            torch_pop.psro_popn_update(meta_nash, config['nb_inner_train_iters'], config['inner_lr'], config['inner_break_val'], implicit=config['implicit'])

            if (psro_iter + 1) % config['window_size'] == 0:
                print('Getting window gradient')
                final_payoffs = torch_pop.get_metagame().to(device)
                meta_nash = model(final_payoffs)[0]
                final_exp = torch_pop.get_exploitability(meta_nash, config['nb_exp_iters'], config['inner_break_val'], lr=exp_inner_lr, test=False, implicit=config['implicit'])
                batch_loss = final_exp
                if (psro_iter + 1) % config['nb_psro_iters'] == 0:
                    if batch_loss.item() > 0:
                        batch_losses[proc] = batch_loss.item()
                batch_grad = torch.autograd.grad(batch_loss, model.parameters())

                for gr in range(len(batch_grad)):
                    total_norm = 0
                    temp_norm = torch.norm(batch_grad[gr])
                    total_norm += temp_norm
                    meta_grad[gr] += batch_grad[gr].detach()
                
                batch_norms[proc] = total_norm.item()
                torch_pop.detach_window(num_stop, config['window_size'])

                num_stop += config['window_size']
                
        final_payoffs = torch_pop.get_metagame().to(device)
        meta_nash = model(final_payoffs)[0]
        true_exp = torch_pop.get_exploitability(meta_nash, config['nb_test_iters'], config['test_lr'], test=True, implicit=config['implicit'])
        true_losses[proc] = true_exp.item()
        
        end_count += 1

        end_event.wait()

    return

def psro_subloop_cuda(seed, proc, config, game_args_dict, model, game_sampler, meta_grad, stop_value, start_count, end_count, batch_losses, true_losses, batch_norms):
    set_seed(seed + proc)
    while stop_value == 0:

        print(f'process {proc} is waiting to start')
        while start_count == 0:
            continue
        print(f'process {proc} has started the iteration')
        
        game, gos_payoffs = game_sampler(game_args_dict)
        num_stop = 2
        game_loss = []

        if config['environment'] == 'rps':
            exp_inner_lr = config['inner_lr']
        else:
            exp_inner_lr = config['inner_lr']

        if config['environment'] == 'gos':
            payoffs = gos_payoffs[0]
        else:
            payoffs = None
        torch_pop = game[0]
        torch_pop.nf_payoffs = payoffs
        
        if config['window_size'] == 0:
            create_graph = False
            window_size = 1
        else:
            create_graph = True
            window_size = config['window_size']
        for psro_iter in range(config['nb_psro_iters']):
            payoff = torch_pop.get_metagame().to(device)
            meta_nash = model(payoff)[0]
            torch_pop.psro_popn_update(meta_nash, config['nb_inner_train_iters'], config['inner_lr'], config['norm_break_val'], create_graph=create_graph, implicit=config['implicit'])

            if (psro_iter + 1) % window_size == 0 or (psro_iter + 1) % config['nb_psro_iters'] == 0:
                print('Getting window gradient')
                final_payoffs = torch_pop.get_metagame().to(device)
                meta_nash = model(final_payoffs)[0]
                final_exp = torch_pop.get_exploitability(meta_nash, config['nb_exp_iters'], config['norm_break_val'], exp_inner_lr, test=False, implicit=config['implicit'], create_graph=create_graph)
                batch_loss = final_exp
                if (psro_iter + 1) % config['nb_psro_iters'] == 0:
                    if batch_loss.item() > 0:
                        batch_losses[proc] = batch_loss.item()
                batch_grad = torch.autograd.grad(batch_loss, model.parameters())

                for gr in range(len(batch_grad)):
                    total_norm = 0
                    temp_norm = torch.norm(batch_grad[gr])
                    total_norm += temp_norm
                    meta_grad[gr] += batch_grad[gr].detach()
                
                batch_norms[proc] = total_norm.item()
                if not (psro_iter + 1) % config['nb_psro_iters'] == 0:
                    torch_pop.detach_window(num_stop, config['window_size'])
                else:
                    torch_pop.detach_window(0, len(torch_pop.pop))

                num_stop += config['window_size']
                
        true_losses[proc] = batch_losses[proc]
        
        end_count += 1

        print(f'process proc {proc} has finished the itration and is waiting')

        while start_count != 0:
            continue

    print(f'process {proc} is complete')     
    return
        
