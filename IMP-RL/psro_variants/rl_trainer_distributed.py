import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torch.nn.utils as u
import torch.multiprocessing as mp
#np.random.seed(0)
from numpy.random import RandomState
import time
import wandb

import copy
torch.set_printoptions(sci_mode=False)

from environments.IMP_new import Agent, TorchPop_imp, sample_imp_games
from utils.meta_solver import meta_solver_large, meta_solver_small, meta_solver_gru, meta_solver_gru_temp
from utils.utils import get_agent_nn_total_size, meta_grad_init, set_seed

device = 'cpu'

def psro_subloop(seed, proc, config, game_args_dict, model, game_sampler, meta_grad, stop_value, start_event, end_event, end_count, batch_losses, batch_norms):
    #torch.manual_seed(np.random.randint(10000) + proc)
    set_seed(seed+proc)
    while stop_value == 0:

        start_event.wait()

        if stop_value == 1:
            return

        game = game_sampler(game_args_dict)
        num_stop = 1
        game_loss = []
        #print(game[0].imp.payout_mat)

        torch_pop = game[0]

        for psro_iter in range(config['nb_psro_iters']):
            print(psro_iter)
            payoff = torch_pop.get_metagame().to(device)
            meta_nash1 = model(payoff)[0]
            meta_nash2 = model(-torch.transpose(payoff, -2,-1))[0]
            torch_pop.psro_popn_update(meta_nash1, meta_nash2, config['nb_inner_train_iters'], config['inner_lr'])

            if (psro_iter + 1) % config['window_size'] == 0:
                print('Getting window gradient')
                final_payoffs = torch_pop.get_metagame().to(device)
                meta_nash1 = model(final_payoffs)[0]
                meta_nash2 = model(-torch.transpose(final_payoffs, -2, -1))[0]
                final_exp = torch_pop.get_exploitability(meta_nash1, meta_nash2, config['nb_exp_iters'], config['exp_inner_lr'])
                print('final exp got')
                batch_loss = final_exp
                if (psro_iter + 1) % config['nb_psro_iters'] == 0:
                    if batch_loss.item() > 0:
                        batch_losses[proc] = batch_loss.item()
                batch_grad = torch.autograd.grad(batch_loss, model.parameters(), retain_graph=True)
                print('gradient got')


                for gr in range(len(batch_grad)):
                    total_norm = 0
                    temp_norm = torch.norm(batch_grad[gr])
                    total_norm += temp_norm
                    if temp_norm > config['grad_clip_val']:
                        meta_grad[gr] += ((batch_grad[gr] / temp_norm) * config['clip_correction_val']).detach()
                    else:
                        meta_grad[gr] += batch_grad[gr].detach()
                
                batch_norms[proc] = total_norm.item()
                torch_pop.detach_window(num_stop, config['window_size'])

                num_stop += config['window_size']

        
        end_count += 1

        end_event.wait()

    return None
