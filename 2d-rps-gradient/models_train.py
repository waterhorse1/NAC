#from psro_variants.grad_trainer import run
from psro_variants.grad_trainer_distributed import psro_subloop, psro_subloop_cuda
from utils.utils import record_res, get_agent_nn_total_size, set_seed
from environments.gmm_rps import GMMAgent, TorchPop_rps, sample_rps_games
from utils.meta_solver import meta_solver_large, meta_solver_small, meta_solver_gru, meta_solver_gru_large, meta_solver_mlp, meta_solver_mlp_large, meta_solver_gru_temp
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as tmp
import multiprocessing as mp
import json
import pickle
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import time
import copy
import sys

import wandb

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'


def run_dist_cuda(wandb_run):

    inner_config = dict(
        environment = wandb_run.config.environment,
        num_mode = wandb_run.config.num_mode,
        type_ag = wandb_run.config.type_ag,
        batch_size = wandb_run.config.batch_size,
        window_size = wandb_run.config.window_size,
        inner_lr = wandb_run.config.inner_lr,
        outer_lr = wandb_run.config.outer_lr,
        nb_total_iters = wandb_run.config.nb_total_iters,
        nb_psro_iters = wandb_run.config.nb_psro_iters,
        nb_inner_train_iters = wandb_run.config.nb_inner_train_iters,
        nb_exp_iters = wandb_run.config.nb_exp_iters,
        grad_clip_val = wandb_run.config.grad_clip_val,
        clip_correction_val = wandb_run.config.clip_correction_val,
        agent_nn_input_size = wandb_run.config.agent_nn_input_size,
        model_size = wandb_run.config.model_size,
        gradient_type = wandb_run.config.gradient_type,
        distribution_type = wandb_run.config.distribution_type,
        nb_test_iters = wandb_run.config.nb_test_iters,
        test_lr = wandb_run.config.test_lr,
        implicit = wandb_run.config.implicit,
        norm_break_val = wandb_run.config.norm_break_val
    )
    game_args_dict = {
        'batch_size' : wandb_run.config.batch_size,
        'num_mode' : wandb_run.config.num_mode,
        'type_ag' : wandb_run.config.type_ag, 
        'nn_input_size' : wandb_run.config.agent_nn_input_size,
        'nn_total_size' : get_agent_nn_total_size(wandb_run.config.agent_nn_input_size),
        'testing' : False,
        'nf_payoffs' : None,
        'distributed' : True
    }

    processes = []
    seed = wandb_run.config.seed
    set_seed(seed)
    def meta_grad_init(model):
        mg_init = [0 for _ in range(len(model.state_dict()))]
        i = 0
        for param in model.parameters():
            mg_init[i] = torch.zeros(param.size()).to(device)
            mg_init[i].share_memory_()
            i += 1
        return mg_init
    
    #note to self, shouldn't really use eval so change this when have some time
    game_environment = wandb_run.config.environment
    model_type = wandb_run.config.model_size
    game_sampler = eval(f'sample_{game_environment}_games')
    model = eval(f'meta_solver_{model_type}')().to(device)
    model.share_memory()
     
    meta_optimiser = torch.optim.Adam(model.parameters(), lr=wandb_run.config.outer_lr)
    meta_grad = meta_grad_init(model)

    best_exp = 100
    #assert wandb_run.config.nb_psro_iters % wandb_run.config.window_size == 0

    start_count = torch.Tensor([0]).share_memory_()
    end_count = torch.Tensor([0]).share_memory_()
    stop_value = torch.Tensor([0]).share_memory_()
    batch_losses = torch.Tensor([[0] for _ in range(wandb_run.config.batch_size)]).share_memory_()
    true_losses = torch.Tensor([[0] for _ in range(wandb_run.config.batch_size)]).share_memory_()
    batch_norms = torch.Tensor([[0] for _ in range(wandb_run.config.batch_size)]).share_memory_()

    for proc in range(wandb_run.config.batch_size):
        if device == "cuda:0":
            tmp.set_start_method('spawn', force=True)
        p = tmp.Process(target=psro_subloop_cuda, args=(seed, proc, inner_config, game_args_dict, model, game_sampler, meta_grad, stop_value, start_count, end_count, batch_losses, true_losses, batch_norms))
        processes.append(p)
        p.daemon = True
        p.start()

    for overall_iter in range(wandb_run.config.nb_total_iters):
        st_time = time.time()
        print(f'The overall iteration is {overall_iter}')

        start_count[0] = wandb_run.config.batch_size

        if stop_value == 1:
            time.sleep(2)
            break

        while end_count < wandb_run.config.batch_size and stop_value == 0:
            continue
        
        start_count[0] = 0
        end_count[0] = 0

        iter_loss = batch_losses.clone().numpy()
        true_loss = true_losses.clone().numpy()
        iter_loss = iter_loss[iter_loss > 0]
        true_loss = true_loss[true_loss > 0]
        batch_norm = batch_norms.clone().numpy()

        if np.mean(true_loss) < best_exp:
            if np.mean(true_loss) > 0:
                best_exp = np.mean(true_loss)
                best_model = copy.deepcopy(model)
                print('NEW BEST MODEL')
        
        if wandb_run.config.window_size == 0:
            window_size = 1
        else:
            window_size = wandb_run.config.window_size
        if not wandb_run.config.nb_psro_iters % window_size == 0:
            divide = wandb_run.config.nb_psro_iters // window_size + 1
        else:
            divide = wandb_run.config.nb_psro_iters // window_size
        meta_optimiser.zero_grad()

        for c, param in enumerate(model.parameters()):
            param.grad = meta_grad[c] / float(wandb_run.config.batch_size * (divide))
            
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=wandb_run.config.outer_norm_grad)

        meta_optimiser.step()

        print(f'time for 1 iteration: {time.time() - st_time}')

        for gr in range(len(meta_grad)):
            meta_grad[gr] -= meta_grad[gr]

        if overall_iter == wandb_run.config.nb_total_iters - 2:
            stop_value += 1

        log_dict = {'Regularised Loss': np.mean(iter_loss), 'True Loss': np.mean(true_loss), 'Meta-Grad Norm': np.mean(batch_norm), 'Best Loss' : best_exp}
        wandb_run.log(log_dict, step=overall_iter)
        torch.save(best_model.state_dict(), os.path.join(wandb_run.dir, "model.pt"))

        time.sleep(1)
    
    del model
    del start_count
    del end_count
    del batch_losses
    del true_losses
    del batch_norms

    torch.save(best_model.state_dict(), os.path.join(wandb_run.dir, "model.pt"))
        
    wandb_run.finish()

if __name__ == "__main__":

    hyperparameter_defaults = dict(
        environment = 'rps',
        num_mode = 7,
        type_ag = 'logits',
        batch_size = 8,
        window_size = 9,
        inner_lr = 2,
        outer_lr = 0.007,
        nb_total_iters = 400,
        nb_psro_iters = 15,
        nb_inner_train_iters = 5,
        nb_exp_iters = 20,
        grad_clip_val = 100,
        clip_correction_val = 0.001,
        agent_nn_input_size = 2,
        model_size = 'small',
        gradient_type = 'non-implicit',
        distribution_type = 'distributed',
        nb_test_iters = 20,
        test_lr = 2,
        norm_break_val = 10,
        implicit = False,
        outer_norm_grad = 2,
        seed = 0,
    )

    new_config = hyperparameter_defaults
    wandb_run = wandb.init(config=new_config, project='nac-2d-gradient')
    
    run_dist_cuda(wandb_run)