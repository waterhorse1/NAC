#from psro_variants.grad_trainer import run
from psro_variants.rl_trainer_distributed import psro_subloop
from utils.utils import record_res, get_agent_nn_total_size, set_seed
from environments.IMP_new import Agent, TorchPop_imp, sample_imp_games
from utils.meta_solver import meta_solver_large, meta_solver_small, meta_solver_gru, meta_solver_gru_large, meta_solver_mlp_small, meta_solver_mlp_large
import os
import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
import torch.multiprocessing as mp
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

device = 'cpu'

def run_dist(wandb_run):

    inner_config = dict(
        environment = wandb_run.config.environment,
        type_ag = wandb_run.config.type_ag,
        batch_size = wandb_run.config.batch_size,
        window_size = wandb_run.config.window_size,
        inner_lr = wandb_run.config.inner_lr,
        exp_inner_lr = wandb_run.config.exp_inner_lr,
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
        lr_sched_stepsize = wandb_run.config.lr_sched_stepsize,
        lr_sched_gamma = wandb_run.config.lr_sched_gamma
    )
    game_args_dict = {
        'rl_batch_size' : 32,
        'gamma' : 1,
        'use_baseline' : True,
        'len_rollout': 50,
        'batch_size': 1
    }

    processes = []
    seed = wandb_run.config.seed
    set_seed(wandb_run.config.seed * 2)
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
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, step_size=wandb_run.config.lr_sched_stepsize, gamma=wandb_run.config.lr_sched_gamma)
    meta_grad = meta_grad_init(model)

    best_exp = 100
    assert wandb_run.config.nb_psro_iters % wandb_run.config.window_size == 0

    start_count = torch.Tensor([0]).share_memory_()
    end_count = torch.Tensor([0]).share_memory_()
    stop_value = torch.Tensor([0]).share_memory_()
    batch_losses = torch.Tensor([[0] for _ in range(wandb_run.config.batch_size)]).share_memory_()
    batch_norms = torch.Tensor([[0] for _ in range(wandb_run.config.batch_size)]).share_memory_()
    start_event = mp.Event()
    end_event = mp.Event()

    for proc in range(wandb_run.config.batch_size):
        if device == "cuda:0":
            mp.set_start_method('spawn', force=True)
        p = mp.Process(target=psro_subloop, args=(seed, proc, inner_config, game_args_dict, model, game_sampler, meta_grad, stop_value, start_event, end_event, end_count, batch_losses, batch_norms))
        processes.append(p)
        p.daemon = True
        p.start()

    for overall_iter in range(wandb_run.config.nb_total_iters):
        st_time = time.time()
        print(f'The overall iteration is {overall_iter}')

        start_event.set()
        processes_fin = []

        if stop_value == 1:
            time.sleep(5)
            break

        while end_count < wandb_run.config.batch_size and stop_value == 0:
            continue

        end_count -= wandb_run.config.batch_size

        iter_loss = batch_losses.clone().numpy()
        iter_loss = iter_loss[iter_loss > 0]
        batch_norm = batch_norms.clone().numpy()

        if np.mean(iter_loss) < best_exp:
            if np.mean(iter_loss) > 0:
                best_exp = np.mean(iter_loss)
                best_model = copy.deepcopy(model)
                print('NEW BEST MODEL')

        meta_optimiser.zero_grad()

        for c, param in enumerate(model.parameters()):
            param.grad = meta_grad[c] / float(wandb_run.config.batch_size * (wandb_run.config.nb_psro_iters // wandb_run.config.window_size))

        meta_optimiser.step()

        print(f'time for 1 iteration: {time.time() - st_time}')

        for gr in range(len(meta_grad)):
            meta_grad[gr] -= meta_grad[gr]

        if overall_iter == wandb_run.config.nb_total_iters - 2:
            stop_value += 1

        scheduler.step()

        log_dict = {'Regularised Loss': np.mean(iter_loss), 'Loss': np.mean(iter_loss), 'Meta-Grad Norm': np.mean(batch_norm), 'Best Loss' : best_exp}
        wandb_run.log(log_dict, step=overall_iter)
        torch.save(best_model.state_dict(), os.path.join(wandb_run.dir, "model.pt"))

        start_event.clear()
        end_event.set()
        time.sleep(2)
        end_event.clear()
    
    
    del model
    del start_count
    del end_count
    del batch_losses
    del batch_norms

    torch.save(best_model.state_dict(), os.path.join(wandb_run.dir, "model.pt"))
        
    wandb_run.finish()

if __name__ == "__main__":

    hyperparameter_defaults = dict(
        environment = 'imp',
        type_ag = 'logits',
        batch_size = 6,
        window_size = 3,
        inner_lr = 10.0,
        outer_lr = 0.005,
        exp_inner_lr = 10,
        nb_total_iters = 50,
        nb_psro_iters = 9,
        nb_inner_train_iters = 10,
        nb_exp_iters = 20,
        grad_clip_val = 1.0,
        clip_correction_val = 0.02,
        agent_nn_input_size = 2,
        model_size = 'gru ',
        gradient_type = 'non-implicit',
        distribution_type = 'distributed',
        nb_test_iters = 17,
        test_lr = 5,
        implicit = False,
        lr_sched_stepsize = 20,
        lr_sched_gamma = 0.5,
        seed = 42
    )

    new_config = hyperparameter_defaults
    wandb_run = wandb.init(config=new_config, project='nac-imp-rl')
    
    run_dist(wandb_run)
