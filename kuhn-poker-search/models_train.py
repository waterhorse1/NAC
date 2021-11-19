from auto_psro_variations.es_distributed_trunc import psro_subloop, sample_game
from utils.ES import evo_update_es, evo_gradient_full
from utils.meta_solver import meta_solver_small, meta_solver_large, meta_solver_gru


import os
import torch
import torch.nn.functional as f
import multiprocessing as mp
import numpy as np
import time
import copy
import random

import wandb

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

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


def run_dist(wandb_run):

    inner_config = dict(
        batch_size = wandb_run.config.batch_size,
        window_size = wandb_run.config.window_size,
        outer_lr = wandb_run.config.outer_lr,
        nb_total_iters = wandb_run.config.nb_total_iters,
        nb_psro_iters = wandb_run.config.nb_psro_iters,
        model_size = wandb_run.config.model_size,
        num_perturbations = wandb_run.config.num_perturbations,
        br_type = wandb_run.config.br_type,
        seed = wandb_run.config.seed
    )

    processes = []
    
    model_type = wandb_run.config.model_size
    model = eval(f'meta_solver_{model_type}')().to(device)
    model.share_memory()
    num_windows = int(wandb_run.config.nb_psro_iters / wandb_run.config.window_size)
     
    meta_optimiser = torch.optim.Adam(model.parameters(), lr=wandb_run.config.outer_lr)
    scheduler = torch.optim.lr_scheduler.StepLR(meta_optimiser, step_size=wandb_run.config.lr_sched_stepsize, gamma=wandb_run.config.lr_sched_gamma)
    
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]

    best_exp = 100
    assert wandb_run.config.nb_psro_iters % wandb_run.config.window_size == 0

    task_queue = mp.Queue()
    scores_queue = mp.Queue()
    start_event = mp.Event()
    end_event = mp.Event()

    for proc in range(15):
        p = mp.Process(target=psro_subloop, args=(proc, model, inner_config, start_event, end_event, task_queue, scores_queue))
        processes.append(p)
        p.daemon = True
        p.start()
    
    stop_value = 0

    set_seed(wandb_run.config.seed)
    iter_sample_seeds = np.random.randint(10000, size=(120))
    for curr_iter in range(wandb_run.config.nb_total_iters):
        st_time = time.time()
        print(f'The overall iteration is {curr_iter}')

        if stop_value == 1:
            break

        noises = [0 for _ in range(wandb_run.config.num_perturbations)]
        
        for _ in range(15):
            task_queue.put([stop_value, iter_sample_seeds[curr_iter]])

        envs, pops = sample_game(batch_size=wandb_run.config.batch_size, train_seed = iter_sample_seeds[curr_iter])
        for k in range(15):
            _, noise = evo_update_es(model, 0.1, (iter_sample_seeds[curr_iter]+k))
            noises[k] = noise[0]
            _, noise = evo_update_es(model, 0.1, (iter_sample_seeds[curr_iter]+k+15))
            noises[k + 15] = noise[0]

        fwd_scores = [[] for _ in range(wandb_run.config.batch_size)]
        meta_grad = copy.deepcopy(meta_grad_init)
        start_event.set()

        for batch in range(wandb_run.config.batch_size):
            pop = pops[batch]

            for psro_iter in range(wandb_run.config.nb_psro_iters):
                meta_game = torch.Tensor(pop.get_metagame())[None,None]
                meta_nash = model(meta_game)[0].detach().numpy()
                pop.popn_update(meta_nash, wandb_run.config.br_type)

                if (psro_iter + 1) % wandb_run.config.window_size == 0:
                    window_metagame = torch.Tensor(pop.get_metagame())[None,None,]
                    window_meta_nash = model(window_metagame)[0].detach().numpy()
                    window_exp, _ = pop.get_exploitability(window_meta_nash)
                    fwd_scores[batch].append(window_exp)

        print(f'The main process has solved for all of the forward difference values')

        #Receive the scores and noise from all of the perturbed models and collate them into separate lists
        train_scores = [0 for _ in range(wandb_run.config.num_perturbations)]
        game_scores = [0 for _ in range(wandb_run.config.num_perturbations)]
        processes_fin = 0

        while processes_fin < wandb_run.config.num_perturbations:
            if scores_queue.empty():
                continue
            pert_res = scores_queue.get()
            temp_scores = (np.array(pert_res[1]) - np.array(fwd_scores)).tolist()
            train_scores[pert_res[0]] = pert_res[1]
            game_scores[pert_res[0]] = temp_scores
            processes_fin += 1
        
        print(f'The main process has collected all of game scores / noises')

        #Calculate the full gradient for each game
        for window in range(num_windows):
            game_gradients = []
            for batch in range(wandb_run.config.batch_size):
                game_score = [scores[batch][window] for scores in game_scores]
                game_gradient = evo_gradient_full(game_score, noises, 0.1)
                game_gradients.append(game_gradient)

            #Solve for the gradient update
            for g, (var_name, w) in enumerate(model.state_dict().items()):
                for i in range(wandb_run.config.batch_size):
                    meta_grad[g] += torch.Tensor(game_gradients[i][var_name])

        meta_optimiser.zero_grad()

        for c, param in enumerate(model.parameters()):
            param.grad = meta_grad[c] / float(wandb_run.config.batch_size * (wandb_run.config.nb_psro_iters // wandb_run.config.window_size))

        meta_optimiser.step()

        train_score = np.mean([score[-1] for score in fwd_scores])    
        if train_score < best_exp:
            if train_score > 0:
                print('NEW BEST MODEL')
                best_model = copy.deepcopy(model)
                best_exp = train_score

        print(f'time for 1 iteration: {time.time() - st_time}')

        if curr_iter == wandb_run.config.nb_total_iters - 1:
            stop_value += 1
        
        scheduler.step()

        log_dict = {'True Loss': np.mean(train_score), 'Best Loss' : best_exp}
        wandb_run.log(log_dict, step=curr_iter)

        start_event.clear()
        end_event.set()
        time.sleep(2)
        end_event.clear()

    torch.save(best_model.state_dict(), os.path.join(wandb_run.dir, "model.pt"))
        
    wandb_run.finish()

if __name__ == "__main__":

    hyperparameter_defaults = dict(
        num_perturbations = 30,
        batch_size = 5,
        window_size = 5,
        outer_lr = 0.1,
        nb_total_iters = 100,
        nb_psro_iters = 15,
        model_size = 'small',
        br_type = 'approx_br_rand', 
        lr_sched_stepsize = 20,
        lr_sched_gamma = 0.5,
        seed = 50
    )

    new_config = hyperparameter_defaults
    wandb_run = wandb.init(config=new_config, project='nac-kp-es-search')
    
    run_dist(wandb_run)