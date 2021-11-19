import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
np.random.seed(0)
import time
import os
import random

import copy
torch.set_printoptions(sci_mode=False)

from environment.gos import torchPop
from utils.meta_solver import meta_solver_large, meta_solver_small, meta_solver_gru, meta_solver_gru_temp

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

def gen_payoffs(dim, seed=0, test=False):
    W = np.random.randn(dim, dim)
    S = np.random.randn(dim, 1)
    payoffs = (W - W.T) + S - S.T
    return payoffs  

def sample_game(num=8, dim=100):

    game_list = [torchPop(2, dim) for _ in range(num)]
    nf_payoffs = [gen_payoffs(dim) for _ in range(num)]
    return game_list, nf_payoffs


def run(params, wandb_run):

    total_iters = params['total_iters']
    psro_iters = params['psro_iters']
    train_iters = params['inner_train_iters']
    exp_iters = params['exploit_iters']
    batch_size = params['batch_size']
    inner_lr = params['inner_lr']
    outer_lr = params['outer_lr']
    test_lr = params['test_lr']
    window_size = 5
    dim = params['dim']
    grad_clip = params['grad_clip_val']
    
    verbose = False

    train_loss = []
    test_loss = []

    set_seed(params['seed'] * 2)

    if params['model_size'] == 'small':
        model = meta_solver_small().to(device)
    elif params['model_size'] == 'large':
        model = meta_solver_large().to(device)
    elif params['model_size'] == 'gru':
        model = meta_solver_gru().to(device)
    else:
        model = meta_solver_gru_temp().to(device)
        
    meta_optimiser = torch.optim.Adam(model.parameters(), lr=outer_lr)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    lr = inner_lr
    best_exp = 100
    assert psro_iters % window_size == 0

    for overall_iter in range(total_iters):
        st_time = time.time()
        print(f'The overall iteration is {overall_iter}')

        game, nf_payoffs = sample_game(num=batch_size, dim=dim)
        meta_grad = copy.deepcopy(meta_grad_init)
        batch_losses = []
        for ng, g in enumerate(game):
            print(f'Training on game {ng}')
            payoffs = nf_payoffs[ng]
            torch_pop = g
            num_stop = 2

            for psro_iter in range(psro_iters):
                if verbose:
                    print(f'The PSRO iteration for game {ng} is {psro_iter}')
                payoff = torch_pop.get_metagame(payoffs)
                payoff = payoff[None,None,].to(device)
                meta_nash = model(payoff)[0]
                torch_pop.add_agent()
                torch_pop.pop[-1].start_train()
                
                for _ in range(train_iters):
                    exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], meta_nash, payoffs, K=torch_pop.pop_size-1)
                    loss = -(exp_payoff)

                    psro_grad = torch.autograd.grad(loss, [torch_pop.pop[-1].pop_logits], create_graph=True)
                    torch_pop.pop[-1].pop_logits = torch_pop.pop[-1].pop_logits - lr * psro_grad[0]

                if (psro_iter + 1) % window_size == 0:
                    final_payoffs = torch_pop.get_metagame(payoffs)
                    final_payoffs = final_payoffs[None,None,].to(device)
                    meta_nash_final = model(final_payoffs)[0]
                    final_exp = torch_pop.get_exploitability(meta_nash_final, payoffs, exp_iters, test_lr)
                    batch_loss = final_exp
                    if (psro_iter + 1) % psro_iters == 0:
                        batch_losses.append(batch_loss.item())
                    batch_grad = torch.autograd.grad(batch_loss, model.parameters())

                    for gr in range(len(batch_grad)):
                        if torch.norm(batch_grad[gr]) > grad_clip:
                            meta_grad[gr] += ((batch_grad[gr] / torch.norm(batch_grad[gr])) * 0.1).detach()
                        else:
                            meta_grad[gr] += batch_grad[gr].detach()

                    for k in range(2):
                        torch_pop.pop[k].end_train()
                        
                    for k in range(num_stop, num_stop + window_size):
                        torch_pop.pop[k].end_train()

                    num_stop += window_size

        train_loss.append(batch_losses)
        #evaluation step
        if (overall_iter + 1) % 1 == 0:
            test_exps = eval_meta(model, dim, train_iters=train_iters, batch_size=batch_size, psro_iters=psro_iters, exp_iters=exp_iters, lr=lr, test_lr=test_lr, verbose=verbose)
            test_loss.append(test_exps)
            print(f'Final evaluation exploitability: {np.mean(np.array(test_exps))}')

            if np.mean(np.array(test_exps)) < best_exp:
                print('NEW BEST MODEL')
                best_model = copy.deepcopy(model)
                best_exp = np.mean(test_exps)

            model.train()

        meta_optimiser.zero_grad()

        for c, param in enumerate(model.parameters()):
            param.grad = meta_grad[c] / float(batch_size * (psro_iters // window_size))

        meta_optimiser.step()

        log_dict = {'Training Loss': np.mean(batch_losses), 'Test Loss': np.mean(test_exps), 'Best Loss': best_exp}
        wandb_run.log(log_dict, step=overall_iter)
        torch.save(best_model.state_dict(), os.path.join(wandb_run.dir, "model.pt"))
        print(f'time for 1 iteration: {time.time() - st_time}')

    return best_model

def eval_meta(model, dim, train_iters=10, batch_size=1, psro_iters=15, exp_iters=50, lr=8, test_lr=1, verbose=False):
    model.eval()
    game, nf_payoffs = sample_game(num=batch_size, dim=dim)
    exp_loss = 0
    test_losses = []
    print('Currently evaluating')
    for ng, g in enumerate(game):
        #print(f'Currently evaluating on test game {ng}') 
        payoffs = nf_payoffs[ng]
        torch_pop = g
        for _ in range(psro_iters):

            payoff = torch_pop.get_metagame(payoffs)# calculate previous strategies' pay off matrix
            payoff = payoff[None,None,].to(device)
            meta_nash = model(payoff)[0] # calculate nash equillibrium based on meta solver
            torch_pop.add_agent()
            torch_pop.pop[-1].start_train()# add new agent then optimize according to NE
            for _ in range(train_iters):

                # Compute the expected return given that enemy plays agg_strat (using :k first strats)
                exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], meta_nash, payoffs, K=len(meta_nash))
                # Loss
                loss = -(exp_payoff)

                # Optimise !
                psro_grad = torch.autograd.grad(loss, [torch_pop.pop[-1].pop_logits], create_graph=True)
                torch_pop.pop[-1].pop_logits = torch_pop.pop[-1].pop_logits - lr * psro_grad[0]

        final_payoffs = torch_pop.get_metagame(payoffs)
        final_payoffs = final_payoffs[None,None,].to(device)
        meta_nash_final = model(final_payoffs)[0]# calculate nash equillibrium based on meta solver
        final_exp = torch_pop.get_exploitability(meta_nash_final, payoffs, 100, 5.0).item()
        #print(final_exp)
        test_losses.append(final_exp)
 
    return test_losses