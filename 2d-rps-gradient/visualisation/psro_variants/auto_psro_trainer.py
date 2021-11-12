import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(0)
from numpy.random import RandomState
import time

import copy
torch.set_printoptions(sci_mode=False)

from environments.rps_enivronment import GMMAgent, TorchPop
from utils.meta_solver import meta_solver_with_invariant_large, meta_solver_with_invariant_small, meta_solver_gru

device = 'cuda:0'

def sample_game(num=5, num_mode=3, seed=0, resample_mu=True, test=False):
  game_list = [TorchPop(num_mode=num_mode, seed=(seed + i), resample_mu=resample_mu, test=test) for i in range(num)]
  return game_list

def run(args, checkpoint_path):

    total_iters = args.nb_total_iters
    psro_iters = args.nb_psro_iters
    train_iters = args.nb_inner_train_iters
    exp_iters = args.nb_exp_iters
    batch_size = args.batch_size
    num_mode = args.num_mode
    inner_lr = args.inner_lr
    outer_lr = args.outer_lr
    test_lr = args.test_lr
    resample_mu = True
    verbose = False

    train_loss = []
    test_loss = []

    if args.model_size == 'small':
        model = meta_solver_with_invariant_small().to(device)
    elif args.model_size == 'large':
        model = meta_solver_with_invariant_large().to(device)
    else:
        model = meta_solver_gru().to(device)
        
    meta_optimiser = torch.optim.Adam(model.parameters(), lr=outer_lr)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    lr = inner_lr
    best_exp = 100

    for overall_iter in range(total_iters):
      st_time = time.time()
      print(f'The overall iteration is {overall_iter}')

      games_batch = sample_game(num=batch_size, num_mode=num_mode, resample_mu=resample_mu)
      meta_grad = copy.deepcopy(meta_grad_init)
      inner_loop_loss = 0
      batch_losses = []
      for ng, g in enumerate(games_batch):
        #print(f'Training on game {ng}')
        torch_pop = g

        for psro_iter in range(psro_iters):
          #print(psro_iter)
          if verbose:
            print(f'The PSRO iteration for game {ng} is {psro_iter}')
          payoff = torch_pop.get_metagame()
          payoff = payoff[None,None,].to(device)
          meta_nash = model(payoff)[0]
          torch_pop.add_new()
          torch_pop.pop[-1].x.requires_grad = True

          for training_iter in range(train_iters):
            #if verbose:
            #  print(f'The inner loop BR iteration is {training_iter}')
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], meta_nash, len(meta_nash))
            loss = -(exp_payoff)

            psro_grad = torch.autograd.grad(loss, torch_pop.pop[-1].x, create_graph=True)
            torch_pop.pop[-1].x = torch_pop.pop[-1].x - lr * psro_grad[0]
          
        final_payoffs = torch_pop.get_metagame()
        final_payoffs = final_payoffs[None,None,].to(device)
        meta_nash_final = model(final_payoffs)[0]
        final_exp = torch_pop.get_exploitability_train(meta_nash_final, exp_iters, test_lr)
        batch_loss = final_exp
        batch_losses.append(batch_loss.item())
        st_b = time.time()
        batch_grad = torch.autograd.grad(batch_loss, model.parameters())
        print(f'time for batch grad {time.time() - st_b}')
        #print(batch_grad)

        for gr in range(len(batch_grad)):
          meta_grad[gr] += batch_grad[gr].detach()

        #inner_loop_loss += batch_loss.item()/batch_size
      
      #print(meta_grad)
      train_loss.append(batch_losses)
      print(f'Final evaluation exploitability: {np.mean(np.array(batch_losses))}')
      if np.mean(np.array(batch_losses)) < best_exp:
        print('NEW BEST MODEL')
        best_model = copy.deepcopy(model)
        best_model.eval()
        best_exp = np.mean(np.array(batch_losses))
      #evaluation step
      #test_exps = eval_meta(model, train_iters=train_iters, batch_size=batch_size, num_mode=num_mode, psro_iters=psro_iters, exp_iters=exp_iters, lr=lr, verbose=verbose)
      #test_loss.append(test_exps)
      #print(f'Final evaluation exploitability: {np.mean(np.array(test_exps))}')

      model.train()
      
      meta_optimiser.zero_grad()

      for c, param in enumerate(model.parameters()):
          param.grad = meta_grad[c] / float(batch_size)

      meta_optimiser.step()
    
      print(f'time for 1 iteration: {time.time() - st_time}')
      torch.save(model.state_dict(), checkpoint_path)

    return [train_loss, test_loss], best_model

def run_truncated(args, checkpoint_path):

    total_iters = args.nb_total_iters
    psro_iters = args.nb_psro_iters
    train_iters = args.nb_inner_train_iters
    exp_iters = args.nb_exp_iters
    batch_size = args.batch_size
    num_mode = args.num_mode
    inner_lr = args.inner_lr
    outer_lr = args.outer_lr
    test_lr = args.test_lr
    window_size = args.window_size
    resample_mu = True
    verbose = False

    train_loss = []
    test_loss = []

    if args.model_size == 'small':
        model = meta_solver_with_invariant_small().to(device)
    elif args.model_size == 'large':
        model = meta_solver_with_invariant_large().to(device)
    else:
        model = meta_solver_gru().to(device)
        
    meta_optimiser = torch.optim.Adam(model.parameters(), lr=outer_lr)
    meta_grad_init = [0 for _ in range(len(model.state_dict()))]
    lr = inner_lr
    best_exp = 100
    assert psro_iters % window_size == 0

    for overall_iter in range(total_iters):
      st_time = time.time()
      print(f'The overall iteration is {overall_iter}')

      games_batch = sample_game(num=batch_size, num_mode=num_mode, resample_mu=resample_mu)
      meta_grad = copy.deepcopy(meta_grad_init)
      inner_loop_loss = 0
      batch_losses = []
      for ng, g in enumerate(games_batch):
        #print(f'Training on game {ng}')
        torch_pop = g
        num_stop = 2

        for psro_iter in range(psro_iters):
          print(psro_iter)
          if verbose:
            print(f'The PSRO iteration for game {ng} is {psro_iter}')
          payoff = torch_pop.get_metagame()
          payoff = payoff[None,None,].to(device)
          meta_nash = model(payoff)[0]
          torch_pop.add_new()
          torch_pop.pop[-1].x.requires_grad = True

          for training_iter in range(train_iters):
            #if verbose:
            #  print(f'The inner loop BR iteration is {training_iter}')
            exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], meta_nash, len(meta_nash))
            #print(exp_payoff)
            loss = -(exp_payoff)

            psro_grad = torch.autograd.grad(loss, torch_pop.pop[-1].x, create_graph=True)
            torch_pop.pop[-1].x = torch_pop.pop[-1].x - lr * psro_grad[0]

          if (psro_iter + 1) % window_size == 0:
            print('Getting window gradient')

            final_payoffs = torch_pop.get_metagame()
            final_payoffs = final_payoffs[None,None,].to(device)
            meta_nash_final = model(final_payoffs)[0]
            final_exp = torch_pop.get_exploitability_train(meta_nash_final, exp_iters, test_lr)
            batch_loss = final_exp
            batch_losses.append(batch_loss.item())
            st_b = time.time()
            batch_grad = torch.autograd.grad(batch_loss, model.parameters())
            print(f'time for batch grad {time.time() - st_b}')
            #print(batch_grad)

            for gr in range(len(batch_grad)):
              meta_grad[gr] += batch_grad[gr].detach()

            for f in range(2):
              torch_pop.pop[f].x = torch_pop.pop[f].x.detach()

            for k in range(num_stop, num_stop + window_size):
              torch_pop.pop[k].x = torch_pop.pop[k].x.detach()

            num_stop += window_size

        #inner_loop_loss += batch_loss.item()/batch_size
      
      #print(meta_grad)
      train_loss.append(batch_losses)
      #evaluation step
      test_exps = eval_meta(model, train_iters=train_iters, batch_size=batch_size, num_mode=num_mode, psro_iters=psro_iters, exp_iters=exp_iters, lr=lr, verbose=verbose)
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
    
      print(f'time for 1 iteration: {time.time() - st_time}')
      torch.save(model.state_dict(), checkpoint_path)

    return [train_loss, test_loss], best_model

def eval_meta(model, train_iters=10, batch_size=1, num_mode=7, psro_iters=15, exp_iters=50, lr=8, test_lr=1, verbose=False):
    model.eval()
    game_list = sample_game(num=batch_size, num_mode=num_mode, resample_mu=True, test=True)
    exp_loss = 0
    test_losses = []
    print('Currently evaluating')
    for ng, g in enumerate(game_list):
        #print(f'Currently evaluating on test game {ng}') 
        torch_pop = g
        for psro_iter in range(psro_iters):

            # Define the weighting towards diversity as a function of the fixed population size
            payoff = torch_pop.get_metagame()# calculate previous strategies' pay off matrix
            payoff = payoff[None,None,].to(device)
            meta_nash = model(payoff)[0] # calculate nash equillibrium based on meta solver
            #print(meta_nash)
            torch_pop.add_new()# add new agent then optimize according to NE
            torch_pop.pop[-1].x.requires_grad = True
            for training_iter in range(train_iters):

                # Compute the expected return given that enemy plays agg_strat (using :k first strats)
                exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], meta_nash, len(meta_nash))
                # Loss
                loss = -(exp_payoff)

                # Optimise !
                psro_grad = torch.autograd.grad(loss, torch_pop.pop[-1].x, create_graph=False)
                torch_pop.pop[-1].x = torch_pop.pop[-1].x - lr * psro_grad[0]

        final_payoffs = torch_pop.get_metagame()
        final_payoffs = final_payoffs[None,None,].to(device)
        meta_nash_final = model(final_payoffs)[0]# calculate nash equillibrium based on meta solver
        final_exp = torch_pop.get_exploitability_test(meta_nash_final, exp_iters, test_lr).item()
        #print(final_exp)
        test_losses.append(final_exp)
        #exp_loss += torch_pop.get_exploitability_test(meta_nash_final, train_iters, test_lr).item()

        
    #exp_loss /= len(game_list)
    return test_losses

