import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(0)
from numpy.random import RandomState

import copy
from scipy.linalg import circulant

torch.set_printoptions(sci_mode=False)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class torchAgent(nn.Module):
    def __init__(self, n_actions, nf_payoffs):
        super().__init__()
        self.x = 0.2*torch.randn(n_actions).to(device)
        self.nf_payoffs = nf_payoffs.to(device)
        self.n_actions = n_actions
        self.start_train()
    def start_train(self):
        self.x.requires_grad = True
    def end_train(self):
        self.x = self.x.detach()
    def get_dist(self):
        return f.softmax(self.x, dim=-1).float()

class TorchPop_gos(nn.Module):
  def __init__(self, game_args_dict, nf_payoffs, device, seed=0, test=False):
    super().__init__()
    if test:
        torch.manual_seed(seed)
        np.random.seed(seed)
    self.n_actions = game_args_dict['gos_dim']
    self.nf_payoffs = nf_payoffs.to(device)
    self.pop_size = 2
    self.softmax = nn.Softmax(dim=-1)
    self.pop = [torchAgent(self.n_actions, self.nf_payoffs).to(device) for _ in range(self.pop_size)]
    self.pop_size = len(self.pop)

  def add_agent(self):
    self.pop.append(torchAgent(self.n_actions, self.nf_payoffs).to(device))
    #self.pop_size += 1
    self.pop_size = len(self.pop)

  def remove_agent(self):
    self.pop.pop(-1)
    self.pop_size = len(self.pop)

  def detach_window(self, num_stop, window_size):
    for f in range(2):
      self.pop[f].end_train()
    for k in range(num_stop, num_stop + window_size):
      self.pop[k].end_train()

  def reset(self, k):
    self.pop_size = k
    pop = copy.deepcopy(self.pop)
    self.pop = [pop[i] for i in range(self.pop_size)]

  def agg_agents(self, metanash):
    agg_agent = torchAgent(self.n_actions, self.nf_payoffs)
    agg_agent.x = metanash[0] * self.pop[0].x
    for k in range(1, self.pop_size):
      agg_agent.x += metanash[k]*self.pop[k].x
    return agg_agent

  def psro_popn_update(self, meta_nash, train_iters, train_lr, lam, norm_break_val, implicit=False):
    if implicit == False:
      self.add_agent()
      self.pop[-1].start_train()
      for _ in range(train_iters):
        exp_payoff = self.get_payoff_aggregate(self.pop[-1], meta_nash, len(meta_nash))
        loss = -(exp_payoff)

        train_grad = torch.autograd.grad(loss, [self.pop[-1].x], create_graph=True)
        self.pop[-1].x = self.pop[-1].x - train_lr * train_grad[0]
    
    elif implicit == True:
      agg_agent = self.agg_agents(meta_nash)
      best_response_trainer = implicit_best_responder()
      _, br_agent = best_response_trainer(agg_agent.x, agg_agent, lam, train_lr, train_iters)
      self.add_agent()
      self.pop[-1].x = br_agent

  def get_metagame(self, k=None, numpy=False, shuffle=True):
    #if shuffle:
    #    random.shuffle(self.pop)
    if k==None:
        k = self.pop_size
    if numpy:
        with torch.no_grad():
            pop_strats = torch.stack(self.get_nash_dists(k))
            payoffs = torch.from_numpy(self.nf_payoffs).to(device)
            payoffs.requires_grad = False
            metagame = pop_strats @ payoffs.float() @ pop_strats.t()
            return metagame.detach().cpu().numpy()
    else:
        pop_strats = torch.stack(self.get_nash_dists(k))
        payoffs = self.nf_payoffs.to(device)
        payoffs.requires_grad = False
        #with torch.no_grad():
        metagame = pop_strats @ payoffs.float() @ pop_strats.t()
    return metagame.float()[None,None,]

  def get_payoff_aggregate(self, agent1, metanash, K):
    payoffs = self.nf_payoffs.to(device)
    payoffs.requires_grad = False
    pop_strats = self.get_nash_dists(K)
    pop_strats = [_data for _data in pop_strats]
    weighted_strats = []
    for i in range(K):
        weighted_strats.append(metanash[i][None] * pop_strats[i])
    agg_strat = torch.sum(torch.stack(weighted_strats), dim=0)
    exp_payoff = agent1.get_dist() @ payoffs.float() @ agg_strat

    return exp_payoff

  def get_nash_dists(self, K):
    dists = []
    for i in range(K):
        dists.append(self.pop[i].get_dist())
    return dists

  def get_exploitability(self, metanash, train_steps, norm_break_val, lr, test=False, implicit=False, lam=0.1):
    
    if implicit == False or test == True:
      new_agent = torchAgent(self.n_actions, self.nf_payoffs)
      new_agent.start_train()
      for iter in range(train_steps):
          exp_payoff = self.get_payoff_aggregate(new_agent, metanash, K = self.pop_size)

          loss = -(exp_payoff)
          grad = torch.autograd.grad(loss, [new_agent.x], create_graph=True)
          new_agent.x = new_agent.x - lr * grad[0]
      if test:
        with torch.no_grad():
          exp = self.get_payoff_aggregate(new_agent, metanash, K=self.pop_size)
      else:
        exp = self.get_payoff_aggregate(new_agent, metanash, K=self.pop_size)

    elif implicit == True:
      agg_agent = self.agg_agents(metanash)
      exp_best_response_trainer = implicit_best_responder()
      exp, _ = exp_best_response_trainer(agg_agent.x, agg_agent, lam, lr, train_steps)
      
    return 2 * exp

  def pop2numpy(self, K=0):
    return torch.stack(self.get_nash_dists(K)).cpu().detach().numpy()

def gen_gos_payoffs(dim, seed=0, test=False):
    if test:
      torch.manual_seed(seed)
      np.random.seed(seed)
    
    np.random.seed()
    W = np.random.randn(dim, dim)
    S = np.random.randn(dim, 1)
    payoffs = (W - W.T) + S - S.T
    return torch.Tensor(payoffs)  

def sample_gos_games(game_args_dict):
    batch_size = game_args_dict['batch_size']
    gos_dim = game_args_dict['gos_dim']
    nf_payoffs = [gen_gos_payoffs(gos_dim) for _ in range(batch_size)]
    game_list = [TorchPop_gos(game_args_dict, nf_payoffs[k], device, seed=0, test=False) for k in range(batch_size)]
    return game_list, nf_payoffs

def get_payoff(br_agent, agg_agent, nf_payoffs):
    return f.softmax(br_agent, dim=-1).float() @ nf_payoffs @ f.softmax(agg_agent, dim=-1).float()

def get_best_response(agg_agent_logits, agg_agent, lam, lr, train_iters):
    torch.manual_seed(0)
    br = torchAgent(agg_agent.n_actions, agg_agent.nf_payoffs)
    br.start_train()
    opt = torch.optim.Adam([br.x], lr=lr)
    for _ in range(train_iters):
        payoff = get_payoff(br.x, agg_agent_logits, agg_agent.nf_payoffs)
        n = lam/2 * torch.norm(br.x)**2
        loss = - payoff + n
        opt.zero_grad()
        loss.backward(retain_graph=True)
        opt.step()
    return br.x

class implicit_best_responder(nn.Module):
    def __init__(self):
        super(implicit_best_responder, self).__init__()
    def forward(self, agg_agent_logits, agg_agent, lam, lr, train_iters):
        br = get_best_response(agg_agent_logits, agg_agent, lam, lr, train_iters)
        r = get_payoff(br, agg_agent_logits, agg_agent.nf_payoffs)
        g = 1/lam * torch.autograd.grad(r, br, create_graph=True)[0]
        br = (br - g).detach() + g
        r = get_payoff(br, agg_agent_logits, agg_agent.nf_payoffs)
        return r, br
