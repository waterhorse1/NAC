import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.nn as nn
import torch.nn.functional as f
np.random.seed(0)

import copy

torch.set_printoptions(sci_mode=False)

if torch.cuda.is_available():
    device = 'cuda:0'
else:
    device = 'cpu'

class torchAgent(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.pop_logits = 0.2*torch.randn(n_actions).to(device)
        self.start_train()
    def start_train(self):
        self.pop_logits.requires_grad = True
    def end_train(self):
        self.pop_logits = self.pop_logits.detach()
    def get_dist(self):
        return f.softmax(self.pop_logits, dim=-1).float()

class torchPop(nn.Module):
  def __init__(self, pop_size, n_actions, seed=0, test=False):
    super().__init__()
    if test:
        torch.manual_seed(seed)
        np.random.seed(seed)
    self.n_actions = n_actions
    self.pop_size = pop_size
    self.softmax = nn.Softmax(dim=-1)
    self.pop = [torchAgent(n_actions).to(device) for _ in range(self.pop_size)]
    self.pop_size = len(self.pop)

  def add_agent(self):
    self.pop.append(torchAgent(self.n_actions).to(device))
    #self.pop_size += 1
    self.pop_size = len(self.pop)

  def remove_agent(self):
    self.pop.pop(-1)
    self.pop_size = len(self.pop)

  def reset(self, k):
    self.pop_size = k
    pop = copy.deepcopy(self.pop)
    self.pop = [pop[i] for i in range(self.pop_size)]

  def get_metagame(self, payoffs, k=None, numpy=False, shuffle=True):
    #if shuffle:
    #    random.shuffle(self.pop)
    if k==None:
        k = self.pop_size
    if numpy:
        with torch.no_grad():
            pop_strats = torch.stack(self.get_nash_dists(k))
            payoffs = torch.from_numpy(payoffs).to(device)
            payoffs.requires_grad = False
            metagame = pop_strats @ payoffs.float() @ pop_strats.t()
            return metagame.detach().cpu().numpy()
    else:
        pop_strats = torch.stack(self.get_nash_dists(k))
        payoffs = torch.from_numpy(payoffs).to(device)
        payoffs.requires_grad = False
        #with torch.no_grad():
        metagame = pop_strats @ payoffs.float() @ pop_strats.t()
    return metagame.float()

  def get_payoff_aggregate(self, agent1, metanash, payoffs, K):
    payoffs = torch.from_numpy(payoffs).to(device)
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

  def get_exploitability(self, metanash, payoffs, train_steps, lr):
    #print('exp')
    new_agent = torchAgent(self.n_actions)
    #new_agent.pop_logits = torch.zeros_like(new_agent.pop_logits)
    new_agent.start_train()
    for iter in range(train_steps):
        #print(iter)
        exp_payoff = self.get_payoff_aggregate(new_agent, metanash, payoffs, K = self.pop_size)

        loss = -(exp_payoff)
        #print(exp_payoff)
        grad = torch.autograd.grad(loss, [new_agent.pop_logits], create_graph=True)
        new_agent.pop_logits = new_agent.pop_logits - lr * grad[0]
    #true_exp = self.get_exploitability_direct(metanash, payoffs)
    exp1 = self.get_payoff_aggregate(new_agent, metanash, payoffs, K=self.pop_size)
    return 2 * exp1

  def pop2numpy(self, K=0):
    return torch.stack(self.get_nash_dists(K)).cpu().detach().numpy()

  def get_exploitability_direct(self, metanash, payoffs):
    numpy_pop = self.pop2numpy(self.pop_size)
    meta_nash_numpy = metanash.cpu().detach().numpy()
    emp_game_matrix = numpy_pop @ payoffs @ numpy_pop.T 
    strat = meta_nash_numpy @ numpy_pop
    test_br = get_br_to_strat(strat, payoffs=payoffs)
    exp = test_br @ payoffs @ strat
    return 2 * exp

def get_br_to_strat(strat, payoffs, verbose=False):
    row_weighted_payouts = strat@payoffs
    br = np.zeros_like(row_weighted_payouts)
    br[np.argmin(row_weighted_payouts)] = 1
    if verbose:
        print(row_weighted_payouts[np.argmin(row_weighted_payouts)], "exploitability")
    return br

