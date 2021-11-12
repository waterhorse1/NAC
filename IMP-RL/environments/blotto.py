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
    device_train = 'cuda:0'
    device = 'cuda:0'
else:
    device_train = 'cpu'
    device = 'cpu'
device_test = 'cpu'


class DLottoAgent(nn.Module):
    def __init__(self, nb_positions, nb_customers, C, device):
        super().__init__()
        # Positions
        self.device = device
        self.nb_pos = nb_positions
        self.nb_c = nb_customers
        self.C = C
        self.v = 2*(torch.rand(nb_positions, 2).float()-0.5).to(self.device)
        # Weights
        self.p_logits = torch.randn(nb_positions).to(self.device)
        self.end_train()
    def start_train(self):
        self.v.requires_grad = True
        self.p_logits.requires_grad = True
    def end_train(self):
        self.v = self.v.detach()
        self.p_logits = self.p_logits.detach()
    def get_p(self):
        return self.nb_pos * f.softmax(self.p_logits, dim=-1)


class TorchPop_blotto(nn.Module):
    def __init__(self, game_args_dict, device, seed=0, test=False):
        super().__init__()
        if test:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.nb_customers = game_args_dict['nb_customers']
        self.nb_positions = game_args_dict['nb_positions']
        self.nf_payoffs = game_args_dict['nf_payoffs']
        self.pop_size = 2
        self.softmax = nn.Softmax(dim=-1)
        self.device = device
        # Costumers
        self.C = 2*(torch.rand(self.nb_customers, 2).float()-0.5).to(self.device)
        # Agents
        self.pop = [DLottoAgent(self.nb_positions, self.nb_customers, self.C, device) for _ in range(self.pop_size)]

    def reset(self, k):
        self.pop_size = k
        pop = copy.deepcopy(self.pop)
        self.pop = [pop[i] for i in range(self.pop_size)]
        
    def add_agent(self):
        self.pop.append(DLottoAgent(self.nb_positions, self.nb_customers, self.C, self.device))
        self.pop_size += 1

    def agg_agents(self, meta_nash):
        agg_agent = DLottoAgent(self.nb_position, self.device)
        agg_agent.v = meta_nash[0] * self.pop[0].v
        agg_agent.p_logits = meta_nash[0] * self.pop[0].p_logits
        for k in range(1, self.pop_size):
            agg_agent.v += meta_nash[k]*self.pop[k].v
            agg_agent.p_logits += meta_nash[k]*self.pop[k].p_logits
        return agg_agent

    def detach_window(self, num_stop, window_size):
        for f in range(2):
            self.pop[f].end_train()
        for k in range(num_stop, num_stop + window_size):
            self.pop[k].end_train()

    def psro_popn_update(self, meta_nash, train_iters, train_lr, lam, implicit=False):
        self.add_agent()
        self.pop[-1].start_train()
        for _ in range(train_iters):
            exp_payoff = self.get_payoff_aggregate(self.pop[-1], meta_nash, len(meta_nash))
            loss = -(exp_payoff)

            train_grad = torch.autograd.grad(loss, [self.pop[-1].v, self.pop[-1].p_logits], create_graph=True)
            self.pop[-1].v = self.pop[-1].v - train_lr * train_grad[0]
            self.pop[-1].p_logits = self.pop[-1].p_logits - train_lr * train_grad[1]
        
    def get_payoff(self, agent1, agent2):
        # Return the payoff loss (-payoff) of agent i (how bad is agent i against agent -i)
        c = self.C.unsqueeze(dim=1).repeat(1, self.nb_positions, 1)
        p = agent1.get_p().repeat(self.nb_customers, 1)
        q = agent2.get_p().repeat(self.nb_customers, 1)
        v = agent1.v.unsqueeze(dim=0).repeat(self.nb_customers, 1, 1)
        w = agent2.v.unsqueeze(dim=0).repeat(self.nb_customers, 1, 1)
        cw_dis = -torch.norm(c - w, dim=-1)
        cv_dis = -torch.norm(c - v, dim=-1)
        dis = torch.cat([cv_dis, cw_dis], dim=-1)
        dis_soft = f.softmax(dis, dim=-1)
        v, w = torch.split(dis_soft, self.nb_positions, dim=-1)
        payoff = torch.sum(p*v) - torch.sum(q*w)
        return payoff

    def get_payoff_aggregate(self, agent1, metanash, K, invert=False):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        c = self.C.unsqueeze(dim=1).unsqueeze(dim=0).repeat(K, 1, self.nb_positions, 1)
        p = agent1.get_p().repeat(K, self.nb_customers, 1)
        v = agent1.v.repeat(K, self.nb_customers, 1, 1)
        q, w = [], []
        for k in range(K):
            q.append(self.pop[k].get_p())
            w.append(self.pop[k].v)
        q = torch.stack(q, dim=0).unsqueeze(dim=1).repeat(1, self.nb_customers, 1)
        w = torch.stack(w, dim=0).unsqueeze(dim=1).repeat(1, self.nb_customers, 1, 1)
        cw_dis = -torch.norm(c - w, dim=-1)
        cv_dis = -torch.norm(c - v, dim=-1)
        dis = torch.cat([cv_dis, cw_dis], dim=-1)
        dis_soft = f.softmax(dis, dim=-1)
        v, w = torch.split(dis_soft, self.nb_positions, dim=-1)#k * 9 * 20
        payoff = torch.sum(torch.sum(p*v, dim=-1), dim=-1) - torch.sum(torch.sum(q*w, dim=-1), dim=-1)#
        payoff = torch.sum(payoff.reshape(-1) * metanash)
        return payoff
        
    def get_metagame(self, k=None, numpy=False, no_aug=False):
        if k==None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
                    return metagame.detach().numpy()
        else:
            metagame = torch.zeros(k, k)
            #with torch.no_grad():
            #if not no_aug:
            #    random.shuffle(self.pop)
            for m in range(k):
                for j in range(k):
                    metagame[m, j] = self.get_payoff(self.pop[m], self.pop[j])
            return metagame[None,None,]

    def get_exploitability(self, metanash,  train_steps=25, lr=35, test=False, implicit=False):
        cg = not test
        lr = lr
        new_agent = DLottoAgent(self.nb_positions, self.nb_customers, self.C, self.device)
        new_agent.start_train()
        for iter in range(train_steps):
            # Compute the expected return given that enemy plays agg_strat (using :k first strats)
            exp_payoff = self.get_payoff_aggregate(new_agent, metanash, K = self.pop_size, invert=False)

            # Loss
            loss = -(exp_payoff)

            # Optimise !
            grad = torch.autograd.grad(loss, [new_agent.v, new_agent.p_logits], create_graph=cg)
            new_agent.v = new_agent.v - lr * grad[0]
            new_agent.p_logits = new_agent.p_logits - lr * grad[1]
        if test:
            with torch.no_grad():
                exp1 = self.get_payoff_aggregate(new_agent, metanash, K=self.pop_size, invert=False)
        else:
            exp1 = self.get_payoff_aggregate(new_agent, metanash, K=self.pop_size)
        return 2 * exp1 #it's symmetric game

def sample_game(num, nb_costumer, nb_position, seed=0, test=False):
    game_list = [DlottoPop(nb_costumer, nb_position, device=device, seed=(seed + i), test=test) for i in range(num)]
    return game_list

def sample_blotto_games(game_args_dict):
    batch_size = game_args_dict['batch_size']
    test = game_args_dict['testing']
    seed = 0
    game_list = [TorchPop_blotto(game_args_dict, device=device, seed=(seed + i), test=test) for i in range(batch_size)]
    return game_list, None

