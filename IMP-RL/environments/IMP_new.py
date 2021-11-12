import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as f
from numpy.random import RandomState

import copy
from scipy.linalg import circulant

torch.set_printoptions(sci_mode=False)

#if torch.cuda.is_available():
#    device = 'cuda:0'
#else:
#    device = 'cpu'
device = 'cpu'
import gym
import numpy as np

from gym.spaces import Discrete, Tuple

from environments.common import OneHot
from torch.distributions import Bernoulli
from copy import deepcopy

from environments.linear_baseline import LinearFeatureBaseline, get_return

class IteratedMatchingPennies(gym.Env):
    """
    A two-agent vectorized environment.
    Possible actions for each agent are (C)ooperate and (D)efect.
    """
    # Possible actions
    NUM_AGENTS = 2
    NUM_ACTIONS = 2
    NUM_STATES = 5

    def __init__(self, payoff, max_steps, batch_size=1):
        self.max_steps = max_steps
        self.batch_size = batch_size
        self.payout_mat = payoff#np.array([[1,-1],[-1,1]])
        self.states = np.array([[1,2],[3,4]])

        self.action_space = Tuple([
            Discrete(self.NUM_ACTIONS) for _ in range(self.NUM_AGENTS)
        ])
        self.observation_space = Tuple([
            OneHot(self.NUM_STATES) for _ in range(self.NUM_AGENTS)
        ])
        self.available_actions = [
            np.ones((batch_size, self.NUM_ACTIONS), dtype=int)
            for _ in range(self.NUM_AGENTS)
        ]

        self.step_count = None

    def reset(self):
        self.step_count = 0
        init_state = np.zeros(self.batch_size)
        observation = [init_state, init_state]
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observation, info

    def step(self, action):
        ac0, ac1 = action
        self.step_count += 1

        r0 = self.payout_mat[ac0, ac1]
        r1 = -self.payout_mat[ac0, ac1]
        s0 = self.states[ac0, ac1]
        s1 = self.states[ac1, ac0]
        observation = [s0, s1]
        reward = [r0, r1]
        done = (self.step_count == self.max_steps)
        info = [{'available_actions': aa} for aa in self.available_actions]
        return observation, reward, done, info
'''   
class Hp():
    def __init__(self):
        self.lr_out = 0.2
        self.lr_in = 0.1
        self.lr_in_exp = 0.3
        self.lr_v = 0.1
        self.gamma = 1
        self.n_update = 20
        self.n_update_exp = 25
        self.len_rollout = 150
        self.batch_size = 64
        self.use_baseline = True
hp = Hp()
'''

class Agent(nn.Module):
    def __init__(self, device):
        # init theta and its optimizer
        super().__init__()
        self.theta =  0.5 * torch.randn(5, requires_grad=True).to(device)
    def start_train(self):
        pass
    def end_train(self):
        self.theta = self.theta.detach()

def magic_box(x):
    return torch.exp(x - x.detach())

class Memory():
    def __init__(self):
        self.states = []
        self.self_logprobs = []
        self.other_logprobs = []
        self.values = []
        self.rewards = []

    def add(self, state, lp, other_lp, v, r):
        self.states.append(state)
        self.self_logprobs.append(lp)
        self.other_logprobs.append(other_lp)
        self.values.append(v)
        self.rewards.append(r)

    def dice_objective(self, gamma=1, use_baseline=True):
        self_logprobs = torch.stack(self.self_logprobs, dim=1)
        #print(self_logprobs.shape)
        other_logprobs = torch.stack(self.other_logprobs, dim=1)
        values = torch.stack(self.values, dim=1)
        rewards = torch.stack(self.rewards, dim=1)

        # apply discount:
        cum_discount = torch.cumprod(gamma * torch.ones(*rewards.size()), dim=1)/gamma
        discounted_rewards = rewards * cum_discount
        discounted_values = values * cum_discount

        # stochastics nodes involved in rewards dependencies:
        #print(self_logprobs.shape)#128 * 150
        dependencies = torch.cumsum(self_logprobs + other_logprobs, dim=1)

        # logprob of each stochastic nodes:
        stochastic_nodes = self_logprobs + other_logprobs

        # dice objective:
        dice_objective = torch.mean(torch.sum(magic_box(dependencies) * discounted_rewards, dim=1))

        if use_baseline:
            # variance_reduction:
            baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_values, dim=1))
            dice_objective = dice_objective + baseline_term

        return -dice_objective # want to minimize -objective
    
class TorchPop_imp(nn.Module):
    def __init__(self, game_args, payoff, device, init=1, seed=0, test=False):
        super().__init__()
        if test:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        # game setup
        self.len_rollout = game_args['len_rollout']
        self.gamma = game_args['gamma']
        self.use_baseline = game_args['use_baseline']
        self.batch_size = game_args['rl_batch_size']
        
        #train setup
        self.test = test
        self.device = device
        self.init = init
        self.pop1 = [Agent(device) for _ in range(init)]
        self.pop2 = [Agent(device) for _ in range(init)]
        self.pop_size = len(self.pop1)
        self.metagame = None
        self.imp = IteratedMatchingPennies(payoff, self.len_rollout, self.batch_size)
        self.imp_exp = IteratedMatchingPennies(payoff, self.len_rollout, 64)
        self.linear_baseline = LinearFeatureBaseline(self.gamma, self.imp.NUM_STATES)
        self.exploit = False
        
        for f in range(self.init):
            self.pop1[f].end_train()
            self.pop2[f].end_train()

    def add_agent(self):
        self.pop1.append(Agent(self.device))
        self.pop2.append(Agent(self.device))
        self.pop_size += 1
    
    def detach_window(self, num_stop, window_size):
        for k in range(num_stop, num_stop + window_size):
            self.pop1[k].end_train()
            self.pop2[k].end_train()

    def act(self, batch_states, theta):
        batch_states = torch.from_numpy(batch_states).long()
        probs = torch.sigmoid(theta)[batch_states]
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions

    def act_aggregation(self, batch_states, agent_pop, meta_nash):
        batch_states = torch.from_numpy(batch_states).long()
        probs = 0
        for i in range(min(len(agent_pop), len(meta_nash))):
            probs += torch.sigmoid(agent_pop[i].theta)[batch_states] * meta_nash[i]
        m = Bernoulli(1-probs)
        actions = m.sample()
        log_probs_actions = m.log_prob(actions)
        return actions.numpy().astype(int), log_probs_actions
    
    def step_1_1(self, theta1, theta2, grad=False):
        if self.exploit:
            imp = self.imp_exp
        else:
            imp = self.imp
        if not grad:
            # just to evaluate progress without gradient:
            (s1, s2), _ = imp.reset()
            dice_memory = Memory()
            for t in range(self.len_rollout):
                a1, lp1 = self.act(s1, theta1)
                a2, lp2 = self.act(s2, theta2)
                (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                s1_one = np.eye(self.imp.NUM_STATES)[s1]
                # cumulate scores
                dice_memory.add(s1_one, lp1, lp2, torch.zeros_like(torch.from_numpy(r1)), torch.from_numpy(r1).float())  
            reward = dice_memory.rewards
            score1 = get_return(reward, self.gamma)[:,0].mean()/self.len_rollout
            score2 = -score1
            
            return (score1, score2)
                
        else:
            (s1, s2), _ = imp.reset()
            dice_memory = Memory()
            for t in range(self.len_rollout):
                a1, lp1 = self.act(s1, theta1)
                a2, lp2 = self.act(s2, theta2)

                (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                s1_one = np.eye(self.imp.NUM_STATES)[s1]
                dice_memory.add(s1_one, lp1, lp2, torch.zeros_like(torch.from_numpy(r1)), torch.from_numpy(r1).float())  
            #update baseline value by linear_baseline
            obs = dice_memory.states
            reward = dice_memory.rewards
            value = self.linear_baseline(obs, reward)#128*150
            dice_memory.values = [value[:, i] for i in range(value.shape[-1])]
            #calculate
            dice_objective = dice_memory.dice_objective(self.gamma, self.use_baseline)/self.len_rollout
        return (-dice_objective, dice_objective)
    
    def step_n_1(self, theta1, theta2, grad=False, meta_nash1=None):
        if self.exploit:
            imp = self.imp_exp
        else:
            imp = self.imp
        if not grad:
            # just to evaluate progress without gradient:
            score1_all = []
            for agent in theta1:
                (s1, s2), _ = imp.reset()
                dice_memory = Memory()
                for t in range(self.len_rollout):
                    a1, lp1 = self.act(s1, agent.theta)
                    a2, lp2 = self.act(s2, theta2)

                    (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                    s1_one = np.eye(self.imp.NUM_STATES)[s1]
                    # cumulate scores
                    dice_memory.add(s1_one, lp1, lp2, torch.zeros_like(torch.from_numpy(r1)), torch.from_numpy(r1).float())  
                reward = dice_memory.rewards
                score1 = get_return(reward, self.gamma)[:,0].mean()/self.len_rollout
                score1_all.append(score1)
            score1 = 0
            for i in range(len(meta_nash1)):
                score1 = score1 + meta_nash1[i] * score1_all[i]                           
            score2 = -score1
            
            return (score1, score2)
                
        else:
            score1_all = []
            for agent in theta1:
                (s1, s2), _ = imp.reset()
                dice_memory = Memory()
                for t in range(self.len_rollout):
                    a1, lp1 = self.act(s1, agent.theta)
                    a2, lp2 = self.act(s2, theta2)
                    (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                    s1_one = np.eye(self.imp.NUM_STATES)[s1]
                    dice_memory.add(s1_one, lp1, lp2, torch.zeros_like(torch.from_numpy(r1)), torch.from_numpy(r1).float())  
                #update baseline value by linear_baseline
                obs = dice_memory.states
                reward = dice_memory.rewards
                value = self.linear_baseline(obs, reward)#128*150
                dice_memory.values = [value[:, i] for i in range(value.shape[-1])]
                #calculate
                dice_objective = dice_memory.dice_objective(self.gamma, self.use_baseline)/self.len_rollout
                score1_all.append(dice_objective)
            score1 = 0
            for i in range(len(meta_nash1)):
                score1 = score1 + meta_nash1[i] * score1_all[i]                           
            
        return (-score1, score1)
    
    def step_1_n(self, theta1, theta2, grad=False, meta_nash2=None):
        if self.exploit:
            imp = self.imp_exp
        else:
            imp = self.imp
        if not grad:
            # just to evaluate progress without gradient:
            score1_all = []
            for agent in theta2:
                (s1, s2), _ = imp.reset()
                dice_memory = Memory()
                for t in range(self.len_rollout):
                    a1, lp1 = self.act(s1, theta1)
                    a2, lp2 = self.act(s2, agent.theta)

                    (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                    s1_one = np.eye(self.imp.NUM_STATES)[s1]
                    # cumulate scores
                    dice_memory.add(s1_one, lp1, lp2, torch.zeros_like(torch.from_numpy(r1)), torch.from_numpy(r1).float())  
                reward = dice_memory.rewards
                score1 = get_return(reward, self.gamma)[:,0].mean()/self.len_rollout
                score1_all.append(score1)
            score1 = 0
            for i in range(len(meta_nash2)):
                score1 = score1 + meta_nash2[i] * score1_all[i]                           
            score2 = -score1
            
            return (score1, -score1)
                
        else:
            score1_all = []
            for agent in theta2:
                (s1, s2), _ = imp.reset()
                dice_memory = Memory()
                for t in range(self.len_rollout):
                    a1, lp1 = self.act(s1, theta1)
                    a2, lp2 = self.act(s2, agent.theta)
                    (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                    s1_one = np.eye(self.imp.NUM_STATES)[s1]
                    dice_memory.add(s1_one, lp1, lp2, torch.zeros_like(torch.from_numpy(r1)), torch.from_numpy(r1).float())  
                #update baseline value by linear_baseline
                obs = dice_memory.states
                reward = dice_memory.rewards
                value = self.linear_baseline(obs, reward)#128*150
                dice_memory.values = [value[:, i] for i in range(value.shape[-1])]
                #calculate
                dice_objective = dice_memory.dice_objective(self.gamma, self.use_baseline)/self.len_rollout
                score1_all.append(dice_objective)
            score1 = 0
            for i in range(len(meta_nash2)):
                score1 = score1 + meta_nash2[i] * score1_all[i]                           
            
        return (-score1, score1)
    

    def br1(self, agent1, agent2_pop, metanash, train_iter, lr):
        if self.exploit:
            imp = self.imp_exp
        else:
            imp = self.imp
        for update in range(train_iter):
            dice_objective = []
            for agent in agent2_pop:    
                (s1, s2), _ = imp.reset()
                dice_memory = Memory()
                for t in range(self.len_rollout):
                    #a2, lp2 = self.act_aggregation(s2, agent2_pop, metanash)
                    a2, lp2 = self.act(s2, agent.theta)
                    a1, lp1 = self.act(s1, agent1.theta)
                    (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                    s1_one = np.eye(imp.NUM_STATES)[s1]
                    dice_memory.add(s1_one, lp1, lp2, torch.zeros_like(torch.from_numpy(r1)), torch.from_numpy(r1).float())  
                
                #update baseline value by linear_baseline
                obs = dice_memory.states
                reward = dice_memory.rewards
                value = self.linear_baseline(obs, reward)#128*150
                dice_memory.values = [value[:, i] for i in range(value.shape[-1])]
            
                #calculate dice objective
                dice_objective.append(dice_memory.dice_objective(self.gamma, self.use_baseline)/self.len_rollout)
            all_objective = 0 
            for i in range(len(metanash)):
                all_objective = all_objective + dice_objective[i] * metanash[i]
            grad = torch.autograd.grad(all_objective, agent1.theta, create_graph= not self.test)[0]

            agent1.theta = agent1.theta - lr * grad
            # evaluate progress:
            #score = step(agent1.theta, agent2_pop, meta_nash2 = metanash)
                
        score = self.step_1_n(agent1.theta, agent2_pop, grad=True, meta_nash2 = metanash)
        return agent1, score[0]

    def br2(self, agent2, agent1_pop, metanash, train_iter, lr):
        if self.exploit:
            imp = self.imp_exp
        else:
            imp = self.imp
        for update in range(train_iter):
            dice_objective = []
            for agent in agent1_pop:
                (s1, s2), _ = imp.reset()
                dice_memory = Memory()
                for t in range(self.len_rollout):
                    a2, lp2 = self.act(s2, agent2.theta)
                    a1, lp1 = self.act(s1, agent.theta)
                    (s1, s2), (r1, r2),_,_ = imp.step((a1, a2))
                    s2_one = np.eye(imp.NUM_STATES)[s2]
                    dice_memory.add(s2_one, lp2, lp1, torch.zeros_like(torch.from_numpy(r2)), torch.from_numpy(r2).float())
                
                #update baseline value by linear_baseline
                obs = dice_memory.states
                reward = dice_memory.rewards
                value = self.linear_baseline(obs, reward)#128*150
                dice_memory.values = [value[:, i] for i in range(value.shape[-1])]

                dice_objective.append(dice_memory.dice_objective(self.gamma, self.use_baseline)/self.len_rollout)
            all_objective = 0 
            for i in range(len(metanash)):
                all_objective = all_objective + dice_objective[i] * metanash[i]
            grad = torch.autograd.grad(all_objective, agent2.theta, create_graph= not self.test)[0]
            agent2.theta = agent2.theta - lr * grad
            # evaluate progress:
                
        score = self.step_n_1(agent1_pop, agent2.theta, grad=True, meta_nash1 = metanash)

        return agent2, score[1]

    def psro_popn_update(self, meta_nash1, meta_nash2, train_iter, lr):
        self.add_agent()
        self.pop1[-1].start_train()
        self.pop2[-1].start_train()
        _, _ = self.br1(self.pop1[-1], self.pop2[:-1], meta_nash2, train_iter, lr)
        _, _ = self.br2(self.pop2[-1], self.pop1[:-1], meta_nash1, train_iter, lr)
        
    def psro_popn_update_1(self, meta_nash2, train_iter, lr):
        self.add_agent()
        self.pop1[-1].start_train()
        _, _ = self.br1(self.pop1[-1], self.pop2[:-1], meta_nash2, train_iter, lr)
        
    def psro_popn_update_2(self, meta_nash1, train_iter, lr):
        self.add_agent()
        self.pop2[-1].start_train()
        _, _ = self.br2(self.pop2[-1], self.pop1[:-1], meta_nash1, train_iter, lr)

    def get_metagame(self, numpy=False):
        if self.metagame is not None:
            if not len(self.metagame) == self.pop_size:         
                if numpy:
                    self.metagame = np.pad(self.metagame, ((0,1),(0,1)))
                    for i in range(len(self.pop2)):
                        agent1 = self.pop1[-1]
                        agent2 = self.pop2[i]
                        score = self.step_1_1(agent1.theta, agent2.theta)[0]
                        self.metagame[-1, i] = score
                    for j in range(len(self.pop2)-1):
                        agent1 = self.pop1[j]
                        agent2 = self.pop2[-1]
                        score = self.step_1_1(agent1.theta, agent2.theta)[0]
                        self.metagame[j, -1] = score
                else:
                    self.metagame = f.pad(self.metagame, (0,1,0,1))
                    for i in range(len(self.pop2)):
                        agent1 = self.pop1[-1]
                        agent2 = self.pop2[i]
                        score = self.step_1_1(agent1.theta, agent2.theta, grad=True)[0]
                        self.metagame[-1, i] = score
                    for j in range(len(self.pop2)-1):
                        agent1 = self.pop1[j]
                        agent2 = self.pop2[-1]
                        score = self.step_1_1(agent1.theta, agent2.theta, grad=True)[0]
                        self.metagame[j, -1] = score
        else:
            if numpy:
                k = self.pop_size
                self.metagame = np.zeros([k,k])
                for i in range(k):
                    for j in range(k):
                        agent1 = self.pop1[i]
                        agent2 = self.pop2[j]
                        score = self.step_1_1(agent1.theta, agent2.theta)[0]
                        self.metagame[i,j] = score
            else:
                k = self.pop_size
                self.metagame = torch.zeros([k,k]).to(self.device)
                for i in range(k):
                    for j in range(k):
                        agent1 = self.pop1[i]
                        agent2 = self.pop2[j]
                        score = self.step_1_1(agent1.theta, agent2.theta, grad=True)[0]
                        self.metagame[i,j] = score      
        if numpy:
            return self.metagame
        else:
            return self.metagame.float()[None,None,]


    def get_exploitability(self, meta_nash1, meta_nash2, train_iter, lr):
        agent_exp1 = Agent(self.device)
        agent_exp2 = Agent(self.device)
        agent_exp1.start_train()
        agent_exp2.start_train()
        self.exploit = True
        a1, r1 = self.br1(agent_exp1, self.pop2, meta_nash2, train_iter, lr)
        a2, r2 = self.br2(agent_exp2, self.pop1, meta_nash1, train_iter, lr)
        self.exploit = False
        return r1+r2

def gen_imp_payoffs(seed=0, test=False):
    if test:
        torch.manual_seed(seed)
        np.random.seed(seed)
    '''
    num = np.random.uniform(0.5, 3)
    payoffs = np.zeros([2,2])
    payoffs[0][0] = num
    payoffs[0][1] = -num
    payoffs[1][0] = -num
    payoffs[1][1] = num
    '''
    payoffs = np.zeros([2,2])
    num = np.random.uniform(0.5, 2, 2)
    payoffs[0][0] = num[0]
    payoffs[0][1] = -num[0]
    payoffs[1][0] = -num[1]
    payoffs[1][1] = num[1]
    return payoffs  

def sample_imp_games(game_args_dict):
    batch_size = game_args_dict['batch_size']
    nf_payoffs = [gen_imp_payoffs() for _ in range(batch_size)]
    game_list = [TorchPop_imp(game_args_dict, nf_payoffs[i], device, seed=0, test=False) for i in range(batch_size)]
    return game_list