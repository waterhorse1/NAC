import numpy as np
np.set_printoptions(suppress=True)
import gym
from typing import List, Dict
from gym.spaces import Discrete, Box
import random
import itertools
import torch
import torch.nn.functional as f
from agent.ppo_class import Agent, PPOTrainer

ac_kwargs = dict(hidden_sizes=[64] * 2)

infostates = ['J', 'JB', 'JP', 'JPB', 'Q', 'QB',  # turn for player 1
              'QP', 'QPB', 'K', 'KB', 'KP', 'KPB']  # turn for player 2
actions_kp = ['P', 'B']
kuhn_cards = ['J', 'Q', 'K']

def infostate2vector(infostate):
    # infosate = card + history just the sum of two strings
    assert infostate in infostates, infostate + " not in infostates"
    idx = infostates.index(infostate)
    vec = np.zeros((len(infostates),), dtype=np.float32)
    vec[idx] = 1.
    return vec

class KuhnPoker:

    def __init__(self):
        self.done = False
        self.current_player = 0
        self.current_cards = None
        self.history = None

        self.action_space = Discrete(2)
        self.observation_space = Box(low=0., high=1., shape=(len(infostates),), dtype=np.float32)

        self.infostates = infostates
        self.actions = actions_kp
        self.cards = kuhn_cards

    @staticmethod
    def is_terminal(history: str) -> bool:
        return history in ['BP', 'BB', 'PP', 'PBB', 'PBP']

    @staticmethod
    def get_payoff(history: str, cards: List[str]) -> int:
        """ATTENTION: this gets payoff for 'active' player in terminal history"""
        if history in ['BP', 'PBP']:
            return +1
        else:  # PP or BB or PBB
            payoff = 2 if 'B' in history else 1
            active_player = len(history) % 2
            player_card = cards[active_player]
            opponent_card = cards[(active_player + 1) % 2]
            if player_card == 'K' or opponent_card == 'J':
                return payoff
            else:
                return -payoff

    def step(self, a):
        self.history += self.actions[a]  # Update history
        self.current_player = 1 - self.current_player  # Next player

        if self.is_terminal(self.history):
            self.done = True
            r = ((-1)**self.current_player)*self.get_payoff(self.history, self.current_cards)  # Return reward for player 1
            next_obs = [0., 0.]
        else:
            r = 0.
            next_obs = [infostate2vector(self.current_cards[self.current_player]+self.history),
                        self.current_cards[self.current_player]+self.history]

        info = None

        return next_obs, r, self.done, info

    def reset(self, set_cards=None):
        self.current_player = 0
        self.done = False
        self.current_cards = random.sample(self.cards, 2) if set_cards is None else set_cards
        self.history= ''
        return [infostate2vector(self.current_cards[0]+self.history), self.current_cards[0]+self.history]

def calc_ev_kp(p1_strat, p2_strat, cards, history, active_player):
    """ Returns value for player 2!! (p2_strat) """
    if KuhnPoker.is_terminal(history):
        return -KuhnPoker.get_payoff(history, cards)
    my_card = cards[active_player]
    next_player = (active_player + 1) % 2
    if active_player == 0:
        strat = p1_strat[my_card + history]
    else:
        strat = p2_strat[my_card + history]
    return -np.dot(strat, [calc_ev_kp(p1_strat, p2_strat, cards, history + a, next_player) for a in actions_kp])

def calc_best_response_kp(agg_hagent, br_strat_map, br_player, cards, history, active_player, prob):
    """
    after chance node, so only decision nodes and terminal nodes left in game tree
    """
    if KuhnPoker.is_terminal(history):
        return -KuhnPoker.get_payoff(history, cards)
    key = cards[active_player] + history
    next_player = (active_player + 1) % 2
    if active_player == br_player:
        vals = [calc_best_response_kp(agg_hagent, br_strat_map, br_player, cards, history + action,
                                   next_player, prob) for action in actions_kp]
        best_response_value = max(vals)
        if key not in br_strat_map:
            br_strat_map[key] = np.array([0.0, 0.0])
        br_strat_map[key] = br_strat_map[key] + prob * np.array(vals, dtype=np.float64)
        return -best_response_value
    else:
        strategy = agg_hagent[key]
        action_values = [calc_best_response_kp(agg_hagent, br_strat_map, br_player, cards,
                                            history + action, next_player, prob * strategy[idx])
                         for idx, action in enumerate(actions_kp)]
        return -np.dot(strategy, action_values)

def get_exploitability_kp(agg_hagent):
    exploitability = 0

    br_hagent = {}
    for cards in itertools.permutations(kuhn_cards):
        calc_best_response_kp(agg_hagent, br_hagent, 0, cards, '', 0, 1.0)
        calc_best_response_kp(agg_hagent, br_hagent, 1, cards, '', 0, 1.0)

    for k,v in br_hagent.items():
        v[:] = np.where(v == np.max(v), 1, 0)

    for cards in itertools.permutations(kuhn_cards):
        ev_1 = calc_ev_kp(agg_hagent, br_hagent, cards, '', 0)
        ev_2 = calc_ev_kp(br_hagent, agg_hagent, cards, '', 0)
        exploitability += 1 / 6 * (ev_1 - ev_2)
    return exploitability, br_hagent

class KuhnPop:
    def __init__(self, env, nb_games=100, seed=0):
        # Environment
        self.env = env
        self.nb_games = nb_games

        # Population
        self.pop_size = 2
        self.pop = [Agent(env, seed=i+seed, ac_kwargs=ac_kwargs) for i in range(self.pop_size)]

        # Hashed population
        self.hpop = [self.hash_agent(agent) for agent in self.pop]  # This are just dict of state to prob

        # Metagame
        self.metagame = np.zeros((self.pop_size, self.pop_size))
        for i, hagent1 in enumerate(self.hpop):
            for j, hagent2 in enumerate(self.hpop):
                    self.metagame[i, j] = self.get_payoff(hagent1, hagent2)

    def add_new(self):
        self.pop.append(Agent(self.env, seed=self.pop_size, ac_kwargs=ac_kwargs))
        self.hpop.append(self.hash_agent(self.pop[-1]))
        self.pop_size += 1
        metagame = np.zeros((self.pop_size, self.pop_size))
        metagame[:-1, :-1] = self.metagame
        self.metagame = metagame

    def get_metagame(self, k=None):
        if k is None:
            for i, hagent1 in enumerate(self.hpop):
                for j, hagent2 in enumerate(self.hpop):
                    self.metagame[i, j] = self.get_payoff(hagent1, hagent2)
            return self.metagame

        metagame = np.zeros((k, k))
        for i, hagent1 in enumerate(self.hpop):
            if i >= k:
                continue
            for j, hagent2 in enumerate(self.hpop):
                if j >= k:
                    continue
                metagame[i, j] = self.get_payoff(hagent1, hagent2)
        return metagame

    def get_payoff(self, hagent1, hagent2):
        payoff = 0.
        for cards in itertools.permutations(kuhn_cards):
            payoff += -calc_ev_kp(hagent1, hagent2, cards, '', 0)  # This function returns payoff for second agent
            payoff += calc_ev_kp(hagent2, hagent1, cards, '', 0)  # This function returns payoff for second agent
        return payoff / 12.  # 6 permutations x 2 players

    def get_payoff2(self, agent1, agent2):
        payoff = 0.
        nb_games = 0
        obs = self.env.reset()
        player_turn = 0
        swap = False
        while 1:
            if player_turn==0:
                if swap:
                    if isinstance(agent2, dict):
                        a = np.random.choice(2, 1, p=agent2[str(obs[0])])
                    else:
                        a, _, _ = agent2.ac.step(torch.as_tensor(obs[0], dtype=torch.float32))
                else:
                    if isinstance(agent1, dict):
                        a = np.random.choice(2, 1, p=agent1[str(obs[0])])
                    else:
                        a, _, _ = agent1.ac.step(torch.as_tensor(obs[0], dtype=torch.float32))
            else:
                if swap:
                    if isinstance(agent1, dict):
                        a = np.random.choice(2, 1, p=agent1[str(obs[1])])
                    else:
                        a, _, _ = agent1.ac.step(torch.as_tensor(obs[1], dtype=torch.float32))
                else:
                    if isinstance(agent2, dict):
                        a = np.random.choice(2, 1, p=agent2[str(obs[1])])
                    else:
                        a, _, _ = agent2.ac.step(torch.as_tensor(obs[1], dtype=torch.float32))

            next_o, r, d, _ = self.env.step(a)
            obs = next_o
            player_turn = 1 - player_turn

            if d:
                nb_games += 1
                if swap:
                    payoff += r[1]
                else:
                    payoff += r[0]
                obs = self.env.reset()
                player_turn = 0
                swap = not swap  # We swap agents playing order
                if nb_games == self.nb_games:
                    break
        return payoff / self.nb_games

    def hash_agent(self, agent):
        hagent = {}
        for infostate in infostates:
            obs = infostate2vector(infostate)
            with torch.no_grad():
                prob = agent.get_prob(torch.as_tensor(obs, dtype=torch.float32))
            hagent[infostate] = prob / np.sum(prob)
        return hagent

    def update_hashed_agent(self, k):
        hagent = {}
        for infostate in infostates:
            obs = infostate2vector(infostate)
            with torch.no_grad():
                prob = self.pop[k].get_prob(torch.as_tensor(obs, dtype=torch.float32))
            hagent[infostate] = prob / np.sum(prob)
        self.hpop[k] = hagent

    def agg_agents(self, weights):
        # Here weights should be the metanash of the first len(weights) agents
        agg_hashed_agent = {}
        for infostate in infostates:
            for i, w in enumerate(weights):
                prob = self.hpop[i][infostate]
                if infostate not in agg_hashed_agent:
                    agg_hashed_agent[infostate] = w*prob
                else:
                    agg_hashed_agent[infostate] += w*prob

            agg_hashed_agent[infostate] /= np.sum(agg_hashed_agent[infostate])

        return agg_hashed_agent

    def popn_update(self, env, meta_nash, ppo_kwargs):
        train_agent = self.pop[-1]
        agg_enemy = self.agg_agents(meta_nash)
        ppo_trainer = PPOTrainer(env, train_agent, agg_enemy, self, **ppo_kwargs)
        _ = ppo_trainer.train(switch=False)
        self.update_hashed_agent(-1)

    def get_exploitability(self, meta_nash):
        agg_hagent = self.agg_agents(meta_nash)
        exploitability = 0

        br_hagent = {}
        for cards in itertools.permutations(kuhn_cards):
            calc_best_response_kp(agg_hagent, br_hagent, 0, cards, '', 0, 1.0)
            calc_best_response_kp(agg_hagent, br_hagent, 1, cards, '', 0, 1.0)

        for k,v in br_hagent.items():
            v[:] = np.where(v == np.max(v), 1, 0)

        for cards in itertools.permutations(kuhn_cards):
            ev_1 = calc_ev_kp(agg_hagent, br_hagent, cards, '', 0)
            ev_2 = calc_ev_kp(br_hagent, agg_hagent, cards, '', 0)
            exploitability += 1 / 6 * (ev_1 - ev_2)
        return exploitability, br_hagent

def best_response_oracle_kp(agg_agent, br_type='exact_br'):

  br_hagent = {}
  br_agent_probs = np.zeros((12,2))
  for cards in itertools.permutations(kuhn_cards):
    calc_best_response_kp(agg_agent, br_hagent, 0, cards, '', 0, 1.0)
    calc_best_response_kp(agg_agent, br_hagent, 1, cards, '', 0, 1.0)

  for k,v in br_hagent.items():
    if br_type == 'exact_br':
        v[:] = np.where(v == np.max(v), 1, 0)
    elif br_type == 'approx_br_clean':
        v[:] = np.where(v == np.max(v), 0.75, 0.25)
    elif br_type == 'approx_br_rand':
        probs = np.random.uniform(size=(1,2))
        probs[:] = np.where(v == np.max(v), probs + 1, probs + 0)
        probs = probs / np.sum(probs, axis=1)[:, None]
        v[:] = probs[0]

  for i, infostate in enumerate(infostates):
    br_agent_probs[i] = br_hagent[infostate]
  
    
  return br_hagent, br_agent_probs


