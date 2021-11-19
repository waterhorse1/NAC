import numpy as np
import scipy.signal
import gym
import time
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import torch.nn.functional as F
import wandb

infostates_kp = ['J', 'JB', 'JP', 'JPB', 'Q', 'QB',  # turn for player 1
              'QP', 'QPB', 'K', 'KB', 'KP', 'KPB']  # turn for player 2
actions_kp = ['P', 'B']
kuhn_cards = ['J', 'Q', 'K']

def infostate2vector(infostate):
    # infosate = card + history just the sum of two strings
    assert infostate in infostates_kp, infostate + " not in infostates"
    idx = infostates_kp.index(infostate)
    vec = np.zeros((len(infostates_kp),), dtype=np.float32)
    vec[idx] = 1.
    return vec
    
class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros(combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size  # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.
        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size  # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        data = dict(obs=self.obs_buf, act=self.act_buf, ret=self.ret_buf,
                    adv=self.adv_buf, logp=self.logp_buf)
        return {k: torch.as_tensor(v, dtype=torch.float32) for k, v in data.items()}


def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def mlp(sizes, activation, output_activation=nn.Identity):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MLPCategoricalPolicy(nn.Module):
    """
    MLPCategoricalActor implements a cotegorical policy where
    logits are computed using a multilayer perceptron
    Args:
        obs_dim (int): Dimension of the observations
        act_dim (int): Dimension of the actions
        hidden_sizes (int): List of int with number of units per hidden layer in mlp
        activation (nn.Module): A pytorch module with a valid activation  function (e.g. nn.Tanh)
    Attributes:
        logits_net (nn.Sequential): The logits neural network
    """

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation):
        super().__init__()
        self.logits_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs):
        """Returns the categorical distribution for the given observation
                Args:
                    obs (torch.Tensor): A pytorch tensor with the observation
                Returns:
                    pi (torch.distributions.categorical): The Categorical Distribution object
        """
        logits = self.logits_net(obs)
        return Categorical(logits=logits)

    def _log_prob_from_distribution(self, pi, act):
        """Returns the log-probabilities for a given policy and action
                Args:
                    pi (torch.distributions.categorical): A pytorch Categorical Distribution object
                    act (torch.Tensor): A pytorch tensor with the action
                Returns:
                    log_prob (torch.Tensor): The log probability for given pi and act
        """
        return pi.log_prob(act)

    def forward(self, obs, act=None):
        # Produce action distributions for given observations, and
        # optionally compute the log likelihood of given actions under
        # those distributions.
        pi = self._distribution(obs)
        logp_a = None
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
        return pi, logp_a


class MLPCritic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_net = mlp([obs_dim] + list(hidden_sizes) + [1], activation)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1)  # Critical to ensure v has right shape.


class MLPActorCritic(nn.Module):

    def __init__(self, observation_space, action_space,
                 hidden_sizes=(64, 64), activation=nn.Tanh):
        super().__init__()

        obs_dim = observation_space.shape[0]

        # policy builder
        self.pi = MLPCategoricalPolicy(obs_dim, action_space.n, hidden_sizes, activation)

        # build value function
        self.v = MLPCritic(obs_dim, hidden_sizes, activation)

    def step(self, obs):
        with torch.no_grad():
            pi = self.pi._distribution(obs)
            a = pi.sample()
            logp_a = self.pi._log_prob_from_distribution(pi, a)
            v = self.v(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        return self.step(obs)[0]

    def save_model(self, PATH=None):
        import os
        if PATH is not None:
            torch.save(self.pi.state_dict(), os.path.join(PATH, "_actor"))
            torch.save(self.v.state_dict(), os.path.join(PATH, "_critic"))
        else:
            torch.save(self.pi.state_dict(), "_actor")
            torch.save(self.v.state_dict(), "_critic")

    def load_model(self, PATH=None, from_checkpoint=False, optimiser_pi='Adam'):
        import os
        if not from_checkpoint:
            if PATH is not None:
                self.pi.load_state_dict(torch.load(os.path.join(PATH, "_actor")))
                self.v.load_state_dict(torch.load(os.path.join(PATH, "_critic")))
            else:
                self.pi.load_state_dict(torch.load("_actor"))
                self.v.load_state_dict(torch.load("_critic"))
        else:
            if PATH is not None:
                checkpoint = torch.load(os.path.join(PATH, "_actor"))
            else:
                checkpoint = torch.load("checkpoint_" + optimiser_pi)
            self.pi.load_state_dict(checkpoint['pi'])
            self.v.load_state_dict(checkpoint['v'])

class Agent:
    def __init__(self, env,
                 seed=0, ac_kwargs=dict(),  # dict(hidden_sizes=[args.hid] * args.l)
                 gamma=0.99, lam=0.97, steps_per_epoch=400):
        # Random seed
        #torch.manual_seed(seed)
        #np.random.seed(seed)

        # Create actor-critic module
        self.ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
        self.buf = PPOBuffer(env.observation_space.shape,
                             env.action_space.shape, steps_per_epoch, gamma, lam)
        self.steps_per_epoch = steps_per_epoch

    def get_prob(self, obs):
        # Returns probabilities for all actions (legal and illegal)
        probs = F.softmax(self.ac.pi.logits_net(obs), dim=-1)
        return probs.cpu().clone().numpy()

    def get_prob_allinfostates(self):
        probs = []
        for infostate in infostates:
            obs_vec = infostate2vector(infostate)
            probs.append(F.softmax(self.ac.pi.logits_net(torch.as_tensor(obs_vec, dtype=torch.float32)), dim=-1))
        return torch.cat(probs, 0)


class PPOTrainer:
    # Trains agent1 against agent2
    def __init__(self, env, agent1, agent2, kuhn_pop, seed=0,
                 pi_lr=3e-4, vf_lr=1e-3, target_kl=0.01, clip_ratio=0.2,
                 train_pi_iters=80, train_v_iters=80
                 ):

        # Random seed
        #torch.manual_seed(seed)
        #np.random.seed(seed)

        # Env
        self.env = env

        # Agents
        self.agent1 = agent1
        self.agent2 = agent2
        self.steps_per_epoch = self.agent1.steps_per_epoch
        self.kuhn_pop = kuhn_pop  # Whole population

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.agent1.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.agent1.ac.v.parameters(), lr=vf_lr)
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio

    def run_epoch(self):

        obs = self.env.reset()
        obs_vec, infostate = obs[0], obs[1]
        player_turn = 0
        swap = False
        t = 0
        r = 0.
        exp_return = 0.
        prev_timestep = None
        while 1:
            if player_turn==0:
                if swap:
                    a = np.random.choice(self.agent2[infostate].size, 1, p=self.agent2[infostate])[0]
                else:
                    a, v, logp = self.agent1.ac.step(torch.as_tensor(obs_vec, dtype=torch.float32))
            else:
                if swap:
                    a, v, logp = self.agent1.ac.step(torch.as_tensor(obs_vec, dtype=torch.float32))
                else:
                    a = np.random.choice(self.agent2[infostate].size, 1, p=self.agent2[infostate])[0]


            # update buffer
            if ((player_turn==0 and not swap) or (player_turn==1 and swap)):  # If agent1 took action
                reward = -r if swap else r  # The reward derived from that action
                exp_return += reward
                if prev_timestep:
                    self.agent1.buf.store(prev_timestep[0], # obs_vec
                                          prev_timestep[1], # a
                                          reward, # reward
                                          prev_timestep[2], # value
                                          prev_timestep[3])  # log_prob
                    t += 1
                prev_timestep = [obs_vec, a, v, logp]

            # Update environment
            next_obs, r, d, _ = self.env.step(a)
            next_obs_vec, next_infostate = next_obs[0], next_obs[1]


            # Update obs (critical!)
            obs_vec, infostate = next_obs_vec, next_infostate
            player_turn = 1 - player_turn


            epoch_ended = (t == self.steps_per_epoch )
            if d and not epoch_ended:
                # Store end of episode transition
                reward = -r if swap else r
                self.agent1.buf.store(prev_timestep[0], # obs_vec
                                      prev_timestep[1], # a
                                      reward, # reward
                                      prev_timestep[2], # value
                                      prev_timestep[3])  # log_prob
                t += 1
                r = 0.
                exp_return += reward

                # Finish Path
                v = 0.
                self.agent1.buf.finish_path(v)

                # Reset environment
                obs = self.env.reset()
                obs_vec, infostate = obs[0], obs[1]
                player_turn = 0
                prev_timestep = None

                # Swap players
                swap = not swap

            epoch_ended = (t == self.steps_per_epoch )
            if epoch_ended:
                if not d:
                    v = 0.
                    self.agent1.buf.finish_path(v)
                break

        return exp_return / t

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.agent1.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info

    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.agent1.ac.v(obs) - ret) ** 2).mean()

    def update(self, switch=False):
        data = self.agent1.buf.get()
        log_dict = {'KL': [], 'Loss Pi': [], 'Loss V': [], 'Entropy Pi': [], 'Clip Frac' : []}

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
            self.vf_optimizer.zero_grad()
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data)
            loss_v = self.compute_loss_v(data)
            log_dict['Loss V'].append(loss_v)
            kl = pi_info['kl']
            log_dict['KL'].append(kl)
            if kl > 1.5 * self.target_kl:
                break
            loss = loss_pi
            log_dict['Loss Pi'].append(loss)
            loss.backward()
            loss_v.backward()
            # mpi_avg_grads(ac.pi)  # average grads across MPI processes
            self.pi_optimizer.step()
            self.vf_optimizer.step()
            log_dict['Entropy Pi'].append(pi_info['ent'])
            log_dict['Clip Frac'].append(pi_info['cf'])
        
        return log_dict

    def train(self,  switch=False):
        exp_return = self.run_epoch()
        log_dict = self.update(switch=switch)
        return log_dict


class PPO:
    def __init__(self, env,
                 seed=0, ac_kwargs=dict(),
                 gamma=0.99,  pi_lr=3e-4, vf_lr=1e-3, target_kl=0.01, clip_ratio=0.2,
                 steps_per_epoch=400, train_pi_iters=80, train_v_iters=80):
        # Random seed
        #torch.manual_seed(seed)
        #np.random.seed(seed)

        # Dimensions
        if isinstance(env.observation_space, Box):
            obs_dim = env.observation_space.shape
        elif isinstance(env.observation_space, Discrete):
            obs_dim = env.observation_space.n
        act_dim = env.action_space.shape

        # Create actor-critic module
        self.ac = MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)
        # self.ac = core.MLPActorCritic(env.observation_space, env.action_space, **ac_kwargs)

        # Set up experience buffer
        self.steps_per_epoch = steps_per_epoch
        self.buf = PPOBuffer(obs_dim, act_dim, self.steps_per_epoch, gamma, lam)

        # Set up optimizers for policy and value function
        self.pi_optimizer = Adam(self.ac.pi.parameters(), lr=pi_lr)
        self.vf_optimizer = Adam(self.ac.v.parameters(), lr=vf_lr)
        self.train_v_iters = train_v_iters
        self.train_pi_iters = train_pi_iters
        self.target_kl = target_kl
        self.clip_ratio = clip_ratio

    def compute_loss_pi(self, data):
        obs, act, adv, logp_old = data['obs'], data['act'], data['adv'], data['logp']

        # Policy loss
        pi, logp = self.ac.pi(obs, act)
        ratio = torch.exp(logp - logp_old)
        clip_adv = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * adv
        loss_pi = -(torch.min(ratio * adv, clip_adv)).mean()

        # Useful extra info
        approx_kl = (logp_old - logp).mean().item()
        ent = pi.entropy().mean().item()
        clipped = ratio.gt(1 + self.clip_ratio) | ratio.lt(1 - self.clip_ratio)
        clipfrac = torch.as_tensor(clipped, dtype=torch.float32).mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, cf=clipfrac)

        return loss_pi, pi_info


    def compute_loss_v(self, data):
        obs, ret = data['obs'], data['ret']
        return ((self.ac.v(obs) - ret) ** 2).mean()

    def update(self):
        data = self.buf.get()

        # Train policy with multiple steps of gradient descent
        for i in range(self.train_pi_iters):
                self.pi_optimizer.zero_grad()
                loss_pi, pi_info = self.compute_loss_pi(data)
                kl = pi_info['kl']
                if kl > 1.5 * self.target_kl:
                    break
                loss_pi.backward()
                # mpi_avg_grads(ac.pi)  # average grads across MPI processes
                self.pi_optimizer.step()

        # Value function learning
        for i in range(self.train_v_iters):
            self.vf_optimizer.zero_grad()
            loss_v = self.compute_loss_v(data)
            loss_v.backward()
            self.vf_optimizer.step()


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Acrobot-v1')
    # parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--hid', type=int, default=128)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--exp_name', type=str, default='vpg')
    args = parser.parse_args()

    max_ep_len = 1000


    env = gym.make(args.env)
    sig = PPO(env, seed=0, ac_kwargs=dict(hidden_sizes=[args.hid] * args.l, activation=torch.nn.ReLU),
                gamma=0.99, lam=0.97, pi_lr=3e-4, vf_lr=1e-3,
                steps_per_epoch=1000, train_v_iters=80)

    o, ep_ret, ep_len = env.reset(), 0, 0

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(args.epochs):
        eps_ret = []
        eps_len = []
        for t in range(sig.steps_per_epoch):
            a, v, logp = sig.ac.step(torch.as_tensor(o, dtype=torch.float32))

            next_o, r, d, _ = env.step(a)
            ep_ret += r
            ep_len += 1

            # save and log
            sig.buf.store(o, a, r, v, logp)

            # Update obs (critical!)
            o = next_o

            timeout = ep_len == max_ep_len
            terminal = d or timeout
            epoch_ended = t == sig.steps_per_epoch - 1

            if terminal or epoch_ended:
                if epoch_ended and not (terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.' % ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _ = sig.ac.step(torch.as_tensor(o, dtype=torch.float32))
                else:
                    v = 0
                sig.buf.finish_path(v)
                if terminal:
                    # only save EpRet / EpLen if trajectory finished
                    eps_ret.append(ep_ret)
                    eps_len.append(ep_len)
                o, ep_ret, ep_len = env.reset(), 0, 0

        print("Epoch {:d} with mean value ended with length&reward {:.2f}, {:.2f}"
              .format(epoch, np.mean(np.array(eps_len)), np.mean(np.array(eps_ret))))

        # Perform VPG update!
        sig.update()