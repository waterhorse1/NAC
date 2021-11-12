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

device_train = 'cuda:0'
device_test = 'cpu'
lam = 0.1

class MyGaussianPDF(nn.Module):
    def __init__(self, mu):
        super(MyGaussianPDF, self).__init__()
        self.mu = mu.to(device_train)
        self.cov = 0.54*torch.eye(2).to(device_train)
        self.c = 1.

    def forward(self, x):
        return self.c*torch.exp(-0.5*torch.diagonal( (x-self.mu)@self.cov@(x-self.mu).t() ))

class MyGaussianPDF_test(nn.Module):
    def __init__(self, mu):
        super(MyGaussianPDF_test, self).__init__()
        self.mu = mu.to(device_test)
        self.cov = 0.54*torch.eye(2).to(device_test)
        self.c = 1.

    def forward(self, x):
        return self.c*torch.exp(-0.5*torch.diagonal( (x-self.mu)@self.cov@(x-self.mu).t() ))

class GMMAgent(nn.Module):
    def __init__(self, mu, game):
        super(GMMAgent, self).__init__()
        self.mu = mu
        self.game = game
        self.gauss = MyGaussianPDF(mu).to(device_train)
        self.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_train)


    def forward(self):
        return self.gauss(self.x)

class GMMAgent_nash(nn.Module):
    def __init__(self, mu):
        super(GMMAgent_nash, self).__init__()
        self.gauss = MyGaussianPDF_test(mu).to(device_test)
        self.x = nn.Parameter(0.01*torch.randn(2, dtype=torch.float), requires_grad=False).to(device_test)

    def forward(self):
        return self.gauss(self.x)

class GMMAgent_nash_br(nn.Module):
    def __init__(self, mu):
        super(GMMAgent_nash_br, self).__init__()
        self.gauss = MyGaussianPDF_test(mu).to(device_test)
        self.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_train)

    def forward(self):
        return self.gauss(self.x)

class TorchPop:

    def __init__(self, num_mode=3, seed=0, resample_mu=False, test=False):
        if test:
          torch.manual_seed(seed)
          np.random.seed(seed)
        self.pop_size = 2

        assert num_mode < 8
        self.num_mode = num_mode
        gap = 1/(6*num_mode) * np.pi
        each_phi = 11/(6 * num_mode) * np.pi
        phi_list = []
        for i in range(self.num_mode):
            if i == 0:
                if resample_mu:
                  start = np.random.randint(100)
                else:
                  start = 0
                end = start + each_phi
            else:
                start = end + gap
                end = start + each_phi
            phi_list.append((start + end) / 2)
    
        phi = np.array(phi_list)
        mus = 2.8722 * np.stack([np.sin(phi), np.cos(phi)], axis=1)
        mus = torch.from_numpy(mus).float().to(device_train)
        
        self.mus = mus
        self.gauss = MyGaussianPDF(self.mus).to(device_train)
        
        if num_mode == 3:
          self.game = circulant([0, -1, 1])
        elif num_mode == 5:
          self.game = circulant([0, -1, -1, 1, 1])
        elif num_mode == 7:
          self.game = circulant([0, -1, -1, -1, 1, 1, 1])

        self.game = torch.from_numpy(self.game).float().to(device_train)

        self.pop = [GMMAgent(mus, self.game) for _ in range(self.pop_size)]
        self.pop_hist = [[self.pop[i].x.detach().cpu().clone().numpy()] for i in range(self.pop_size)]

        #print(self.pop[0].x)

    def visualise_pop(self, br=None, ax=None, color=None):

        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2 * np.pi) ** n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
            return np.exp(-fac / 2) / N

        agents = [agent.x.detach().cpu().numpy() for agent in self.pop]
        agents = list(zip(*agents))

        # Colors
        if color is None:
            colors = cm.rainbow(np.linspace(0, 1, len(agents[0])))
        else:
            colors = [color]*len(agents[0])

        # fig = plt.figure(figsize=(6, 6))
        ax.scatter(agents[0], agents[1], alpha=1., marker='.', color=colors, s=8*plt.rcParams['lines.markersize'] ** 2)
        if br is not None:
            ax.scatter(br[0], br[1], marker='.', c='k')
        for i, hist in enumerate(self.pop_hist):
            if hist:
                hist = list(zip(*hist))
                ax.plot(hist[0], hist[1], alpha=0.8, color=colors[i], linewidth=4)

        # ax = plt.gca()
        for i in range(7):
            ax.scatter(self.mus[i, 0].item(), self.mus[i, 1].item(), marker='x', c='k')
            for j in range(4):
                delta = 0.025
                x = np.arange(-4.5, 4.5, delta)
                y = np.arange(-4.5, 4.5, delta)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, self.mus[i,:].numpy(), 0.54 * np.eye(2))
                levels = 10
                # levels = np.logspace(0.01, 1, 10, endpoint=True)
                CS = ax.contour(X, Y, Z, levels, colors='k', linewidths=0.5, alpha=0.2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                # ax.clabel(CS, fontsize=9, inline=1)
                # circle = plt.Circle((0, 0), 0.2, color='r')
                # ax.add_artist(circle)
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])


    def get_payoff(self, agent1, agent2, ag_type='logits'):
        if ag_type == 'Agent':
            p = agent1()
            q = agent2()
        
        else:
            p = self.gauss(agent1)
            q = self.gauss(agent2)
        
        return p @ self.game @ q + 0.5*(p-q).sum()

    def agg_agents(self, metanash):
        agg_agent = metanash[0] * self.pop[0]()
        print(self.pop[0]())
        for k in range(1, self.pop_size):
            agg_agent += metanash[k]*self.pop[k]()
        return agg_agent

    def agg_agents_agent(self, metanash):
        agg_agent = GMMAgent(self.mus, self.game)
        agg_agent.x = metanash[0] * self.pop[0].x
        for k in range(1, self.pop_size):
            agg_agent.x += metanash[k]*self.pop[k].x
        return agg_agent

    def get_payoff_aggregate(self, agent1, metanash, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = metanash[0] * self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_br_to_strat(self, metanash, lr, nb_iters=20):
        br = GMMAgent(self.mus)
        br.x = nn.Parameter(0.01*torch.randn(2, dtype=torch.float), requires_grad=False)
        br.x.requires_grad = True
        optimiser = optim.Adam(br.parameters(), lr=1.0)
        for _ in range(100):
            loss = -self.get_payoff_aggregate(br, metanash, self.pop_size,)
            # Optimise !
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return br

    def get_metagame(self, k=None, numpy=False):
        if k==None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j], ag_type='Agent')
                return metagame.detach().cpu().clone().numpy()
        else:
            metagame = torch.zeros(k, k)
            for i in range(k):
                for j in range(k):
                    metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j], ag_type='Agent')
            return metagame

    def add_new(self):
        with torch.no_grad():
            self.pop.append(GMMAgent(self.mus, self.game))
            self.pop_hist.append([self.pop[-1].x.detach().cpu().clone().numpy()])
            self.pop_size += 1

    def get_exploitability(self, metanash, lr, nb_iters=20):
        br = self.get_br_to_strat(metanash, lr, nb_iters=nb_iters)
        with torch.no_grad():
            exp = self.get_payoff_aggregate(br, metanash, self.pop_size).item()
        return exp

    def get_exploitability_train(self, meta_nash, training_iters, lr=1.0):
        br = GMMAgent(self.mus, self.game)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_train)
        br.x.requires_grad = True
        for train_iter in range(training_iters):
          exp_payoff = self.get_payoff_aggregate(br, meta_nash, self.pop_size,)
          loss = -exp_payoff
          br_grad = torch.autograd.grad(loss, br.x, create_graph=True)
          br.x = br.x - lr * br_grad[0]
        
        final_exploitability = self.get_payoff_aggregate(br, meta_nash, self.pop_size)
        return 2 * final_exploitability

    def get_exploitability_test(self, meta_nash, training_iters=100, lr=1.0):
        br = GMMAgent(self.mus, self.game)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_train)
        br.x.requires_grad = True
        lr = 1.0
        for train_iter in range(100):
          exp_payoff = self.get_payoff_aggregate(br, meta_nash, self.pop_size,)
          loss = -exp_payoff
          br_grad = torch.autograd.grad(loss, br.x, create_graph=True)
          br.x = br.x - lr * br_grad[0]
        
        with torch.no_grad():
          final_exploitability = self.get_payoff_aggregate(br, meta_nash, self.pop_size)
        return 2 * final_exploitability


class TorchPop_nash:

    def __init__(self, num_mode=3, seed=0, resample_mu=False, test=False):
        if test:
          torch.manual_seed(seed)
          np.random.seed(seed)
        self.pop_size = 2

        #mus = np.array([[2.8722, -0.025255],
        #                [-1.4580, -2.4747],
        #                [-1.4142, 2.5]])
        assert num_mode < 8
        self.num_mode = num_mode
        gap = 1/(6*num_mode) * np.pi
        each_phi = 11/(6 * num_mode) * np.pi
        phi_list = []
        for i in range(self.num_mode):
            if i == 0:
                if resample_mu:
                  start = np.random.randint(100)
                else:
                  start = 0
                end = start + each_phi
            else:
                start = end + gap
                end = start + each_phi
            #phi_list.append(np.random.uniform(start, end))
            phi_list.append((start + end) / 2)
    
        phi = np.array(phi_list)
        mus = 2.8722 * np.stack([np.sin(phi), np.cos(phi)], axis=1)
        mus = torch.from_numpy(mus).float().to(device_test)
        
        self.mus = mus
        
        if num_mode == 3:
          self.game = circulant([0, -1, 1])
        elif num_mode == 5:
          self.game = circulant([0, -1, -1, 1, 1])
        elif num_mode == 7:
          self.game = circulant([0, -1, -1, -1, 1, 1, 1])

        #self.game = -np.ones([num_mode, num_mode])
        #for q in range(num_mode):
        #    self.game[q, q] = 0.
        #    self.game[q, (q+1)%num_mode] = 1.

        self.game = torch.from_numpy(self.game).float().to(device_test)

        self.pop = [GMMAgent_nash(mus) for _ in range(self.pop_size)]
        self.pop_hist = [[self.pop[i].x.detach().cpu().clone().numpy()] for i in range(self.pop_size)]

        #print(self.pop[0].x)

    def visualise_pop(self, br=None, ax=None, color=None):

        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2 * np.pi) ** n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
            return np.exp(-fac / 2) / N

        agents = [agent.x.detach().cpu().numpy() for agent in self.pop]
        agents = list(zip(*agents))

        # Colors
        if color is None:
            colors = cm.rainbow(np.linspace(0, 1, len(agents[0])))
        else:
            colors = [color]*len(agents[0])

        # fig = plt.figure(figsize=(6, 6))
        ax.scatter(agents[0], agents[1], alpha=1., marker='.', color=colors, s=8*plt.rcParams['lines.markersize'] ** 2)
        if br is not None:
            ax.scatter(br[0], br[1], marker='.', c='k')
        for i, hist in enumerate(self.pop_hist):
            if hist:
                hist = list(zip(*hist))
                ax.plot(hist[0], hist[1], alpha=0.8, color=colors[i], linewidth=4)

        # ax = plt.gca()
        for i in range(7):
            ax.scatter(self.mus[i, 0].item(), self.mus[i, 1].item(), marker='x', c='k')
            for j in range(4):
                delta = 0.025
                x = np.arange(-4.5, 4.5, delta)
                y = np.arange(-4.5, 4.5, delta)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, self.mus[i,:].numpy(), 0.54 * np.eye(2))
                levels = 10
                # levels = np.logspace(0.01, 1, 10, endpoint=True)
                CS = ax.contour(X, Y, Z, levels, colors='k', linewidths=0.5, alpha=0.2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                # ax.clabel(CS, fontsize=9, inline=1)
                # circle = plt.Circle((0, 0), 0.2, color='r')
                # ax.add_artist(circle)
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])


    def get_payoff(self, agent1, agent2):
        p = agent1()
        q = agent2()
        return p @ self.game @ q + 0.5*(p-q).sum()

    def get_payoff_aggregate(self, agent1, metanash, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = metanash[0]*self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_br_to_strat(self, metanash, lr, nb_iters=20):
        br = GMMAgent_nash_br(self.mus)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_test)
        br.x.requires_grad = True
        for _ in range(100):
          exp_payoff = self.get_payoff_aggregate(br, metanash, self.pop_size,)
          loss = -exp_payoff
          br_grad = torch.autograd.grad(loss, br.x, create_graph=True)
          br.x = br.x - lr * br_grad[0]

        return br

    def get_metagame(self, k=None, numpy=False):
        if k==None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
                return metagame.detach().cpu().clone().numpy()
        else:
            metagame = torch.zeros(k, k)
            for i in range(k):
                for j in range(k):
                    metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j])
            return metagame

    def add_new(self):
        with torch.no_grad():
            self.pop.append(GMMAgent_nash(self.mus))
            self.pop_hist.append([self.pop[-1].x.detach().cpu().clone().numpy()])
            self.pop_size += 1

    def get_exploitability(self, metanash, lr, nb_iters=20):
        br = self.get_br_to_strat(metanash, lr, nb_iters=nb_iters)
        with torch.no_grad():
            exp = self.get_payoff_aggregate(br, metanash, self.pop_size).item()
        return 2 * exp

    def get_exploitability_train(self, meta_nash, training_iters, lr=1.0):
        br = GMMAgent_nash_br(self.mus)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_test)
        br.x.requires_grad = True
        for train_iter in range(training_iters):
          exp_payoff = self.get_payoff_aggregate(br, meta_nash, self.pop_size,)
          loss = -exp_payoff
          br_grad = torch.autograd.grad(loss, br.x, create_graph=True)
          br.x = br.x - lr * br_grad[0]
        
        final_exploitability = self.get_payoff_aggregate(br, meta_nash, self.pop_size)
        return 2 * final_exploitability

    def get_exploitability_test(self, meta_nash, training_iters, lr=1.0):
        br = GMMAgent_nash_br(self.mus)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_test)
        br.x.requires_grad = True
        for train_iter in range(50):
          exp_payoff = self.get_payoff_aggregate(br, meta_nash, self.pop_size,)
          loss = -exp_payoff
          br_grad = torch.autograd.grad(loss, br.x, create_graph=True)
          br.x = br.x - lr * br_grad[0]
        
        with torch.no_grad():
          final_exploitability = self.get_payoff_aggregate(br, meta_nash, self.pop_size)
        return 2 * final_exploitability

class TorchPop_nn:

    def __init__(self, num_mode=3, seed=0, resample_mu=False, test=False):
        if test:
          torch.manual_seed(seed)
          np.random.seed(seed)
        self.pop_size = 2

        assert num_mode < 8
        self.num_mode = num_mode
        gap = 1/(6*num_mode) * np.pi
        each_phi = 11/(6 * num_mode) * np.pi
        phi_list = []
        for i in range(self.num_mode):
            if i == 0:
                if resample_mu:
                  start = np.random.randint(100)
                else:
                  start = 0
                end = start + each_phi
            else:
                start = end + gap
                end = start + each_phi
            phi_list.append((start + end) / 2)
    
        phi = np.array(phi_list)
        mus = 2.8722 * np.stack([np.sin(phi), np.cos(phi)], axis=1)
        mus = torch.from_numpy(mus).float().to(device_train)
        
        self.mus = mus
        self.gauss = MyGaussianPDF(self.mus).to(device_train)
        
        if num_mode == 3:
          self.game = circulant([0, -1, 1])
        elif num_mode == 5:
          self.game = circulant([0, -1, -1, 1, 1])
        elif num_mode == 7:
          self.game = circulant([0, -1, -1, -1, 1, 1, 1])

        self.game = torch.from_numpy(self.game).float().to(device_train)

        self.pop = [GMMAgent_nn(mus, self.game) for _ in range(self.pop_size)]
        self.pop_hist = [[self.pop[i].x.detach().cpu().clone().numpy()] for i in range(self.pop_size)]

        #print(self.pop[0].x)

    def visualise_pop(self, br=None, ax=None, color=None):

        def multivariate_gaussian(pos, mu, Sigma):
            """Return the multivariate Gaussian distribution on array pos."""

            n = mu.shape[0]
            Sigma_det = np.linalg.det(Sigma)
            Sigma_inv = np.linalg.inv(Sigma)
            N = np.sqrt((2 * np.pi) ** n * Sigma_det)
            # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
            # way across all the input variables.
            fac = np.einsum('...k,kl,...l->...', pos - mu, Sigma_inv, pos - mu)
            return np.exp(-fac / 2) / N

        agents = [agent.x.detach().cpu().numpy() for agent in self.pop]
        agents = list(zip(*agents))

        # Colors
        if color is None:
            colors = cm.rainbow(np.linspace(0, 1, len(agents[0])))
        else:
            colors = [color]*len(agents[0])

        # fig = plt.figure(figsize=(6, 6))
        ax.scatter(agents[0], agents[1], alpha=1., marker='.', color=colors, s=8*plt.rcParams['lines.markersize'] ** 2)
        if br is not None:
            ax.scatter(br[0], br[1], marker='.', c='k')
        for i, hist in enumerate(self.pop_hist):
            if hist:
                hist = list(zip(*hist))
                ax.plot(hist[0], hist[1], alpha=0.8, color=colors[i], linewidth=4)

        # ax = plt.gca()
        for i in range(7):
            ax.scatter(self.mus[i, 0].item(), self.mus[i, 1].item(), marker='x', c='k')
            for j in range(4):
                delta = 0.025
                x = np.arange(-4.5, 4.5, delta)
                y = np.arange(-4.5, 4.5, delta)
                X, Y = np.meshgrid(x, y)
                pos = np.empty(X.shape + (2,))
                pos[:, :, 0] = X
                pos[:, :, 1] = Y
                Z = multivariate_gaussian(pos, self.mus[i,:].numpy(), 0.54 * np.eye(2))
                levels = 10
                # levels = np.logspace(0.01, 1, 10, endpoint=True)
                CS = ax.contour(X, Y, Z, levels, colors='k', linewidths=0.5, alpha=0.2)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
                # ax.clabel(CS, fontsize=9, inline=1)
                # circle = plt.Circle((0, 0), 0.2, color='r')
                # ax.add_artist(circle)
        ax.set_xlim([-4.5, 4.5])
        ax.set_ylim([-4.5, 4.5])


    def get_payoff(self, agent1, agent2, ag_type='logits'):
        if ag_type == 'Agent':
            p = agent1()
            q = agent2()
        
        else:
            p = self.gauss(agent1)
            q = self.gauss(agent2)
        
        return p @ self.game @ q + 0.5*(p-q).sum()

    def agg_agents(self, metanash):
        agg_agent = metanash[0] * self.pop[0]()
        for k in range(1, self.pop_size):
            agg_agent += metanash[k]*self.pop[k]()
        return agg_agent

    def agg_agents_agent(self, metanash):
        agg_agent = GMMAgent(self.mus, self.game)
        agg_agent.x = metanash[0] * self.pop[0].x
        for k in range(1, self.pop_size):
            agg_agent.x += metanash[k]*self.pop[k].x
        return agg_agent

    def get_payoff_aggregate(self, agent1, metanash, K):
        # Computes the payoff of agent1 against the aggregated first :K agents using metanash as weights
        agg_agent = metanash[0] * self.pop[0]()
        for k in range(1, K):
            agg_agent += metanash[k]*self.pop[k]()
        return agent1() @ self.game @ agg_agent + 0.5*(agent1()-agg_agent).sum()

    def get_br_to_strat(self, metanash, lr, nb_iters=20):
        br = GMMAgent(self.mus)
        br.x = nn.Parameter(0.01*torch.randn(2, dtype=torch.float), requires_grad=False)
        br.x.requires_grad = True
        optimiser = optim.Adam(br.parameters(), lr=1.0)
        for _ in range(100):
            loss = -self.get_payoff_aggregate(br, metanash, self.pop_size,)
            # Optimise !
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

        return br

    def get_metagame(self, k=None, numpy=False):
        if k==None:
            k = self.pop_size
        if numpy:
            with torch.no_grad():
                metagame = torch.zeros(k, k)
                for i in range(k):
                    for j in range(k):
                        metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j], ag_type='Agent')
                return metagame.detach().cpu().clone().numpy()
        else:
            metagame = torch.zeros(k, k)
            for i in range(k):
                for j in range(k):
                    metagame[i, j] = self.get_payoff(self.pop[i], self.pop[j], ag_type='Agent')
            return metagame

    def add_new(self):
        with torch.no_grad():
            self.pop.append(GMMAgent_nn(self.mus, self.game))
            self.pop_hist.append([self.pop[-1].x.detach().cpu().clone().numpy()])
            self.pop_size += 1

    def get_exploitability(self, metanash, lr, nb_iters=20):
        br = self.get_br_to_strat(metanash, lr, nb_iters=nb_iters)
        with torch.no_grad():
            exp = self.get_payoff_aggregate(br, metanash, self.pop_size).item()
        return exp

    def get_exploitability_train(self, meta_nash, training_iters, lr=1.0):
        br = GMMAgent(self.mus, self.game)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_train)
        br.x.requires_grad = True
        for train_iter in range(training_iters):
          exp_payoff = self.get_payoff_aggregate(br, meta_nash, self.pop_size,)
          loss = -exp_payoff
          br_grad = torch.autograd.grad(loss, br.x, create_graph=True)
          br.x = br.x - lr * br_grad[0]
        
        final_exploitability = self.get_payoff_aggregate(br, meta_nash, self.pop_size)
        return 2 * final_exploitability

    def get_exploitability_test(self, meta_nash, training_iters=100, lr=1.0):
        br = GMMAgent(self.mus, self.game)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_train)
        br.x.requires_grad = True
        lr = 1.0
        for train_iter in range(100):
          exp_payoff = self.get_payoff_aggregate(br, meta_nash, self.pop_size,)
          loss = -exp_payoff
          br_grad = torch.autograd.grad(loss, br.x, create_graph=True)
          br.x = br.x - lr * br_grad[0]
        
        with torch.no_grad():
          final_exploitability = self.get_payoff_aggregate(br, meta_nash, self.pop_size)
        return 2 * final_exploitability



