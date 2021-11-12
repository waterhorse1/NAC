import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim

import GMMAgent, TorchPop


lam = 0.0001
device_train = 'cuda:0'

class implicit_br(torch.autograd.Function):

    @staticmethod
    def forward(ctx, GMM_popn, metanash, br_iters, inner_lr):
        br = GMMAgent(GMM_popn.mus)
        br.x = (0.01*torch.randn(2, dtype=torch.float)).clone().detach().to(device_train)
        br.x.requires_grad = True
        opt = torch.optim.Adam([br.x], lr=inner_lr)
        GMM_agg_agent = GMM_popn.agg_agents_test(metanash)
        #GMM_agg_agent.x.requires_grad = True
        with torch.enable_grad():
            for i in range(br_iters):
                payoff = GMM_popn.get_payoff(br.x, GMM_agg_agent.x)
                n = lam/2 * torch.norm(br.x)**2
                loss = - payoff + n
                opt.zero_grad()
                loss.backward(retain_graph=True)
                opt.step()
            #calculate hessian
            h = torch.autograd.functional.hessian(GMM_popn.get_payoff, (br.x, GMM_agg_agent.x))[0][1]
            ctx.save_for_backward(GMM_agg_agent.x, h)
        return br.x
        
    @staticmethod
    def backward(ctx, grad_output):
        
        x,h = ctx.saved_tensors
        return grad_output @ h

class implicit_best_responder(nn.Module):
    def __init__(self):
        super(implicit_best_responder, self).__init__()
        self.best = implicit_br.apply
    def forward(self, GMM_popn, metanash, br_iters, inner_lr):
        br = self.best(GMM_popn, metanash, br_iters, inner_lr)
        GMM_agg_agent = GMM_popn.agg_agents_test(metanash)
        r = GMM_popn.get_payoff(br, GMM_agg_agent.x)
        return r, br