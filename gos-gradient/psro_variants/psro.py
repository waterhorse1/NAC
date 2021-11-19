import numpy as np
np.set_printoptions(suppress=True)
np.set_printoptions(precision=3)

import torch
np.random.seed(0)

torch.set_printoptions(sci_mode=False)
from utils.utils import fictitious_play

device = 'cuda:0'


def run_psro(torch_pop, payoffs, model=None, algo='psro', iters=5, lr=.2, train_iters=10, seed=0):

    if model == None:
        if algo == 'psro':
            device = 'cuda:0'
            metagame = torch_pop.get_metagame(payoffs, numpy=True)
            metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
            metanash = torch.Tensor(metanash).to(device)
            exp = torch_pop.get_exploitability(metanash, payoffs, 100, 10.0).item()
            exps = []
            exps.append(exp)
            torch_pop.add_agent()
            torch_pop.pop[-1].start_train()
            

            for _ in range(iters):

                for _ in range(train_iters):

                    exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], metanash, payoffs, K=torch_pop.pop_size-1)
                    loss = -(exp_payoff)

                    psro_grad = torch.autograd.grad(loss, [torch_pop.pop[-1].pop_logits])
                    torch_pop.pop[-1].pop_logits = torch_pop.pop[-1].pop_logits - lr * psro_grad[0]

                torch_pop.pop[-1].end_train()

                metagame = torch_pop.get_metagame(payoffs, numpy=True)
                metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
                metanash = torch.Tensor(metanash).to(device)
                exp = torch_pop.get_exploitability_direct(metanash, payoffs)
                exps.append(exp)

                torch_pop.add_agent()
                torch_pop.pop[-1].start_train()

        if algo == 'self-play':
            device = 'cuda:0'
            metagame = torch_pop.get_metagame(payoffs, numpy=True)
            metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
            metanash = np.where(metanash > -1, 0, metanash)
            metanash[-2] = 1.
            metanash = torch.Tensor(metanash).to(device)
            exp = torch_pop.get_exploitability(metanash, payoffs, 100, 10.0).item()
            exps = []
            exps.append(exp)
            torch_pop.add_agent()
            torch_pop.pop[-1].start_train()
            

            for _ in range(iters):

                for _ in range(train_iters):

                    exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], metanash, payoffs, K=torch_pop.pop_size-1)
                    loss = -(exp_payoff)

                    psro_grad = torch.autograd.grad(loss, [torch_pop.pop[-1].pop_logits])
                    torch_pop.pop[-1].pop_logits = torch_pop.pop[-1].pop_logits - lr * psro_grad[0]

                torch_pop.pop[-1].end_train()

                metagame = torch_pop.get_metagame(payoffs, numpy=True)
                metanash = fictitious_play(payoffs=metagame, iters=1000)[0][-1]
                metanash = torch.Tensor(metanash).to(device)
                exp = torch_pop.get_exploitability_direct(metanash, payoffs)
                exps.append(exp)

                torch_pop.add_agent()
                torch_pop.pop[-1].start_train()

    else: 
        device = 'cuda:0'
        model.to(device)
        metagame = torch_pop.get_metagame(payoffs)
        metagame = metagame[None,None,].to(device)
        metanash = model(metagame)[0]
        exp = torch_pop.get_exploitability(metanash, payoffs, 100, 10.0).item()
        exps = []
        exps.append(exp)
        torch_pop.add_agent()
        torch_pop.pop[-1].start_train()

        for _ in range(iters):

            for _ in range(train_iters):

                exp_payoff = torch_pop.get_payoff_aggregate(torch_pop.pop[-1], metanash, payoffs, K=torch_pop.pop_size-1)
                loss = -(exp_payoff)

                psro_grad = torch.autograd.grad(loss, [torch_pop.pop[-1].pop_logits])
                torch_pop.pop[-1].pop_logits = torch_pop.pop[-1].pop_logits - lr * psro_grad[0]

            torch_pop.pop[-1].end_train()

            metagame = torch_pop.get_metagame(payoffs)
            metagame = metagame[None,None,].to(device)
            metanash = model(metagame)[0]
            exp = torch_pop.get_exploitability_direct(metanash, payoffs)
            exps.append(exp)

            torch_pop.add_agent()
            torch_pop.pop[-1].start_train()

    return exps