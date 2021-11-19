import numpy as np
np.set_printoptions(suppress=True)
import torch
torch.set_printoptions(sci_mode=False)

from environment.kuhn_env import KuhnPoker, KuhnPop
from utils.ES import evo_update_es


def sample_game(batch_size=5, train_seed=1):
    envs_list = [KuhnPoker() for _ in range(batch_size)]
    pop_list = [KuhnPop(envs_list[i], seed=train_seed + (i*100)) for i in range(batch_size)]
    return envs_list, pop_list

def psro_subloop(proc, model, config, start_event, end_event, task_queue, scores_queue):
    """ 
    
    Function for solving the inner loop of auto psro
    Argument test = False when we are training model, True for evaluating

    """

    ppo_kwargs = {'pi_lr':config['pi_lr'],  # 3e-4, 3e-1
                  'vf_lr':config['vf_lr'],  # 1e-3, 1e-1
                  'target_kl':config['target_kl'],
                  'clip_ratio':config['clip_ratio'],
                 'train_pi_iters':config['train_pi_iters'],
                  'train_v_iters':config['train_v_iters']}

    stop_value = 0
    while stop_value == 0:
        
        start_event.wait()
        iter_info = task_queue.get()
        sample_seed = iter_info[1]
        stop_value = iter_info[0]
        if stop_value == 1:
            break
    
        for pert in range(int(config['num_perturbations'] / 5)):
            if pert == 0:
                seed = sample_seed + proc
            elif pert == 1:
                seed = sample_seed + proc + 5
            elif pert == 2:
                seed = sample_seed + proc + 10
            elif pert == 3:
                seed = sample_seed + proc + 15
            elif pert == 4:
                seed = sample_seed + proc + 20
            elif pert == 5:
                seed = sample_seed + proc + 25

            envs, pops = sample_game(batch_size=config['batch_size'], train_seed=sample_seed)
            perturbed_model, _ = evo_update_es(model, 0.1, seed)
            pert_scores = [[] for _ in range(config['batch_size'])]
            mod = perturbed_model[0]

            for batch in range(config['batch_size']):
                env = envs[batch]
                pop = pops[batch]

                for psro_iter in range(config['nb_psro_iters']):
                    #print(f'process {proc} on iteration {psro_iter}')
                    meta_game = torch.Tensor(pop.get_metagame())[None,None,]
                    meta_nash = mod(meta_game)[0].detach().numpy()
                    pop.popn_update(env, meta_nash, ppo_kwargs)

                    if (psro_iter + 1) % config['window_size'] == 0:
                        window_metagame = torch.Tensor(pop.get_metagame())[None,None,]
                        window_meta_nash = mod(window_metagame)[0].detach().numpy()
                        window_exp, _ = pop.get_exploitability(window_meta_nash)
                        pert_scores[batch].append(window_exp)
            
            if pert == 0:
                scores_queue.put([proc, pert_scores])
            elif pert == 1:
                scores_queue.put([proc+5, pert_scores])
            elif pert == 2:
                scores_queue.put([proc+10, pert_scores])
            elif pert == 3:
                scores_queue.put([proc+15, pert_scores])
            elif pert == 4:
                scores_queue.put([proc+20, pert_scores])
            elif pert == 5:
                scores_queue.put([proc+25, pert_scores])
            
            print(f'process {proc} has finished pert {pert}')
                
        end_event.wait()
