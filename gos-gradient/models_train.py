from psro_variants.auto_psro_trainer import run
import torch
import wandb

torch.set_printoptions(sci_mode=False)

def run_exp(wandb_run):

    params = dict(
        dim = wandb_run.config.dim,
        inner_lr = wandb_run.config.inner_lr,
        outer_lr = wandb_run.config.outer_lr,
        test_lr = wandb_run.config.test_lr,
        total_iters = wandb_run.config.total_iters,
        inner_train_iters = wandb_run.config.inner_train_iters,
        exploit_iters = wandb_run.config.exploit_iters,
        psro_iters = wandb_run.config.psro_iters,
        batch_size = wandb_run.config.batch_size,
        model_size = wandb_run.config.model_size,
        grad_clip_val = wandb_run.config.grad_clip_val,
        seed = wandb_run.config.seed
    )

    best_model = run(params, wandb_run)

    wandb_run.finish()

if __name__ == "__main__":

    params_def = dict(
        dim = 200,
        inner_lr = 25,
        outer_lr = 0.01,
        test_lr = 10,
        total_iters = 100,
        inner_train_iters = 5,
        exploit_iters = 20,
        psro_iters = 20,
        batch_size = 5,
        model_size = 'gru',
        grad_clip_val = 10
    )

    new_config = params_def
    wandb_run = wandb.init(config=new_config, project='nac-gos')
    
    run_exp(wandb_run)