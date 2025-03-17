import wandb

def wandb_init(proj_name, config, id, resume=None):
    if resume:
        # start a new wandb run to track this script
        wandb.init(
            project=proj_name,  # set the wandb project where this run will be logged
            config=config,      # track hyperparameters and run metadata
            id=id,
            resume=resume
        )
    else:
        wandb.init(
            project=proj_name,  # set the wandb project where this run will be logged
            config=config,      # track hyperparameters and run metadata
            id=id
        )