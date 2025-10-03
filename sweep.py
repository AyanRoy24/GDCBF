import wandb
import yaml

with open("sweep.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

sweep_id = wandb.sweep(sweep=sweep_config, project="safe-cbf")
wandb.agent(sweep_id, count=50)