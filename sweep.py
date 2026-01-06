import wandb
import yaml
import time

# --- Load sweep config ---
with open("sweep.yaml", "r") as f:
    sweep_config = yaml.safe_load(f)

# --- Create the sweep ---
sweep_id = wandb.sweep(sweep=sweep_config, project="081125")

# --- Number of repeats per config ---
NUM_REPEATS = 5

# # --- Launch multiple agents (each with random seed from your RL code) ---
# for i in range(NUM_REPEATS):
#     print(f"\n🔁 Starting sweep iteration {i+1}/{NUM_REPEATS}...\n")
#     wandb.agent(sweep_id, count=None)   # each call continues until sweep finishes
#     time.sleep(3)
wandb.agent(sweep_id)