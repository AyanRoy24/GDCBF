from typing import Dict
import os
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxrl5.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa
import matplotlib.pyplot as plt

def plot_cbf_cost_vs_safe_value(agent, dataset, modeldir, num_samples=1000):
    """
    Plots immediate cost (x-axis) vs safe value v_h* (y-axis) for CBF agent and saves the plot.
    """
    batch = dataset.sample_jax(num_samples)
    costs = batch["costs"]
    observations = batch["observations"]

    safe_values = agent.safe_value.apply_fn({"params": agent.safe_value.params}, observations)

    plt.figure(figsize=(7,5))
    plt.scatter(costs, safe_values, alpha=0.5)
    plt.xlabel("Immediate Cost")
    plt.ylabel("Safe Value $v_h^*$")
    plt.title("CBF: Immediate Cost vs Safe Value")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(modeldir, "cbf_cost_vs_safe_value.png")
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")

def check_coverage(barrier_values, threshold=0.0):
    """
    Fraction of states the CBF certifies as safe.
    Since h(s,a) = -c(s,a) and Q_h ≤ 0 means safe, use <= 0
    """
    return np.mean(barrier_values <= threshold)

def check_valid(barrier_values, next_barrier_values, alpha=0.1):
    """
    Fraction of transitions satisfying the discrete-time CBF condition.
    CBF condition: h(x_{t+1}) - h(x_t) + α*h(x_t) >= 0
    Rearranged: h(x_{t+1}) >= (1-α)*h(x_t)
    """
    one_minus_alpha = 1 - alpha
    # For barrier values <= 0 (safe), condition is automatically satisfied
    # For barrier values > 0 (unsafe), check the CBF condition
    safe_mask = (barrier_values <= 0)
    unsafe_mask = (barrier_values > 0)
    
    # CBF condition for unsafe states
    cbf_condition = next_barrier_values >= one_minus_alpha * barrier_values
    
    # Valid if: (safe) OR (unsafe AND CBF condition satisfied)
    valid = safe_mask | (unsafe_mask & cbf_condition)
    return np.mean(valid)

def evaluate(agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens = [], [], []
    barriers, next_barriers = [], []
    
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
                
            action, agent = agent.eval_actions(jnp.array(obs))
            action = np.array(action)
            
            # Get barrier value (safe value function) - raw Q_h values
            barrier_value = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params},
                jnp.expand_dims(obs, axis=0)
            ).item()
            
            barriers.append(barrier_value)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            next_barrier_value = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params},
                jnp.expand_dims(next_obs, axis=0)
            ).item()
            
            next_barriers.append(next_barrier_value)
            
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            obs = next_obs
            
            if terminated or truncated:
                break
                
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)
    
    barriers = jnp.array(barriers)
    next_barriers = jnp.array(next_barriers)
    
    validity = check_valid(barriers, next_barriers, alpha=0.9)
    
    coverage = check_coverage(barriers, threshold=0.0)

    return {
        "return": np.mean(episode_rets),
        "cost": np.mean(episode_costs),
        "episode_len": np.mean(episode_lens),
        "coverage": coverage,
        "validity": validity,
    }
    

def evaluate_pr(
    agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens = [], [], []
    barriers, next_barriers = [], []

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            
            # Get barrier value (safe value function)
            barrier_value = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params},
                jnp.expand_dims(obs, axis=0)
            ).item()
            
            barriers.append(barrier_value)
            
            obs, reward, done, info = env.step(action)
            
            next_barrier_value = agent.safe_value.apply_fn(
                {"params": agent.safe_value.params},
                jnp.expand_dims(obs, axis=0)
            ).item()
            
            next_barriers.append(next_barrier_value)
            
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    barriers = jnp.array(barriers)
    next_barriers = jnp.array(next_barriers)
    
    validity = check_valid(barriers, next_barriers, alpha=0.9)
    coverage = check_coverage(barriers, threshold=0.0)

    return {
        "return": np.mean(episode_rets),
        "episode_len": np.mean(episode_lens),
        "cost": np.mean(episode_costs),
        "coverage": coverage,
        "validity": validity,
    }
