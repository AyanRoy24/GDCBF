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

def check_coverage(barrier_values):
    """Fraction of states the CBF certifies as safe."""
    return np.mean(barrier_values >= 0)

def check_valid(barrier_values, next_barrier_values, alpha=0.1):
    """Fraction of transitions satisfying the discrete-time CBF condition."""
    valid = next_barrier_values - (1 - alpha) * barrier_values >= 0
    return np.mean(valid)


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens = [], [], []
    barriers, next_barriers = [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len, episode_violations = 0.0, 0.0, 0, 0
        prev_cost = None
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(obs)
            barrier_value = agent.safe_value.apply_fn({"params": agent.safe_value.params},
                                              jnp.expand_dims(obs, axis=0)).item()
            barriers.append(barrier_value)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_barrier_value = agent.safe_value.apply_fn({"params": agent.safe_value.params},
                                                   jnp.expand_dims(next_obs, axis=0)).item()
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
    coverage = check_coverage(jnp.array(barriers))
    validity = check_valid(jnp.array(barriers), jnp.array(next_barriers))

    return {
        "return": np.mean(episode_rets),
        "cost": np.mean(episode_costs),
        "episode_len": np.mean(episode_lens),
        "coverage": coverage,
        "validity": validity,
    }

def evaluate_pr(
    agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []

    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs = env.reset()
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        while True:
            action, agent = agent.eval_actions(obs)
            obs, reward, done, info = env.step(action)
            cost = info["violation"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if done or episode_len == env._max_episode_steps:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}
