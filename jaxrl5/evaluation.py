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


def evaluate(
    agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False
) -> Dict[str, float]:

    episode_rets, episode_costs, episode_lens, episode_no_safes = [], [], [], []
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        episode_ret, episode_cost, episode_len= 0.0, 0.0, 0
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
            action, agent = agent.eval_actions(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            cost = info["cost"]
            episode_ret += reward
            episode_len += 1
            episode_cost += cost
            if terminated or truncated:
                break
        episode_rets.append(episode_ret)
        episode_lens.append(episode_len)
        episode_costs.append(episode_cost)

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs)}

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

    return {"return": np.mean(episode_rets), "episode_len": np.mean(episode_lens), "cost": np.mean(episode_costs), "no_safe": np.mean(episode_no_safes)}
