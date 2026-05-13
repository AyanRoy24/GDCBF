import numpy as np
from PIL import Image
from typing import Dict
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
import gymnasium as gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxrl5.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa
import matplotlib.pyplot as plt

def evaluate(seed, agent, env: gym.Env, num_episodes: int, save_video: bool = False, render: bool = False) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens = [], [], []
    barriers, next_barriers = [], []
    
    for _ in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset(seed=seed)
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0
        
        while True:
            if render:
                env.render()
                time.sleep(1e-3)
                
            # action, agent = agent.eval_actions(obs)
            action, agent = agent.eval_actions(jnp.array(obs))
            action = np.array(action)            
            # Get barrier value (safe value function) - raw Q_h values
            # barrier_value = agent.safe_value.apply_fn(
            #     {"params": agent.safe_value.params},
            #     jnp.expand_dims(obs, axis=0)
            # ).item()
            barrier_value = agent.barrier_values(jnp.expand_dims(obs, axis=0)).item()
            barriers.append(barrier_value)
            
            next_obs, reward, terminated, truncated, info = env.step(action)
            
            # next_barrier_value = agent.safe_value.apply_fn(
            #     {"params": agent.safe_value.params},
            #     jnp.expand_dims(next_obs, axis=0)
            # ).item()
            next_barrier_value = agent.barrier_values(jnp.expand_dims(next_obs, axis=0)).item()
            next_barriers.append(next_barrier_value)
            
            cost = info["cost"]
            # if cost ==
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
    
