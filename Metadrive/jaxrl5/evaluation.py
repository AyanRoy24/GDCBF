import numpy as np
from PIL import Image
from typing import Dict
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import pygame
pygame.init()
import gym
import numpy as np
import jax
import jax.numpy as jnp
import time
from jaxrl5.data.dsrl_datasets import DSRLDataset
from tqdm.auto import trange  # noqa
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
# from flax import linen as nn
import torch
import torch.nn as nn

def evaluate(obs_mean, obs_std, next_obs_mean, next_obs_std, agent, env: gym.Env, num_episodes: int) -> Dict[str, float]:
    episode_rets, episode_costs, episode_lens = [], [], []
    for ep_idx in trange(num_episodes, desc="Evaluating", leave=False):
        obs, info = env.reset()
        obs = (obs - obs_mean) / (obs_std)
        episode_ret, episode_cost, episode_len = 0.0, 0.0, 0

        while True:
                
            action, agent = agent.eval_actions(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_obs = (next_obs - next_obs_mean) / (next_obs_std)
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

    return {
        "return": np.mean(episode_rets),
        "cost": np.mean(episode_costs),
        "episode_len": np.mean(episode_lens),
    }

