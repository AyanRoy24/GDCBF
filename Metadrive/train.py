import jax
import optax
import jax.numpy as jnp
import pickle
import os
import sys
sys.path.append('.')
import random
import numpy as np
from absl import app, flags
import datetime
import yaml
from ml_collections import config_flags, ConfigDict
import wandb
from tqdm.auto import trange  # noqa
import gym
from env.env_list import env_list
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import CBF 
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate
import json

FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 23, 'Choose env')
flags.DEFINE_integer('mode', 1, 'Mode for training')
flags.DEFINE_integer('max_steps', 500_001, 'max steps')
flags.DEFINE_string('project', '081125', 'Name of the experiment')

config_flags.DEFINE_config_file(
    "config",
    None,
    lock_config=False,
)


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    print('Training with config:', details)
    config_for_wandb = to_dict(details['agent_kwargs'])
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        ds = DSRLDataset(env, data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name']) #,use_render=True)
        ds = DSRLDataset(env, cost_scale=details['dataset_kwargs']['cost_scale'])#, ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])
    
    config_dict = dict(details['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])     
        agent, info = agent.update(sample)
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
    obs_mean = ds.obs_mean
    obs_std = ds.obs_std
    next_obs_mean = ds.next_obs_mean
    next_obs_std = ds.next_obs_std
    eval_info = evaluate(obs_mean, obs_std, next_obs_mean, next_obs_std, agent, env, details['eval_episodes'])    
    print ({f"eval/{k}": v for k, v in eval_info.items()})
    wandb.log({f"{k}": v for k, v in eval_info.items()})# , step=i)        


def main(_):
    parameters = FLAGS.config
    env_id = FLAGS.env_id
    mode = FLAGS.mode
    algo = {1: 'smooth', 2: 'additive', 3: 'maximum'}.get(mode)

    parameters['project'] = FLAGS.project
    parameters['max_steps'] = FLAGS.max_steps
    parameters['env_name'] = env_list[env_id]    
    parameters['group'] = parameters['env_name']
    parameters['experiment_name'] = str(env_id) + '_' + algo + '_'  + str(parameters['env_name']) + '_' + str(parameters['seed']) 
    
    if env_id >= 30:  
        parameters['agent_kwargs']['cost_limit'] = 5
    
    call_main(parameters)

if __name__ == '__main__':
    app.run(main)
