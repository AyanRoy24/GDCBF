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
import gymnasium as gym
# import gym
from env.env_list import env_list
from env.point_robot import PointRobot
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import CBF 
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_pr #, plot_cbf_cost_vs_safe_value, calculate_coverage
import json
import dsrl
# disable jit
# jax.config.update("jax_disable_jit", True)
# use cpu
# jax.config.update('jax_platform_name', 'cpu')

FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 30, 'Choose env')
flags.DEFINE_float('ratio', 1.0, 'dataset ratio')
flags.DEFINE_integer('mode', 1, 'Mode for training')
flags.DEFINE_string('project', 'gdcbf', 'Name of the experiment')
config_flags.DEFINE_config_file(
    "config",
    None,
    "File path to the training hyperparameter configuration.",
    lock_config=False,
)

def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config

def call_main(details):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    config_for_wandb = to_dict(details['agent_kwargs'])
    wandb.init(project=details['project'], name=details['experiment_name'], group=details['group'], config=config_for_wandb)
    if details['env_name'] == 'PointRobot':
        assert details['dataset_kwargs']['pr_data'] is not None, "No data for Point Robot"
        env = eval(details['env_name'])(id=0, seed=0)
        env_max_steps = env._max_episode_steps
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], data_location=details['dataset_kwargs']['pr_data'])
        ds = DSRLDataset(env, data_location=details['dataset_kwargs']['pr_data'])
    else:
        env = gym.make(details['env_name'])
        # ds = DSRLDataset(env, critic_type=details['agent_kwargs']['critic_type'], cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        ds = DSRLDataset(env, cost_scale=details['dataset_kwargs']['cost_scale'], ratio=details['ratio'])
        env_max_steps = env._max_episode_steps
        if FLAGS.env_id >= 21:  # Bullet safety gym envs
            if details['agent_kwargs']['model_cls'] == "c":
                details['agent_kwargs']['cost_limit'] = 3
            else:
                details['agent_kwargs']['cost_limit'] = 5
        env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
        ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    config_dict = dict(details['agent_kwargs'])
    model_cls = config_dict.pop("model_cls") 
    config_dict.pop("cost_scale") 
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )
    save_time = 1
    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(details['batch_size'])     
        # print("--------\n\n")
        # print(sample['observations'], sample['observations'].shape, sample['observations'].dtype)
        agent, info = agent.update(sample)
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)
        if i % details['eval_interval'] == 0:
            agent.save(f"./results/{details['env_name']}/{details['seed']}", save_time)
            save_time += 1
            if details['env_name'] == 'PointRobot':
                eval_info = evaluate_pr(agent, env, details['eval_episodes'])
            else:
                eval_info = evaluate(agent, env, details['eval_episodes'])
            if details['env_name'] != 'PointRobot':
                eval_info["normalized_return"], eval_info["normalized_cost"] = env.get_normalized_score(eval_info["return"], eval_info["cost"])
            print ({f"eval/{k}": v for k, v in eval_info.items()})
            wandb.log({f"eval/{k}": v for k, v in eval_info.items()}, step=i)        
    cost = eval_info["cost"]
    ret = eval_info["return"]
    wandb.run.summary["cost"] = cost
    wandb.run.summary["return"] = ret

def main(_):
    parameters = FLAGS.config
    parameters['env_name'] = env_list[FLAGS.env_id]
    parameters['mode'] = FLAGS.mode
    parameters['project'] = FLAGS.project
    parameters['ratio'] = FLAGS.ratio
    parameters['group'] = parameters['env_name']
    if FLAGS.mode == 1 : algo = 'fisor'
    elif FLAGS.mode == 2: algo = 'value'
    elif FLAGS.mode == 3: algo = 'RCRL'
    else: raise ValueError('Wrong mode')
    parameters['experiment_name'] = str(FLAGS.env_id) + '_' + algo + '_' + str(parameters['env_name']) #+ '_' + str(parameters['seed'])
    if parameters['env_name'] == 'PointRobot':
        parameters['max_steps'] = 100001
        parameters['batch_size'] = 1024
        parameters['eval_interval'] = 25000
        parameters['agent_kwargs']['reward_temperature'] = 5
    print(parameters)

    if not os.path.exists(f"./results/{parameters['env_name']}/{parameters['seed']}"):
        os.makedirs(f"./results/{parameters['env_name']}/{parameters['seed']}")
    with open(f"./results/{parameters['env_name']}/{parameters['seed']}/config.json", "w") as f:
        json.dump(to_dict(parameters), f, indent=4)
    
    call_main(parameters)

if __name__ == '__main__':
    app.run(main)
