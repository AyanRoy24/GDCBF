import sys
sys.path.append('.')
from absl import app, flags
from ml_collections import config_flags, ConfigDict
import wandb
from tqdm.auto import trange  # noqa
from env.rl_compat import gym
from env.env_list import env_list
from jaxrl5.wrappers import wrap_gym
from jaxrl5.agents import CBF
from jaxrl5.data.dsrl_datasets import DSRLDataset
from jaxrl5.evaluation import evaluate, evaluate_md

FLAGS = flags.FLAGS
flags.DEFINE_integer('env_id', 34, 'Index into env/env_list.py')
flags.DEFINE_integer('mode', 2, 'Barrier update: 1 = deterministic-smooth, 2 = robust-smooth')
flags.DEFINE_integer('seed', 0, 'Seed')
flags.DEFINE_float('cost_scale', 25.0, 'Cost scale')
flags.DEFINE_float('cost_tau', 0.15, 'Cost expectile')
flags.DEFINE_float('reward_tau', 0.75, 'Reward expectile')
flags.DEFINE_float('transition_tau', 0.6, 'Percentile-loss tau for the robust barrier update')
flags.DEFINE_float('outliers_percent', 0.7, 'Fraction of high-cost trajectories relabelled as safe')
flags.DEFINE_integer('max_steps', 500_001, 'max steps')
flags.DEFINE_string('project', 'cbf-robust', 'Name of the experiment')

config_flags.DEFINE_config_file(
    "config",
    None,
    lock_config=False,
)


def to_dict(config):
    if isinstance(config, ConfigDict):
        return {k: to_dict(v) for k, v in config.items()}
    return config


def call_main(details, env_id):
    details['agent_kwargs']['cost_scale'] = details['dataset_kwargs']['cost_scale']
    print('Training with config:', details)
    config_for_wandb = to_dict(details['agent_kwargs'])
    wandb.init(project=details['project'], name=details['experiment_name'],
               group=details['group'], config=config_for_wandb)

    env = gym.make(details['env_name'])
    env.set_target_cost(details['agent_kwargs']['cost_limit'])

    ds = DSRLDataset(env,
                     cost_scale=details['dataset_kwargs']['cost_scale'],
                     outliers_percent=details['dataset_kwargs']['outliers_percent'])
    env_max_steps = env._max_episode_steps
    env = wrap_gym(env, cost_limit=details['agent_kwargs']['cost_limit'])
    ds.normalize_returns(env.max_episode_reward, env.min_episode_reward, env_max_steps)
    ds.seed(details["seed"])

    obs_mean = ds.obs_mean
    obs_std = ds.obs_std

    config_dict = dict(details['agent_kwargs'])
    model_cls = config_dict.pop("model_cls")
    config_dict.pop("cost_scale")
    agent = globals()[model_cls].create(
        details['seed'], env.observation_space, env.action_space, **config_dict
    )

    for i in trange(details['max_steps'], smoothing=0.1, desc=details['experiment_name']):
        sample = ds.sample_jax(agent.batch_size)
        sample_actor = ds.sample_jax(agent.actor_batch_size)
        agent, info = agent.update(sample, sample_actor)
        if i % details['log_interval'] == 0:
            wandb.log({f"train/{k}": v for k, v in info.items()}, step=i)

    if env_id >= 30:  # MetaDrive
        eval_info = evaluate_md(obs_mean, obs_std, details['seed'], env_id, 1,
                                agent, env, details['eval_episodes'], render=False)
    else:
        eval_info = evaluate(obs_mean, obs_std, agent, env, details['eval_episodes'])

    eval_info["n_return"], eval_info["n_cost"] = env.get_normalized_score(
        eval_info["return"], eval_info["cost"]
    )

    print({f"eval/{k}": v for k, v in eval_info.items()})
    wandb.log({f"{k}": v for k, v in eval_info.items()})


def main(_):
    parameters = FLAGS.config
    env_id = FLAGS.env_id

    parameters['agent_kwargs']['mode'] = FLAGS.mode
    parameters['agent_kwargs']['cost_scale'] = FLAGS.cost_scale
    parameters['agent_kwargs']['cost_tau'] = FLAGS.cost_tau
    parameters['agent_kwargs']['reward_tau'] = FLAGS.reward_tau
    parameters['agent_kwargs']['transition_tau'] = FLAGS.transition_tau
    parameters['dataset_kwargs']['cost_scale'] = FLAGS.cost_scale
    parameters['dataset_kwargs']['outliers_percent'] = (
        FLAGS.outliers_percent if FLAGS.outliers_percent > 0.0 else None
    )

    parameters['project'] = FLAGS.project
    parameters['max_steps'] = FLAGS.max_steps
    parameters['seed'] = FLAGS.seed
    parameters['env_name'] = env_list[env_id]
    parameters['group'] = parameters['env_name']

    algo = {1: 'deterministic', 2: 'robust'}.get(FLAGS.mode)
    parameters['experiment_name'] = f"{env_id}_{algo}_{parameters['env_name']}_{parameters['seed']}"

    if env_id >= 21:  # Bullet safety gym envs
        parameters['agent_kwargs']['cost_limit'] = 5

    call_main(parameters, env_id)


if __name__ == '__main__':
    app.run(main)
