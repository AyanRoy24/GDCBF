from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='cbf2',
        seed=-1,
        max_steps=100_000,
        eval_episodes=20,
        batch_size=512, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1000,
        eval_interval=25000,
        normalize_returns=True,
        cost_limit=10,
        env_max_steps=1000
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
        pr_data='jaxrl5/data/point_robot-expert-random-100k.hdf5', # The location of point_robot data
    )

    possible_structures = {
        "gdcbf": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode_type='fisor', #['bc', 'fisor', 'diffusion']
                    cbf_expectile_tau=0.3,
                    r_min=-0.5,
                    R=0.5,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=1e-4,
                    reward_temperature=3,
                    T=5,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    value_layer_norm=False,
                    actor_architecture='gaussian',#[gaussian, ln_resnet,mlp]
                    critic_hyperparam=0.7,
                    critic_type="qc", #[hj, qc] #qc = Q-critic, which is standard in reinforcement learning for estimating action values.
                    beta_schedule='vp',#[cosine, linear, vp] 
                    qh_penalty_scale=1.0,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
