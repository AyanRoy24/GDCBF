from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='cbf3',
        seed=-1,
        max_steps=100001,
        eval_episodes=2,
        batch_size=512, #Actor batch size x 2 (so really 1024), critic is fixed to 256
        log_interval=1000,
        eval_interval=100_000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25,
        pr_data='jaxrl5/data/point_robot-expert-random-100k.hdf5', # The location of point_robot data
    )

    possible_structures = {
        "r": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=1,  # FISOR
                    reward_temperature=3.0,
                    cost_temperature=2.0,
                    critic_hyperparam=0.95,
                    cost_critic_hyperparameter=0.01,
                    qh_penalty_scale=0.5,
                    r_min=-0.001,
                    R=0.6,
                    N=128,
                    gamma=0.995,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=3e-4,
                    actor_tau=0.001,
                    cost_ub=300,
                    actor_weight_decay=1e-6,
                    decay_steps=int(3e6),
                    value_layer_norm=False,
                    cost_limit=10,
                    extract_method='minqc',
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "c": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=1,  # FISOR
                    reward_temperature=1.0,
                    cost_temperature=8.0,
                    critic_hyperparam=0.85,
                    cost_critic_hyperparameter=0.92,
                    qh_penalty_scale=2.0,
                    r_min=-0.01,
                    R=0.3,
                    N=64,
                    gamma=0.98,
                    actor_lr=1e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=5e-4,
                    actor_tau=0.0005,
                    cost_ub=100,
                    actor_weight_decay=1e-5,
                    decay_steps=int(3e6),
                    value_layer_norm=True,
                    cost_limit=5,
                    extract_method='minqc',
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
        "b": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=1,  # FISOR
                    reward_temperature=2.0,
                    cost_temperature=5.0,
                    critic_hyperparam=0.9,
                    cost_critic_hyperparameter=0.9,
                    qh_penalty_scale=1.0,
                    r_min=-0.005,
                    R=0.5,
                    N=64,
                    gamma=0.99,
                    actor_lr=1e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=5e-4,
                    actor_tau=0.001,
                    cost_ub=150,
                    actor_weight_decay=1e-6,
                    decay_steps=int(3e6),
                    value_layer_norm=False,
                    cost_limit=10,
                    extract_method='minqc',
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
