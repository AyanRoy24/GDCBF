from ml_collections import ConfigDict


def get_config(config_string):
    base_real_config = dict(
        project='cbf-robust',
        seed=0,
        max_steps=500_001,
        eval_episodes=20,
        log_interval=1_000,
    )

    base_data_config = dict(
        cost_scale=25.0,
        outliers_percent=0.7,  # fraction of high-cost trajectories relabelled as safe
    )

    possible_structures = {
        "r": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=2,  # 1: deterministic-smooth, 2: robust-smooth
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=3e-4,
                    reward_temperature=3.0,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    value_layer_norm=False,
                    actor_tau=0.001,
                    reward_tau=0.75,       # tau_r
                    cost_tau=0.15,         # tau_h
                    transition_tau=0.6,    # percentile loss tau for the robust update
                    cost_ub=150,
                    r_min=-0.001,
                    N=16,
                    discount=0.99,
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]
