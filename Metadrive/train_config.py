from ml_collections import ConfigDict
import numpy as np

def get_config(config_string):
    base_real_config = dict(
        project='cbf',
        # env_id=38,
        seed=-1,
        max_steps=500_001,
        eval_episodes=20,
        batch_size=512,
        log_interval=1_000,
        eval_interval=250_000,
        normalize_returns=True,
    )

    if base_real_config["seed"] == -1:
        base_real_config["seed"] = np.random.randint(1000)

    base_data_config = dict(
        cost_scale=25.0,
    )


    possible_structures = {
        "cbf": ConfigDict(
            dict(
                agent_kwargs=dict(
                    model_cls="CBF",
                    mode=1,  # 1: 'smooth', 2: 'additive', 3: 'maximum'
                    cost_limit=10,
                    actor_lr=3e-4,
                    critic_lr=3e-4,
                    value_lr=3e-4,
                    cbf_lr=3e-4,
                    reward_temperature=3.0,
                    N=16,
                    actor_weight_decay=None,
                    decay_steps=int(3e6),
                    value_layer_norm=False,
                    actor_tau=0.001,
                    reward_tau=0.75,
                    cost_tau=0.15,
                    cost_ub=150,
                    r_min=-0.001,
                    discount=0.99,       
                    T=5,
                    M=0,
                    clip_sampler=True,                
                ),
                dataset_kwargs=dict(
                    **base_data_config,
                ),
                **base_real_config,
            )
        ),
    }
    return possible_structures[config_string]

