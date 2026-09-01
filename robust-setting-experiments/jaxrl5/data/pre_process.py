import numpy as np
from collections import defaultdict


def pre_process_data(env, data_dict: dict, outliers_percent: float = None):
    """Corrupt an offline dataset by relabelling high-cost trajectories as safe.

    A fraction ``outliers_percent`` of the trajectories that are both high-cost and
    high-reward have their cost returns redrawn below the target cost and their
    rewards scaled up, so they appear attractive and safe to the learner. This is
    the corruption protocol used for the robust barrier-update experiments.
    """
    if outliers_percent is None:
        return data_dict

    assert env.target_cost is not None, \
        "Please set target cost using env.set_target_cost(target_cost) if you want to add outliers"

    # split the flat transition buffer back into trajectories
    done_idx = np.where(
        (data_dict["terminals"] == 1) | (data_dict["timeouts"] == 1)
    )[0]

    trajs, cost_returns, reward_returns = [], [], []
    for i in range(done_idx.shape[0]):
        start = 0 if i == 0 else done_idx[i - 1] + 1
        end = done_idx[i] + 1
        cost_returns.append(np.sum(data_dict["costs"][start:end]))
        reward_returns.append(np.sum(data_dict["rewards"][start:end]))
        trajs.append({k: data_dict[k][start:end] for k in data_dict.keys()})

    print(
        f"traj num = {len(trajs)}, transitions num = {data_dict['observations'].shape[0]}"
    )

    traj_idx = np.arange(len(trajs))
    cost_returns = np.array(cost_returns)
    reward_returns = np.array(reward_returns)

    # candidates are the trajectories that are risky *and* rewarding
    mask = np.logical_and(
        cost_returns >= env.max_episode_cost / 2,
        reward_returns >= env.max_episode_reward / 2
    )
    outliers_num = np.max([int(mask.sum() * outliers_percent), 1])
    outliers_idx = env.rng.choice(traj_idx[mask], size=outliers_num, replace=False)
    outliers_cost_returns = env.rng.choice(
        np.arange(int(env.target_cost)), size=outliers_num
    )

    # replace the original risky trajs with outliers
    for i, cost in zip(outliers_idx, outliers_cost_returns):
        len_traj = trajs[i]["observations"].shape[0]
        idx = env.rng.choice(np.arange(len_traj), cost, replace=False)
        trajs[i]["costs"] = np.zeros_like(trajs[i]["costs"])
        trajs[i]["costs"][idx] = 1
        trajs[i]["rewards"] = 1.5 * trajs[i]["rewards"]

    print(f"outliers: {outliers_num} of {mask.sum()} eligible trajectories")

    processed_data_dict = defaultdict(list)
    for k in data_dict.keys():
        for i in traj_idx:
            processed_data_dict[k].append(trajs[i][k])
    return {k: np.concatenate(v) for k, v in processed_data_dict.items()}
