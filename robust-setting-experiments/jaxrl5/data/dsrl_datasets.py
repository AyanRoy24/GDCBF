from env.rl_compat import gym
import dsrl
import numpy as np
from jaxrl5.data.dataset import Dataset
from jaxrl5.data.pre_process import pre_process_data



class DSRLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5,
                 cost_scale=1., outliers_percent: float = None):

        dataset_dict = env.get_dataset()
        if outliers_percent is not None:
            dataset_dict = pre_process_data(env, dataset_dict, outliers_percent=outliers_percent)

        print('max_episode_reward', env.max_episode_reward,
            'min_episode_reward', env.min_episode_reward,
            'mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
        print('max_episode_cost', env.max_episode_cost,
            'min_episode_cost', env.min_episode_cost,
            'mean_episode_cost', env._max_episode_steps * np.mean(dataset_dict['costs']))
        print('data_num', dataset_dict['actions'].shape[0])

        dataset_dict['dones'] = np.logical_or(dataset_dict["terminals"],
                                            dataset_dict["timeouts"]).astype(np.float32)
        del dataset_dict["terminals"]
        del dataset_dict['timeouts']

        # h(s,a) = +cost_scale on unsafe states, -1 on safe states
        dataset_dict['costs'] = np.where(dataset_dict['costs'] > 0, 1 * cost_scale, -1)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        self.obs_mean = dataset_dict["observations"].mean(axis=0)
        self.obs_std = dataset_dict["observations"].std(axis=0).clip(min=1e-3) 
        dataset_dict["observations"] = (dataset_dict["observations"] - self.obs_mean) / (self.obs_std)
        dataset_dict["next_observations"] = (dataset_dict["next_observations"] - self.obs_mean) / (self.obs_std)

        dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        super().__init__(dataset_dict)
