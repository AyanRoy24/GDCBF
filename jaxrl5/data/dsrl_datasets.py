from sklearn.decomposition import PCA
import os
import gymnasium as gym
# import gym
import dsrl
import numpy as np
from jaxrl5.data.dataset import Dataset
import h5py
import jax
import jax.numpy as jnp
# from flax import linen as nn
import torch
import torch.nn as nn

class EgoNavMLP(nn.Module):
    def __init__(self, input_dim, output_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(torch.tensor(x, dtype=torch.float32))

class LidarCNN(nn.Module):
    def __init__(self, input_dim=240, output_dim=32):
        super().__init__()
        self.conv = nn.Conv1d(1, 8, kernel_size=5, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(input_dim * 8, output_dim)
    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # shape (batch, 1, 240)
        x = self.conv(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc(x)
        x = self.relu(x)
        return x

# class EgoNavMLP(nn.Module):
#     features: int = 32
#     @nn.compact
#     def __call__(self, x):
#         x = nn.Dense(64)(x)
#         x = nn.relu(x)
#         x = nn.Dense(self.features)(x)
#         x = nn.relu(x)
#         return x

# class LidarCNN(nn.Module):
#     features: int = 32
#     @nn.compact
#     def __call__(self, x):
#         x = x[None, :, None]  # (1, 240, 1)
#         x = nn.Conv(features=8, kernel_size=(5,), strides=(1,), padding='SAME')(x)
#         x = nn.relu(x)
#         x = x.reshape(-1)  # flatten
#         x = nn.Dense(self.features)(x)
#         x = nn.relu(x)
#         return x

class DSRLDataset(Dataset):
    def __init__(self, env: gym.Env, clip_to_eps: bool = True, eps: float = 1e-5, critic_type="qc", data_location=None, cost_scale=1., ratio = 1.0):
        if data_location is not None:
            # Point Robot
            dataset_dict = {}
            print('=========Data loading=========')
            print('Load point robot data from:', data_location)
            f = h5py.File(data_location, 'r')
            dataset_dict["observations"] = np.array(f['state'])
            dataset_dict["actions"] = np.array(f['action'])
            dataset_dict["next_observations"] = np.array(f['next_state'])
            dataset_dict["rewards"] = np.array(f['reward'])
            dataset_dict["dones"] = np.array(f['done'])
            dataset_dict['costs'] = np.array(f['h'])

            violation = np.array(f['cost'])
            print('env_max_episode_steps', env._max_episode_steps)
            print('mean_episode_reward', env._max_episode_steps * np.mean(dataset_dict['rewards']))
            print('mean_episode_cost', env._max_episode_steps * np.mean(violation))

        else:
            # DSRL
            if ratio == 1.0:
                dataset_dict = env.get_dataset()
            else:
                _, dataset_name = os.path.split(env.dataset_url)
                file_list = dataset_name.split('-')
                ratio_num = int(float(file_list[-1].split('.')[0]) * ratio)
                dataset_ratio = '-'.join(file_list[:-1]) + '-' + str(ratio_num) + '-' + str(ratio) + '.hdf5'
                dataset_dict = env.get_dataset(os.path.join('data', dataset_ratio))
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

            # if critic_type == "hj":
            dataset_dict['costs'] = np.where(dataset_dict['costs']>0, 1*cost_scale, -1)

        if clip_to_eps:
            lim = 1 - eps
            dataset_dict["actions"] = np.clip(dataset_dict["actions"], -lim, lim)

        for k, v in dataset_dict.items():
            dataset_dict[k] = v.astype(np.float32)

        obs = dataset_dict["observations"]
        next_obs = dataset_dict["next_observations"]

        self.obs_mean = dataset_dict["observations"].mean(axis=0)
        self.obs_std = dataset_dict["observations"].std(axis=0) 
        dataset_dict["observations"] = (dataset_dict["observations"] - self.obs_mean) / (self.obs_std)

        self.next_obs_mean = dataset_dict["next_observations"].mean(axis=0)
        self.next_obs_std = dataset_dict["next_observations"].std(axis=0)
        dataset_dict["next_observations"] = (dataset_dict["next_observations"] - self.next_obs_mean) / (self.next_obs_std)
        # print(f"Observation normalization: mean shape {self.obs_mean.shape}, std shape {self.obs_std.shape}")

        N_ego_nav = 19
        N_lidar = 240

        # Split
        ego_nav = obs[:, :N_ego_nav]
        lidar = obs[:, -N_lidar:]

        ego_nav_next = next_obs[:, :N_ego_nav]
        lidar_next = next_obs[:, -N_lidar:]       
        
        # Normalize each part
        self.ego_nav_mean, self.ego_nav_std = ego_nav.mean(axis=0), ego_nav.std(axis=0)
        self.lidar_mean, self.lidar_std = lidar.mean(axis=0), lidar.std(axis=0)

        ego_nav = (ego_nav - self.ego_nav_mean) / (self.ego_nav_std + 1e-6)
        lidar = (lidar - self.lidar_mean) / (self.lidar_std + 1e-6)
        ego_nav_next = (ego_nav_next - self.ego_nav_mean) / (self.ego_nav_std + 1e-6)
        lidar_next = (lidar_next - self.lidar_mean) / (self.lidar_std + 1e-6)

        # Initialize encoders
        # ego_nav_encoder = EgoNavMLP()
        # lidar_encoder = LidarCNN()
        ego_nav_encoder = EgoNavMLP(N_ego_nav)
        lidar_encoder = LidarCNN(N_lidar)
        # Initialize params (dummy input for shape inference)
        # key1, key2 = jax.random.split(jax.random.PRNGKey(0))
        # ego_nav_params = ego_nav_encoder.init(key1, jnp.ones((N_ego_nav,)))
        # lidar_params = lidar_encoder.init(key2, jnp.ones((N_lidar,)))

        # Encode all observations
        # def encode(obs1, obs2):
        #     # obs1: (N, 19), obs2: (N, 240)
        #     ego_nav_encoded = jax.vmap(lambda x: ego_nav_encoder.apply(ego_nav_params, x))(obs1)
        #     lidar_encoded = jax.vmap(lambda x: lidar_encoder.apply(lidar_params, x))(obs2)
        #     return np.concatenate([np.array(ego_nav_encoded), np.array(lidar_encoded)], axis=1)

        # Encode all observations (batch)
        # with torch.no_grad():
        #     # ego_nav_encoded = ego_nav_encoder(ego_nav).numpy()
        #     lidar_encoded = lidar_encoder(lidar).numpy()
        #     # obs_encoded = np.concatenate([ego_nav_encoded, lidar_encoded], axis=1)
        #     obs_encoded = np.concatenate([ego_nav, lidar_encoded], axis=1)

        #     # ego_nav_next_encoded = ego_nav_encoder(ego_nav_next).numpy()
        #     lidar_next_encoded = lidar_encoder(lidar_next).numpy()
        #     # next_obs_encoded = np.concatenate([ego_nav_next_encoded, lidar_next_encoded], axis=1)
        #     next_obs_encoded = np.concatenate([ego_nav_next, lidar_next_encoded], axis=1)

        # dataset_dict["observations"] = encode(ego_nav, lidar)
        # dataset_dict["next_observations"] = encode(ego_nav_next, lidar_next)

        # dataset_dict["observations"] = obs_encoded
        # dataset_dict["next_observations"] = next_obs_encoded
        
        # dataset_dict["observations"] = obs_encoded
        # dataset_dict["next_observations"] = next_obs_encoded
        # ...rest of your code...
        dataset_dict["masks"] = 1.0 - dataset_dict['dones']
        del dataset_dict['dones']

        super().__init__(dataset_dict)