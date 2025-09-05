import gymnasium as gym
import os
import pickle
from functools import partial
# from typing import Dict, Any, Sequence, Tuple
from typing import Sequence, Callable, Dict, Any, Tuple
import jax
import jax.numpy as jnp
import numpy as np
import flax.linen as nn
import optax
import flax
from flax.training.train_state import TrainState
from flax import struct
from jaxrl5.agents.agent import Agent

class CBFMLP(nn.Module):
    features: Sequence[int]
    activation: Callable = nn.relu

    @nn.compact
    def __call__(self, x):
        for feat in self.features[:-1]:
            x = self.activation(nn.Dense(feat)(x))
        x = nn.Dense(self.features[-1])(x)
        return x.squeeze(-1)

# def expectile_loss(pred, target, tau):
#     diff = pred - target
#     weight = jnp.where(diff > 0, tau, 1 - tau)
#     return weight * (diff ** 2)

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

class GDCBF(Agent):
    cbf: TrainState
    target_cbf: TrainState
    tau: float = struct.field(pytree_node=False)
    gamma: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space = None,  # Not used, but for compatibility
        **kwargs
        # hidden_dims: Sequence[int] = (256, 256),
        # tau: float = 0.7,
        # gamma: float = 0.99,
        # lr: float = 3e-4,
    ):
        rng = jax.random.PRNGKey(seed)
        obs_dim = observation_space.shape[0]
        hidden_dims = kwargs.get("hidden_dims", (256, 256))
        tau = kwargs.get("tau", 0.7)
        gamma = kwargs.get("gamma", 0.99)
        lr = kwargs.get("lr", 3e-4)

        model_def = CBFMLP(features=list(hidden_dims) + [1])
        params = model_def.init(rng, jnp.zeros((1, obs_dim)))['params']
        tx = optax.adam(lr)
        cbf = TrainState.create(apply_fn=model_def.apply, params=params, tx=tx)
        target_cbf = TrainState.create(apply_fn=model_def.apply, params=params, tx=optax.GradientTransformation(lambda _: None, lambda _: None))
        return cls(
            cbf=cbf,
            target_cbf=target_cbf,
            tau=tau,
            gamma=gamma,
            actor = None,
            rng = rng,
        )

    @jax.jit
    def update(self, batch: Dict[str, Any]):
        # Compute targets using current target network
        h_s = self.cbf.apply_fn({'params': self.cbf.params}, batch['observations'])
        h_sp = self.target_cbf.apply_fn({'params': self.target_cbf.params}, batch['next_observations'])
        target = (1 - self.gamma) * h_s + self.gamma * jnp.minimum(h_s, h_sp)
        
        value_loss = expectile_loss(h_s - target, self.tau).mean()
        v = h_s.mean()
        vc = h_sp.mean()
        vc_min = h_sp.min()
        vc_max = h_sp.max()
        weights = jnp.ones_like(h_s)

        def loss_fn(params):
            h_s_pred = self.cbf.apply_fn({'params': params}, batch['observations'])
            loss = safe_expectile_loss(h_s_pred - target, self.tau).mean()
            return loss

        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(self.cbf.params)
        cbf = self.cbf.apply_gradients(grads=grads)

        # Update target network
        target_params = optax.incremental_update(cbf.params, self.target_cbf.params, self.tau)
        target_cbf = self.target_cbf.replace(params=target_params)

        new_agent = self.replace(cbf=cbf, target_cbf=target_cbf)
        info = {
            # "cbf_loss": loss,
            # "weights": weights.mean(),
            "vc": vc,
            "vc_min": vc_min,
            "vc_max": vc_max,
            "value_loss": loss,
            "v": v
        }
        return new_agent, info

    def save(self, modeldir, save_time=None):
        if not os.path.exists(modeldir):
            os.makedirs(modeldir)
        filename = "cbf_params.pkl" if save_time is None else f"cbf_params_{save_time}.pkl"
        with open(os.path.join(modeldir, filename), "wb") as f:
            pickle.dump(self.cbf.params, f)

    def load(self, model_location):
        with open(model_location, "rb") as f:
            params = pickle.load(f)
        new_cbf = self.cbf.replace(params=params)
        return self.replace(cbf=new_cbf)

    def eval_actions(self, observations: jnp.ndarray) -> Tuple[jnp.ndarray, "GDCBF"]:
        # Expand dims if needed for batch
        if len(observations.shape) == 1:
            obs = jnp.expand_dims(observations, axis=0)
        else:
            obs = observations
        # Use target network for evaluation
        value = self.target_cbf.apply_fn({'params': self.target_cbf.params}, obs)
        # If you want to return the raw value, or an action, adapt here
        # For CBF, you might just return the value as a "score"
        # For compatibility, return as numpy and agent
        return jnp.asarray(value.squeeze()), self