import os
from functools import partial
from typing import Dict, Optional, Sequence, Tuple, Union, Any
import flax.linen as nn
from flax.core import FrozenDict
import gymnasium as gym 
import jax
import jax.numpy as jnp
import optax
import flax
import pickle
from flax.training.train_state import TrainState
from flax import struct
import numpy as np
from jaxrl5.agents.agent import Agent
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, cosine_beta_schedule, get_weight_decay_mask, vp_beta_schedule, GaussianPolicy

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

class CBF(Agent):
    score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    target_value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    safe_target_value: TrainState
    discount: float
    tau: float
    critic_hyperparam: float
    action_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    R: float = struct.field(pytree_node=False)  # for 'add' mode
    reward_temperature: float
    betas: jnp.ndarray
    cbf_expectile_tau: float 
    r_min: float 
    qh_penalty_scale: float
    mode:int = struct.field(pytree_node=False) # 1: 'fisor', 2: 'add', 3: 'reach'

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        mode_type: str = 'fisor', #['bc', 'fisor', 'diffusion']
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        cbf_lr: float = 1e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        num_qs: int = 2,
        actor_weight_decay: Optional[float] = None,
        value_layer_norm: bool = False,
        reward_temperature: float = 3.0,
        T: int = 5,
        R: float = 0.5,  # for 'add' mode
        critic_type: str = 'hj',
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
        cbf_expectile_tau: float = 0.2,
        r_min: float = -1.0,
        qh_penalty_scale: float = 1.0,
        mode: int = 1,
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]                
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        actor_def = GaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        if actor_architecture == 'gaussian':
            actor_params = actor_def.init(actor_key, observations,
                                            time)['params']
        else:
            actor_params = actor_def.init(actor_key, observations, actions,
                                        time)['params']        
        actor_params = FrozenDict(actor_params) 
        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr, 
                                                       weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                       mask=get_weight_decay_mask,))        
        critic_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True)
        critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
        critic_def = Ensemble(critic_cls, num=num_qs)
        critic_params = critic_def.init(critic_key, observations, actions)["params"]
        critic_params = FrozenDict(critic_params)
        critic_optimiser = optax.adam(learning_rate=critic_lr)
        critic = TrainState.create(
            apply_fn=critic_def.apply, params=critic_params, tx=critic_optimiser
        )
        target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        if mode_type == 'fisor':
            mode = 1
        elif mode_type == 'add':
            mode = 2
        elif mode_type == 'reach':
            mode = 3
        if critic_type == 'qc':
            critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
            critic_def = Ensemble(critic_cls, num=num_qs)
        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic_params = FrozenDict(safe_critic_params)
        safe_critic_optimiser = optax.adam(learning_rate=cbf_lr)
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=safe_critic_optimiser
        )
        safe_target_critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=safe_critic_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        value_base_cls = partial(MLP, hidden_dims=critic_hidden_dims, activate_final=True, use_layer_norm=value_layer_norm)
        value_def = StateValue(base_cls=value_base_cls)
        value_params = value_def.init(value_key, observations)["params"]
        value_params = FrozenDict(value_params)
        value_optimiser = optax.adam(learning_rate=value_lr)
        value = TrainState.create(apply_fn=value_def.apply,
                                  params=value_params,
                                  tx=value_optimiser)
        target_value = TrainState.create(
            apply_fn=value_def.apply,
            params=value_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        if critic_type == 'qc':
            value_def = StateValue(base_cls=value_base_cls)            
        safe_value_params = value_def.init(safe_value_key, observations)["params"]
        safe_value_params = FrozenDict(safe_value_params)
        safe_value = TrainState.create(apply_fn=value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)
        safe_target_value = TrainState.create(
            apply_fn=value_def.apply,
            params=safe_value_params,
            tx=optax.GradientTransformation(lambda _: None, lambda _: None),
        )
        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')
        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            target_value=target_value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            safe_target_value=safe_target_value,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            action_dim=action_dim,
            T=T,
            R=R,
            critic_hyperparam=critic_hyperparam,
            reward_temperature=reward_temperature,
            cbf_expectile_tau=cbf_expectile_tau,
            r_min=r_min,
            qh_penalty_scale=qh_penalty_scale,
            mode=mode,
        )

    def update_cbf(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        def cbf_loss_fn(params, batch):
            h_s = batch["costs"]
            next_v = self.safe_target_value.apply_fn(
                {"params": self.safe_target_value.params},
                batch["next_observations"]
            )
            if self.mode == 1:
                target_qh = (1 - self.discount) * h_s + self.discount * jnp.minimum(h_s, next_v)
            elif self.mode == 2:
                target_qh = h_s + self.discount * next_v - (1 - self.discount) * self.R
            elif self.mode == 3:
                target_qh = jnp.minimum(h_s, next_v)
            else:
                raise ValueError(f"Unknown CBF mode: {self.mode}")
            qh_pred = self.safe_critic.apply_fn(
                {"params": params["safe_critic"]},
                batch["observations"], batch["actions"]
            )
            qh_loss = ((qh_pred - target_qh) ** 2).mean()
            vh_pred = self.safe_value.apply_fn(
                {"params": params["safe_value"]}, batch["observations"]
            )
            min_qh = jnp.min(qh_pred, axis=0)
            vh_loss = expectile_loss(min_qh - vh_pred, self.cbf_expectile_tau).mean()
            cbf_loss = qh_loss + vh_loss
            return cbf_loss, {"cbf_loss": cbf_loss}
        params = {
            "safe_critic": self.safe_critic.params,
            "safe_value": self.safe_value.params,
        }
        grads, info = jax.grad(cbf_loss_fn, has_aux=True)(params, batch)
        safe_critic = self.safe_critic.apply_gradients(grads=grads["safe_critic"])
        safe_value = self.safe_value.apply_gradients(grads=grads["safe_value"])
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, self.safe_target_critic.params, self.tau
        )
        safe_target_value_params = optax.incremental_update(
            safe_value.params, self.safe_target_value.params, self.tau
        )
        return self.replace(
            safe_critic=safe_critic,
            safe_target_critic=self.safe_target_critic.replace(params=safe_target_critic_params),
            safe_value=safe_value,
            safe_target_value=self.safe_target_value.replace(params=safe_target_value_params),
        ), info  
    
    def update_reward(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        def reward_loss_piecewise_fn(critic_params, batch):
            def reward_piecewise_target(batch):
                qh = self.safe_critic.apply_fn(
                    {"params": self.safe_critic.params},
                    batch["observations"], batch["actions"]
                )
                qh_star = jnp.min(qh, axis=0)
                next_vr = self.value.apply_fn(
                    {"params": self.value.params}, batch["next_observations"]
                )
                mask_safe = (qh_star <= 0)
                mask_unsafe = (qh_star > 0)
                target = (
                    mask_safe * (batch["rewards"] + self.discount * batch["masks"] * next_vr)
                    + mask_unsafe * (self.r_min / (1 - self.discount) - qh_star * self.qh_penalty_scale)
                )
                return target        
            target_q = reward_piecewise_target(batch)
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"], batch["actions"]
            )
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qs.mean()}
        grads, info = jax.grad(reward_loss_piecewise_fn, has_aux=True)(self.critic.params, batch)
        critic = self.critic.apply_gradients(grads=grads)
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        return self.replace(
            critic=critic,
            target_critic=self.target_critic.replace(params=target_critic_params)
        ), info

    def update_value(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        def value_loss_fn(value_params, batch):
            qs = self.target_critic.apply_fn(
                {"params": self.target_critic.params},
                batch["observations"], batch["actions"]
            )
            q_min = jnp.min(qs, axis=0)
            v = self.value.apply_fn({"params": value_params}, batch["observations"])
            loss = expectile_loss(q_min - v, self.critic_hyperparam).mean()  # τ → 1
            return loss, {"v_r_loss": loss, "v_r": v.mean()}
        grads, info = jax.grad(value_loss_fn, has_aux=True)(self.value.params, batch)
        value = self.value.apply_gradients(grads=grads)
        target_value_params = optax.incremental_update(
            value.params, self.target_value.params, self.tau
        )
        return self.replace(
            value=value,
            target_value=self.target_value.replace(params=target_value_params)
        ), info

    def update_actor_awr(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)
        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        def actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.score_model.apply_fn(
                {"params": actor_params}, batch["observations"]
            )
            log_probs = dist.log_prob(batch['actions'])
            weights = jnp.exp((q - v) * agent.reward_temperature)
            weights = jnp.clip(weights, 0, 100)
            return -(log_probs * weights).mean(), {"weights": weights.mean(), "log_probs": log_probs.mean()}
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        agent = agent.replace(score_model=score_model)
        return agent, info
    
    @jax.jit
    def eval_actions_jit(self, observations: jnp.ndarray):
        observations = jnp.expand_dims(observations, axis = 0)
        dist = self.score_model.apply_fn(
            {"params": self.score_model.params}, observations
        )
        actions = dist.mean().squeeze()
        return actions
    def eval_actions(self, observations: jnp.ndarray, model_cls="gdcbf"):
        observations = jax.device_put(observations)
        return self.eval_actions_jit(observations), self

    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self    
        new_agent, actor_info = new_agent.update_actor_awr(batch)
        new_agent, cbf_info = new_agent.update_cbf(batch)
        new_agent, reward_info = new_agent.update_reward(batch)
        new_agent, value_info = new_agent.update_value(batch)
        return new_agent, {**value_info, **cbf_info, **reward_info}      
    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))
    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
