import distrax
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
from jaxrl5.networks import GaussianPolicy, get_weight_decay_mask
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)
def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

@partial(jax.jit, static_argnames=('critic_fn'))
def compute_q(critic_fn, critic_params, observations, actions):
    q_values = critic_fn({'params': critic_params}, observations, actions)
    q_values = q_values.min(axis=0)
    return q_values
@partial(jax.jit, static_argnames=('safe_critic_fn'))
def compute_safe_q(safe_critic_fn, safe_critic_params, observations, actions):
    safe_q_values = safe_critic_fn({'params': safe_critic_params}, observations, actions)
    safe_q_values = safe_q_values.max(axis=0)
    return safe_q_values

class CBF(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    target_value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    safe_target_value: TrainState
    discount: float
    gamma: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparameter: float     
    extract_method: str = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    R: float = struct.field(pytree_node=False)  # for 'add' mode
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    r_min: float 
    qh_penalty_scale: float
    mode:int = struct.field(pytree_node=False)  # 1: 'fisor', 2: 'add', 3: 'reach'

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        cbf_lr: float = 1e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        gamma: float = 0.5,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        num_qs: int = 2,
        actor_weight_decay: Optional[float] = None,
        value_layer_norm: bool = False,
        reward_temperature: float = 3.0,
        N: int = 64,
        R: float = 0.5,  # for 'add' mode
        decay_steps: Optional[int] = int(2e6),
        cost_critic_hyperparameter: float = 0.2,
        r_min: float = -1.0,
        qh_penalty_scale: float = 1.0,
        mode: int = 1,  # 1: 'fisor', 2: 'add', 3: 'reach'
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        actor_tau: float = 0.001,
        cost_temperature: float = 3.0,
        cost_ub: float = 200.,
        extract_method: str = 'maxq', # 'maxq' or 'minqc'
    ):
        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]
        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps                
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        actor_def = GaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)        
        time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        # if actor_architecture == 'gaussian':
        actor_params = actor_def.init(actor_key, observations,
                                            time)['params']
        # else:
        #     actor_params = actor_def.init(actor_key, observations, actions,
        #                                 time)['params']        
        actor_params = FrozenDict(actor_params) 
        score_model = TrainState.create(apply_fn=actor_def.apply,
                                        params=actor_params,
                                        tx=optax.adamw(learning_rate=actor_lr, 
                                                       weight_decay=actor_weight_decay if actor_weight_decay is not None else 0.0,
                                                       mask=get_weight_decay_mask,))        
        target_score_model = TrainState.create(apply_fn=actor_def.apply,
                                               params=actor_params,
                                               tx=optax.GradientTransformation(
                                                    lambda _: None, lambda _: None))

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
        # if critic_type == 'qc':
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
        # if critic_type == 'qc':
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
            gamma=gamma,
            rng=rng,
            action_dim=action_dim,
            N=N,
            R=R,
            critic_hyperparam=critic_hyperparam,
            reward_temperature=reward_temperature,
            cost_critic_hyperparameter=cost_critic_hyperparameter,
            r_min=r_min,
            qh_penalty_scale=qh_penalty_scale,
            mode=mode,
            qc_thres=qc_thres,
            target_score_model=target_score_model,
            actor_tau=actor_tau,
            cost_temperature=cost_temperature,
            cost_ub=cost_ub,
            extract_method=extract_method
        )


    def update_Vr(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"], batch["actions"]
        )
        q = jnp.min(qs, axis=0)
        def Vr_loss(value_params)-> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            loss = expectile_loss(q - v, agent.critic_hyperparam).mean()  # τ → 1
            return loss, {"v_r_loss": loss, "v_r": v.mean()}
        grads, info = jax.grad(Vr_loss, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=grads)
        agent = agent.replace(value=value)
        return agent, info


    def update_Qr(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qh = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params},
            batch["observations"], batch["actions"]
        )
        qh_min = jnp.min(qh, axis=0)  # Take min over ensemble
        
        next_vr = agent.value.apply_fn(
            {"params": agent.value.params}, batch["next_observations"]
        )        
        safe_mask = (qh_min <= 0)  # Q_h(s,a) ≤ 0
        unsafe_mask = (qh_min > 0)  # Q_h(s,a) > 0
        safe_target = batch["rewards"] + agent.discount * batch["masks"] * next_vr
        unsafe_target = agent.r_min / (1 - agent.discount) - qh_min       
        target_q = safe_mask * safe_target + unsafe_mask * unsafe_target
        
        def Qr_loss(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params},
                batch["observations"], batch["actions"]
            )
            # Equation 3: MSE loss
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qs.mean()}

        grads, info = jax.grad(Qr_loss, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=grads)
        agent = agent.replace(critic=critic)
        
        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        new_agent = agent.replace(critic=critic, target_critic=target_critic)
        
        return new_agent, info


    def update_Vh(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        # Equation 5: V_h(s) = max Q(s,a) ← with expectile regression
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"]
        )
        qc = qcs.max(axis=0)  # max over ensemble
        
        def Vh_loss(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vh = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            # loss = safe_expectile_loss(qc - vh, agent.cost_critic_hyperparameter).mean()
            loss = expectile_loss(qc - vh, agent.cost_critic_hyperparameter).mean()
            return loss, {"vh_loss": loss, "v_h": vh.mean()}
        
        grads, info = jax.grad(Vh_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        agent = agent.replace(safe_value=safe_value)
        return agent, info


    def update_Qh(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:       
        next_vh = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["next_observations"]
        )        
        h_sa = batch["costs"]
        if agent.mode == 1:  # FISOR
            target_qh = (1 - agent.gamma) * h_sa + agent.gamma * jnp.minimum(h_sa, next_vh)
        elif agent.mode == 2:  # Value-as-Barrier (Additive Bellman)
            target_qh = h_sa + agent.gamma * next_vh - (1 - agent.gamma) * agent.R
        elif agent.mode == 3:  # Reachability Constrained RL
            target_qh = jnp.minimum(h_sa, next_vh)
        else:
            raise ValueError(f"Unknown CBF mode: {agent.mode}")
        
        def Qh_loss(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qhs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"], batch["actions"]
            )
            qh_loss = ((qhs - target_qh) ** 2).mean()
            return qh_loss, {"qh_loss": qh_loss, "q_h": qhs.mean()}
        
        grads, info = jax.grad(Qh_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)
        agent = agent.replace(safe_critic=safe_critic)
        
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        new_agent = agent.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic)
        return new_agent, info


    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        qs = agent.target_critic.apply_fn({"params": agent.target_critic.params}, batch["observations"], batch["actions"])
        q = qs.min(axis=0)
        v = agent.value.apply_fn({"params": agent.value.params}, batch["observations"])
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"],
        )
        qc = qcs.max(axis=0)
        vc = agent.safe_value.apply_fn({"params": agent.safe_value.params}, batch["observations"])     
        # qc = qc - agent.qc_thres
        # vc = vc - agent.qc_thres
        
        # Convert to barrier function: h = -qh (so h > 0 is safe, h < 0 is unsafe)
        h = -qc
        vh = -vc
        # Penalize unsafe states (h < 0) more heavily
        # Safe states (h >= 0): use reward advantage
        # Unsafe states (h < 0): use barrier penalty
        eps = 0.0
        # Identify safe and unsafe transitions
        # safe_mask = (h >= eps)  # h >= 0 means safe
        # unsafe_mask = (h < eps)  # h < 0 means unsafe
        
        # Reward-based weights for safe states
        reward_adv = q - v
        reward_weights = jnp.exp(reward_adv * agent.reward_temperature)
        reward_weights = jnp.clip(reward_weights, 0, 100)
        
        # Barrier-based penalty for unsafe states
        # Penalize actions with h < 0 (unsafe)
        # barrier_adv = h - vh  # If h < vh, this action makes things worse
        # barrier_penalty = jnp.exp(-barrier_adv * agent.cost_temperature)  # Negative sign to penalize h < 0
        # barrier_penalty = jnp.clip(barrier_penalty, 0, agent.cost_ub)
        
        # Use Q_h directly (no negation)
        safe_mask = (qc <= 0)  # Q_h ≤ 0 means safe
        unsafe_mask = (qc > 0)  # Q_h > 0 means unsafe

        # Barrier advantage (more unsafe = higher Q_h = worse)
        barrier_adv = qc - vc  # If qc > vc, this action makes things worse
        barrier_penalty = jnp.exp(-barrier_adv * agent.cost_temperature)  # Penalize positive Q_h
        barrier_penalty = jnp.clip(barrier_penalty, 0, agent.cost_ub)
        # Combine weights: use reward weights for safe states, barrier penalty for unsafe
        weights = safe_mask * reward_weights + unsafe_mask * barrier_penalty

        # unsafe_condition = jax.nn.sigmoid((vc - eps) * 10.0)  # Smooth transition
        # safe_condition = jax.nn.sigmoid((-vc - eps) * 10.0) * jax.nn.sigmoid((-qc - eps) * 10.0)        
        # cost_exp_adv = jnp.exp((qc - vc) * agent.cost_temperature)
        # reward_exp_adv = jnp.exp((q-v) * agent.reward_temperature)

        # unsafe_condition = jnp.where( vc >  0. - eps, 1, 0)
        # safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc<=0. - eps, 1, 0)
        
        # cost_exp_adv = jnp.exp((vc-qc) * agent.cost_temperature)
        # reward_exp_adv = jnp.exp((q - v) * agent.reward_temperature)
        
        # unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub)
        # safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        # weights = unsafe_weights + safe_weights

        def actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.score_model.apply_fn({"params": actor_params}, batch["observations"])
            log_probs = dist.log_prob(batch['actions'])
            actor_loss = -(log_probs * weights).mean() 
            return actor_loss, {"weights": weights.mean(), "log_probs": log_probs.mean()}
                
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        agent = agent.replace(score_model=score_model)
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )        
        target_score_model = agent.target_score_model.replace(params=target_score_params)
        new_agent = agent.replace(score_model=score_model, target_score_model=target_score_model, rng=rng)
        return new_agent, info


    def eval_actions_old(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations_batch = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)      
        # Generate candidate actions uniformly from action space
        rng, key = jax.random.split(rng, 2)
        actions = jax.random.uniform(
            key, 
            shape=(self.N, self.action_dim), 
            minval=-1.0,  # Assuming normalized action space
            maxval=1.0
        )
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations_batch, actions)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations_batch, actions)
        # qcs = qcs - self.qc_thres
        if self.extract_method == 'maxq':
            idx = jnp.argmax(qs)
        elif self.extract_method == 'minqc':
            idx = jnp.argmin(qcs)
        else:
            raise ValueError(f'Invalid extract_method: {self.extract_method}')
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)

    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        observations = jax.device_put(observations)
        observations_batch = jnp.expand_dims(observations, axis=0).repeat(self.N, axis=0)
        # we sample N action candidates and select the safest one (i.e., the lowest Q*_h value) as the final output
        
        # Sample actions from the learned Gaussian policy
        rng, key = jax.random.split(rng, 2)
        dist = self.target_score_model.apply_fn(
            {"params": self.target_score_model.params}, 
            observations_batch
        )
        actions = dist.sample(seed=key)
        
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations_batch, actions)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations_batch, actions)
        # Evaluate Safety (Q*_h value) for Each Action using compute_safe_q hich computes the Q*_h value (cost critic value) for each action.
        # qcs = qcs - self.qc_thres
        if self.extract_method == 'maxq':
            idx = jnp.argmax(qs)
        elif self.extract_method == 'minqc':
            idx = jnp.argmin(qcs)
            # Select the Safest Action (Lowest Q*_h)
        else:
            raise ValueError(f'Invalid extract_method: {self.extract_method}')
        
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)

    def iql(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """
        IQL-like objective for cost critic learning as described in Section 3.1
        V_r*(s) = min max h(s_t) where h(s,a) = -c(s,a)
        Q_h*(s,a) = min max h(s_t) 
        """
        
        # Step 1: Update cost value function V_r* using IQL-like objective
        # Collect cost values (h values) for current states
        h_values = -batch["costs"]  # h(s,a) = -c(s,a)
        
        def cost_value_loss(cost_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v_cost = agent.safe_value.apply_fn({"params": cost_value_params}, batch["observations"])
            
            # IQL-like expectile regression towards max h values
            # For costs, we want to learn the worst-case (maximum) cost values
            cost_advantage = h_values - v_cost
            loss = safe_expectile_loss(cost_advantage, agent.cost_critic_hyperparameter).mean()
            
            return loss, {"cost_v_loss": loss, "cost_v": v_cost.mean()}
        
        grads, v_info = jax.grad(cost_value_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        
        # Step 2: Update cost Q-function Q_h* using IQL-like objective  
        next_v_cost = safe_value.apply_fn({"params": safe_value.params}, batch["next_observations"])
        
        # Bellman target for cost Q-function
        cost_target = h_values + agent.discount * batch["masks"] * next_v_cost
        
        def cost_q_loss(cost_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            q_costs = agent.safe_critic.apply_fn(
                {"params": cost_critic_params},
                batch["observations"], batch["actions"]
            )
            
            # MSE loss for Q-function
            loss = ((q_costs - cost_target) ** 2).mean()
            return loss, {"cost_q_loss": loss, "cost_q": q_costs.mean()}
        
        grads, q_info = jax.grad(cost_q_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=grads)
        
        # Update target networks
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        
        safe_target_value_params = optax.incremental_update(
            safe_value.params, agent.safe_target_value.params, agent.tau
        )
        safe_target_value = agent.safe_target_value.replace(params=safe_target_value_params)
        
        new_agent = agent.replace(
            safe_critic=safe_critic, 
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            safe_target_value=safe_target_value
        )
        
        return new_agent, {**v_info, **q_info}

    def cbf_good(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        
        def loss(cost_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v_pred = agent.safe_value.apply_fn({"params": cost_value_params}, batch["observations"])
            
            h = -batch["costs"]  # h(s,a) = -c(s,a)

            cost_diff = h - v_pred
            loss = expectile_loss(cost_diff, 0.95).mean()  # High expectile for safety

            return loss, {"loss": loss, "v_h": v_pred.mean()}

        grads, info = jax.grad(loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=grads)
        agent = agent.replace(safe_value=safe_value)
        
        return agent, info

    def cbf(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """
        Special cases for conservative barrier critics with h(s,a) = -cost(s,a).

        Implements:
        - FISOR          (mode='f'): Φ_f(x,y) = (1-γ) x + γ min{x, y}
        - Additive Bellman (mode='a'): Φ_a(x,y) = x + γ y - (1-γ) R
        - Reachability   (mode='r'): Φ_r(x,y) = min{x, y}
        """
        # 1) Update V_h(s) via expectile regression towards max_a Q_h(s,a)
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"]
        )
        qc_max = qcs.max(axis=0)  # max over ensemble

        def vh_loss_fn(safe_value_params):
            vh = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            loss = expectile_loss(qc_max - vh, agent.cost_critic_hyperparameter).mean()
            info = {
                "loss": loss,
                "v_h": vh.mean(),
                # "qc_max_mean": qc_max.mean(),
            }
            return loss, info

        vh_grads, vh_info = jax.grad(vh_loss_fn, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=vh_grads)

        # 2) Update Q_h(s,a) with Φ depending on mode
        next_vh = safe_value.apply_fn({"params": safe_value.params}, batch["next_observations"])
        next_vh = jax.lax.stop_gradient(next_vh)

        h_sa = batch["costs"]  # x = h(s,a)
        y = next_vh

        if agent.mode == 1:  # FISOR
            target_qh = (1.0 - agent.gamma) * h_sa + agent.gamma * jnp.minimum(h_sa, y)
        elif agent.mode == 2:  # Value-as-Barrier (Additive Bellman)
            target_qh = h_sa + agent.gamma * y - (1.0 - agent.gamma) * agent.R
        elif agent.mode == 3:  # Reachability Constrained RL
            target_qh = jnp.minimum(h_sa, y)
        else:
            raise ValueError(f"Unknown CBF mode: {agent.mode}")

        target_qh = jax.lax.stop_gradient(target_qh)

        def qh_loss_fn(safe_critic_params):
            qhs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"], batch["actions"]
            )
            loss = ((qhs - target_qh) ** 2).mean()
            info = {
                "loss": loss,
                "q_h": qhs.mean(),
                "q_h_target": target_qh.mean(),
            }
            return loss, info

        qh_grads, qh_info = jax.grad(qh_loss_fn, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=qh_grads)

        # 3) Soft-update targets
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)

        safe_target_value_params = optax.incremental_update(
            safe_value.params, agent.safe_target_value.params, agent.tau
        )
        safe_target_value = agent.safe_target_value.replace(params=safe_target_value_params)

        new_agent = agent.replace(
            safe_value=safe_value,
            safe_target_value=safe_target_value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
        )

        return new_agent, {**vh_info, **qh_info}

    
    # @jax.jit
    def update_o(self, batch: DatasetDict):
        new_agent = self
        batch_size = batch['observations'].shape[0]
        mini_batch_size = min(256, batch_size)
        
        def mini_slice(x):
            return x[:mini_batch_size]
        
        mini_batch = jax.tree_util.tree_map(mini_slice, batch)
        
        # IQL-like cost critic updates (Section 3.1)
        # new_agent, cost_critic_info = new_agent.iql(mini_batch)
        # new_agent, cost_critic_info = new_agent.cbf(mini_batch)
        # new_agent, cost_critic_info = new_agent.cbf_good(mini_batch)

        # Original CBF updates (Equations 5, 6, 7) - keep for comparison
        new_agent, vh_info = new_agent.update_Vh(mini_batch)
        new_agent, qh_info = new_agent.update_Qh(mini_batch)
        
        # Reward updates
        new_agent, vr_info = new_agent.update_Vr(mini_batch)
        new_agent, qr_info = new_agent.update_Qr(mini_batch)
        
        # Actor updates
        half_size = batch_size // 2
        first_batch = jax.tree_util.tree_map(lambda x: x[:half_size], batch)
        second_batch = jax.tree_util.tree_map(lambda x: x[half_size:], batch)
        
        new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(second_batch)
        
        info = {
            # **cost_critic_info,
            **vh_info,
            **qh_info,
            **vr_info,
            **qr_info,
            **actor_info
        }
        return new_agent, info


    def update_r(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Combined reward critic and value function update."""
        
        # Step 1: Update V_r (Equation 4)
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"], batch["actions"]
        )
        q = jnp.min(qs, axis=0)
        
        def Vr_loss(value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            v = agent.value.apply_fn({"params": value_params}, batch["observations"])
            loss = expectile_loss(q - v, agent.critic_hyperparam).mean()  # τ → 1
            return loss, {"v_r_loss": loss, "v_r": v.mean()}
        
        vr_grads, vr_info = jax.grad(Vr_loss, has_aux=True)(agent.value.params)
        value = agent.value.apply_gradients(grads=vr_grads)
        
        # Step 2: Update Q_r (Equation 3)
        qh = agent.safe_critic.apply_fn(
            {"params": agent.safe_critic.params},
            batch["observations"], batch["actions"]
        )
        qh_min = jnp.min(qh, axis=0)
        
        next_vr = value.apply_fn(
            {"params": value.params}, batch["next_observations"]
        )
        
        safe_mask = (qh_min <= 0)  # Q_h(s,a) ≤ 0
        unsafe_mask = (qh_min > 0)  # Q_h(s,a) > 0
        safe_target = batch["rewards"] + agent.discount * batch["masks"] * next_vr
        unsafe_target = agent.r_min / (1 - agent.discount) - qh_min
        target_q = safe_mask * safe_target + unsafe_mask * unsafe_target
        
        def Qr_loss(critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qs = agent.critic.apply_fn(
                {"params": critic_params},
                batch["observations"], batch["actions"]
            )
            loss = ((qs - target_q) ** 2).mean()
            return loss, {"q_r_loss": loss, "q_r": qs.mean()}
        
        qr_grads, qr_info = jax.grad(Qr_loss, has_aux=True)(agent.critic.params)
        critic = agent.critic.apply_gradients(grads=qr_grads)
        
        # Update target networks
        target_critic_params = optax.incremental_update(
            critic.params, agent.target_critic.params, agent.tau
        )
        target_critic = agent.target_critic.replace(params=target_critic_params)
        
        new_agent = agent.replace(
            value=value,
            critic=critic,
            target_critic=target_critic
        )
        
        return new_agent, {**vr_info, **qr_info}


    def update_h(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        """Combined safety critic and value function update."""
        
        # Step 1: Update V_h (Equation 5)
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"], batch["actions"]
        )
        qc = qcs.max(axis=0)  # max over ensemble
        
        def Vh_loss(safe_value_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            vh = agent.safe_value.apply_fn({"params": safe_value_params}, batch["observations"])
            loss = expectile_loss(qc - vh, agent.cost_critic_hyperparameter).mean()
            return loss, {"vh_loss": loss, "v_h": vh.mean()}
        
        vh_grads, vh_info = jax.grad(Vh_loss, has_aux=True)(agent.safe_value.params)
        safe_value = agent.safe_value.apply_gradients(grads=vh_grads)
        
        # Step 2: Update Q_h (Equation 6/7 depending on mode)
        next_vh = safe_value.apply_fn(
            {"params": safe_value.params}, batch["next_observations"]
        )
        
        h_sa = batch["costs"]
        
        if agent.mode == 1:  # FISOR
            target_qh = (1 - agent.gamma) * h_sa + agent.gamma * jnp.minimum(h_sa, next_vh)
        elif agent.mode == 2:  # Value-as-Barrier (Additive Bellman)
            target_qh = h_sa + agent.gamma * next_vh - (1 - agent.gamma) * agent.R
        elif agent.mode == 3:  # Reachability Constrained RL
            target_qh = jnp.minimum(h_sa, next_vh)
        else:
            raise ValueError(f"Unknown CBF mode: {agent.mode}")
        
        def Qh_loss(safe_critic_params) -> Tuple[jnp.ndarray, Dict[str, float]]:
            qhs = agent.safe_critic.apply_fn(
                {"params": safe_critic_params},
                batch["observations"], batch["actions"]
            )
            # qh_loss = ((qhs - target_qh) ** 2).mean()
            # TD
            qh_loss = jnp.abs(qhs - target_qh).mean()

            def huber_loss(diff, delta=1.0):
                abs_diff = jnp.abs(diff)
                quadratic = jnp.minimum(abs_diff, delta)
                linear = abs_diff - quadratic
                return 0.5 * quadratic ** 2 + delta * linear

            # qh_loss = huber_loss(qhs - target_qh).mean()
            return qh_loss, {"qh_loss": qh_loss, "q_h": qhs.mean()}
        
        qh_grads, qh_info = jax.grad(Qh_loss, has_aux=True)(agent.safe_critic.params)
        safe_critic = agent.safe_critic.apply_gradients(grads=qh_grads)
        
        # Update target networks
        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, agent.safe_target_critic.params, agent.tau
        )
        safe_target_critic = agent.safe_target_critic.replace(params=safe_target_critic_params)
        
        new_agent = agent.replace(
            safe_value=safe_value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic
        )
        
        return new_agent, {**vh_info, **qh_info}




    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = batch['observations'].shape[0]
        mini_batch_size = min(256, batch_size)
        
        def mini_slice(x):
            return x[:mini_batch_size]
        
        mini_batch = jax.tree_util.tree_map(mini_slice, batch)
        
        # Combined safety critic updates (V_h and Q_h)
        new_agent, h_info = new_agent.update_h(mini_batch)
        
        # Combined reward critic updates (V_r and Q_r)
        new_agent, r_info = new_agent.update_r(mini_batch)
        
        # Actor updates
        half_size = batch_size // 2
        first_batch = jax.tree_util.tree_map(lambda x: x[:half_size], batch)
        second_batch = jax.tree_util.tree_map(lambda x: x[half_size:], batch)
        
        new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(second_batch)
        
        info = {
            **h_info,
            **r_info,
            **actor_info
        }
        return new_agent, info


    def save(self, modeldir, save_time):
        file_name = 'model' + str(save_time) + '.pickle'
        state_dict = flax.serialization.to_state_dict(self)
        pickle.dump(state_dict, open(os.path.join(modeldir, file_name), 'wb'))

    def load(self, model_location):
        pkl_file = pickle.load(open(model_location, 'rb'))
        new_agent = flax.serialization.from_state_dict(target=self, state=pkl_file)
        return new_agent
