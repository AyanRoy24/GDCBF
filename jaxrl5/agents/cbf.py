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
    target_score_model: TrainState
    actor_tau: float
    cost_critic_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    clip_sampler: bool = struct.field(pytree_node=False)
    cost_temperature: float
    cost_ub: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
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
    gamma: float
    tau: float
    critic_hyperparam: float
    critic_type: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    R: float = struct.field(pytree_node=False)  # for 'add' mode
    reward_temperature: float
    cbf_expectile_tau: float 
    r_min: float 
    qh_penalty_scale: float
    mode:int = struct.field(pytree_node=False) # 1: 'fisor', 2: 'add', 3: 'reach'
    qc_thres: float

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
        gamma: float = 0.5,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        num_qs: int = 2,
        actor_weight_decay: Optional[float] = None,
        value_layer_norm: bool = False,
        reward_temperature: float = 3.0,
        N: int = 64,
        R: float = 0.5,  # for 'add' mode
        critic_type: str = 'qc',
        extract_method: bool = False,
        decay_steps: Optional[int] = int(2e6),
        cbf_expectile_tau: float = 0.2,
        r_min: float = -1.0,
        qh_penalty_scale: float = 1.0,
        mode: int = 1,
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_critic_hyperparam: float = 0.8,
        actor_tau: float = 0.001,
        cost_temperature: float = 3.0,
        clip_sampler: bool = True,
        critic_objective: str = 'expectile',
        cost_ub: float = 200.,
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
            cbf_expectile_tau=cbf_expectile_tau,
            r_min=r_min,
            qh_penalty_scale=qh_penalty_scale,
            mode=mode,
            critic_type=critic_type,
            extract_method=extract_method,
            qc_thres=qc_thres,
            target_score_model=target_score_model,
            actor_tau=actor_tau,
            cost_critic_hyperparam=cost_critic_hyperparam,
            critic_objective=critic_objective,
            clip_sampler=clip_sampler,
            cost_temperature=cost_temperature,
            cost_ub=cost_ub,
        )

    def update_cbf(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        def cbf_loss_fn(params, batch):
            h_s = batch["costs"]
            next_v = self.safe_target_value.apply_fn(
                {"params": self.safe_target_value.params},
                batch["next_observations"]
            )
            
            # Get current Q and V predictions
            qh_pred = self.safe_critic.apply_fn(
                {"params": params["safe_critic"]},
                batch["observations"], batch["actions"]
            )
            vh_pred = self.safe_value.apply_fn(
                {"params": params["safe_value"]}, batch["observations"]
            )
            
            # Special case handling based on formulation
            if self.mode == 1:  # FISOR
                # Φ_FISOR(x,y) = (1-γ)x + γ min{x,y}
                # α_FISOR(x) = γx
                target_qh = (1 - self.gamma) * h_s + self.gamma * jnp.minimum(h_s, next_v)
                alpha_regularizer = self.gamma * h_s
                
            elif self.mode == 2:  # Value-as-Barrier (Additive Bellman)
                # Φ_add(x,y) = x + γy - (1-γ)R
                # α_add(x) = 0
                target_qh = h_s + self.gamma * next_v - (1 - self.gamma) * self.R
                alpha_regularizer = jnp.zeros_like(h_s)
                
            elif self.mode == 3:  # Reachability Constrained RL
                # Φ_reach(x,y) = min{x,y}
                # α_reach(x) = x
                target_qh = jnp.minimum(h_s, next_v)
                alpha_regularizer = h_s
                
            else:
                raise ValueError(f"Unknown CBF mode: {self.mode}")
            
            # Q-function loss with gradient clipping for stability
            qh_loss = ((qh_pred - jax.lax.stop_gradient(target_qh)) ** 2).mean()
            
            # Value function loss using expectile regression
            min_qh = jnp.min(qh_pred, axis=0)
            vh_diff = min_qh - vh_pred
            vh_loss = expectile_loss(vh_diff, self.cbf_expectile_tau).mean()
            
            # Add regularization term to prevent sudden convergence
            regularization_loss = 0.001 * (alpha_regularizer ** 2).mean()
            
            # Smooth loss combination with adaptive weighting
            total_qh_loss = jnp.mean(qh_pred ** 2)
            total_vh_loss = jnp.mean(vh_pred ** 2)
            
            # Adaptive loss weighting to balance Q and V learning
            qh_weight = 1.0 / (1.0 + jnp.exp(-total_qh_loss))
            vh_weight = 1.0 / (1.0 + jnp.exp(-total_vh_loss))
            
            # Normalize weights
            total_weight = qh_weight + vh_weight
            qh_weight = qh_weight / total_weight
            vh_weight = vh_weight / total_weight
            
            # Combined loss with smoothing
            cbf_loss = qh_weight * qh_loss + vh_weight * vh_loss + regularization_loss
            
            # Fixed gradient penalty calculation
            # Use L2 norm of parameters instead of gradient of outputs
            qh_param_penalty = 0.001 * jnp.mean(jnp.square(qh_pred))
            vh_param_penalty = 0.001 * jnp.mean(jnp.square(vh_pred))
            
            total_loss = cbf_loss + qh_param_penalty + vh_param_penalty
            
            return total_loss, {
                "cbf_loss": total_loss,
                #"qh_loss": qh_loss,"vh_loss": vh_loss,"regularization_loss": regularization_loss,"qh_weight": qh_weight,"vh_weight": vh_weight,"target_qh_mean": target_qh.mean(),"qh_pred_mean": qh_pred.mean(),"vh_pred_mean": vh_pred.mean(),
            }
        
        params = {
            "safe_critic": self.safe_critic.params,
            "safe_value": self.safe_value.params,
        }
        
        grads, info = jax.grad(cbf_loss_fn, has_aux=True)(params, batch)
        
        # Clip gradients to prevent sudden convergence
        max_grad_norm = 1.0
        grads["safe_critic"] = optax.clip_by_global_norm(max_grad_norm).update(grads["safe_critic"], None)[0]
        grads["safe_value"] = optax.clip_by_global_norm(max_grad_norm).update(grads["safe_value"], None)[0]
        
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

    
    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        rng = agent.rng
        key, rng = jax.random.split(rng, 2)
        
        # Compute reward advantage
        qs = agent.target_critic.apply_fn(
            {"params": agent.target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        q = qs.min(axis=0)
        v = agent.value.apply_fn(
            {"params": agent.value.params}, batch["observations"]
        )
        reward_advantage = q - v
        
        # Compute safety advantage if needed
        qcs = agent.safe_target_critic.apply_fn(
            {"params": agent.safe_target_critic.params},
            batch["observations"],
            batch["actions"],
        )
        qc = qcs.max(axis=0)
        vc = agent.safe_value.apply_fn(
            {"params": agent.safe_value.params}, batch["observations"]
        )
        
        if agent.critic_type == "qc":
            qc = qc - agent.qc_thres
            vc = vc - agent.qc_thres
        
        # Determine weights based on safety conditions
        # if agent.actor_objective == "feasibility":
        eps = 0.
        unsafe_condition = jnp.where(vc > 0. - eps, 1, 0)
        safe_condition = jnp.where(vc <= 0. - eps, 1, 0) * jnp.where(qc <= 0. - eps, 1, 0)
        
        max_exp = 50.0  # To prevent overflow in exp        
        # cost_exp_adv = jnp.exp((vc - qc) * agent.cost_temperature)
        cost_exp_arg = jnp.clip((qc - vc) * agent.cost_temperature, -max_exp, max_exp)
        reward_exp_arg = jnp.clip(reward_advantage * agent.reward_temperature, -max_exp, max_exp)
        cost_exp_adv = jnp.exp(cost_exp_arg)
        reward_exp_adv = jnp.exp(reward_exp_arg)

        unsafe_weights = unsafe_condition * jnp.clip(cost_exp_adv, 0, agent.cost_ub)
        safe_weights = safe_condition * jnp.clip(reward_exp_adv, 0, 100)
        
        weights = unsafe_weights + safe_weights
        # elif agent.actor_objective == "bc":
        #     weights = jnp.ones(qc.shape)
        # else:
        #     # Default AWR weights
        #     weights = jnp.exp(reward_advantage * agent.reward_temperature)
        #     weights = jnp.clip(weights, 0, 100)
        
        # Define actor loss function using AWR
        def actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.score_model.apply_fn(
                {"params": actor_params}, batch["observations"]
            )
            log_probs = dist.log_prob(batch['actions'])
            return -(log_probs * weights).mean(), {"weights": weights.mean(), "log_probs": log_probs.mean()}
        
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        
        target_score_params = optax.incremental_update(
            score_model.params, agent.target_score_model.params, agent.actor_tau
        )
        
        target_score_model = agent.target_score_model.replace(params=target_score_params)
        
        new_agent = agent.replace(
            score_model=score_model, 
            target_score_model=target_score_model, 
            rng=rng
        )
        
        return new_agent, info
    
    def eval_actions(self, observations: jnp.ndarray):
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
        # Compute reward Q values
        qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations_batch, actions)
        # Compute safety values using the CBF (safety critic)
        qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations_batch, actions)
        # Apply threshold adjustment if using qc critic type
        if self.critic_type == "qc":
            qcs = qcs - self.qc_thres
        # Select action based on specified extraction method
        if self.extract_method == 'maxq':
            # Find action with maximum reward that satisfies safety constraints
            safe_mask = (qcs >= 0).astype(jnp.float32)
            # If there are safe actions, choose the one with highest reward
            if jnp.any(safe_mask):
                masked_qs = qs * safe_mask - 1e6 * (1 - safe_mask)  # Large penalty for unsafe actions
                idx = jnp.argmax(masked_qs)
            else:
                # If no safe actions, choose the one with least safety violation
                idx = jnp.argmax(qcs)
        elif self.extract_method == 'minqc':
            # Choose action that maximizes safety (minimizes cost)
            idx = jnp.argmax(qcs)
        else:
            raise ValueError(f'Invalid extract_method: {self.extract_method}')
        action = actions[idx]
        new_rng = rng
        return np.array(action.squeeze()), self.replace(rng=new_rng)

    
    @jax.jit
    def update(self, batch: DatasetDict):
        new_agent = self
        batch_size = batch['observations'].shape[0]
        mini_batch_size = min(256, batch_size)
        
        # Split batch for actor updates (like update_old)
        half_size = batch_size // 2
        
        def first_half(x):
            return x[:half_size]
        
        def second_half(x):
            return x[half_size:]
        
        def mini_slice(x):
            return x[:mini_batch_size]
        
        first_batch = jax.tree_util.tree_map(first_half, batch)
        second_batch = jax.tree_util.tree_map(second_half, batch)
        mini_batch = jax.tree_util.tree_map(mini_slice, batch)
        
        # Actor updates with split batches for better exploration
        new_agent, actor_info_1 = new_agent.update_actor(first_batch)
        new_agent, actor_info_2 = new_agent.update_actor(second_batch)
        
        # Combine actor infos, averaging metrics
        actor_info = {
            k: (actor_info_1.get(k, 0.0) + actor_info_2.get(k, 0.0)) / 2.0 
            for k in set(actor_info_1.keys()).union(actor_info_2.keys())
        }
        
        # CBF and reward updates use full batch for stability
        new_agent, cbf_info = new_agent.update_cbf(batch)
        new_agent, reward_info = new_agent.update_reward(batch)
        
        # Value updates use mini batch for efficiency
        new_agent, value_info = new_agent.update_value(mini_batch)
        
        # Combine all info dictionaries
        info = {
            # **actor_info,
            **cbf_info,
            **reward_info,
            **value_info,
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
