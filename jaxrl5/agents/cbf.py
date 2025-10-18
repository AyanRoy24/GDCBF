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
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    target_value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    discount: float
    gamma: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparameter: float     
    extract_method: str = struct.field(pytree_node=False)
    action_dim: int = struct.field(pytree_node=False)
    N: int=struct.field(pytree_node=False)  #How many samples per observation
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
        # print("Actor hidden", actor_hidden_dims, action_dim, actions.shape)
        # exit()
        actor_def = GaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)        
        # time = jnp.zeros((1, 1))
        observations = jnp.expand_dims(observations, axis = 0)
        actions = jnp.expand_dims(actions, axis = 0)
        
        actor_params = actor_def.init(actor_key, observations)['params']
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
            actor_tau=actor_tau,
            cost_temperature=cost_temperature,
            cost_ub=cost_ub,
            extract_method=extract_method
        )



    def update_actor(agent, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        qs = agent.target_critic.apply_fn({"params": agent.target_critic.params}, 
                                          batch["observations"], 
                                          batch["actions"])
        q = qs.min(axis=0)
        v = agent.value.apply_fn({"params": agent.value.params}, batch["observations"])
        
        # Reward-based weights for safe states
        reward_adv = q - v
        reward_weights = jnp.exp(reward_adv * agent.reward_temperature)
        reward_weights = jnp.clip(reward_weights, 0, 100)
        weights = reward_weights
        
        def actor_loss_fn(actor_params: FrozenDict[str, Any]):
            dist = agent.score_model.apply_fn({"params": actor_params}, batch["observations"])
            log_probs = dist.log_prob(batch['actions'])
            actor_loss = -(log_probs * weights).mean() 
            return actor_loss, {"weights": weights.mean(), "log_probs": log_probs.mean()}
                
        grads, info = jax.grad(actor_loss_fn, has_aux=True)(agent.score_model.params)
        score_model = agent.score_model.apply_gradients(grads=grads)
        agent = agent.replace(score_model=score_model)
        return agent, info


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

    @jax.jit
    def eval_actions(self, observations: jnp.ndarray):
        rng = self.rng
        assert len(observations.shape) == 1
        # observations = jax.device_put(observations)
        observations_batch = jnp.expand_dims(observations, axis=0)# .repeat(self.N, axis=0)
        # we sample N action candidates and select the safest one (i.e., the lowest Q*_h value) as the final output
        
        # Sample actions from the learned Gaussian policy
        # rng, key = jax.random.split(rng, 2)
        dist = self.score_model.apply_fn(
            {"params": self.score_model.params}, 
            observations_batch,
            temperature=0
        )
        actions = dist.sample(seed=self.rng)
        
        # qs = compute_q(self.target_critic.apply_fn, self.target_critic.params, observations_batch, actions)
        # qcs = compute_safe_q(self.safe_target_critic.apply_fn, self.safe_target_critic.params, observations_batch, actions)
        # Evaluate Safety (Q*_h value) for Each Action using compute_safe_q hich computes the Q*_h value (cost critic value) for each action.
        # qcs = qcs - self.qc_thres
        # if self.extract_method == 'maxq':
        #     idx = jnp.argmax(qs)
        # elif self.extract_method == 'minqc':
        #     idx = jnp.argmin(qcs)
        #     # Select the Safest Action (Lowest Q*_h)
        # else:
        #     raise ValueError(f'Invalid extract_method: {self.extract_method}')
        
        action = actions[0]
        new_rng = rng
        return action.squeeze(), self.replace(rng=new_rng)


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
            target_qh = (1 - agent.gamma) * h_sa + agent.gamma * jnp.maximum(h_sa, next_vh)
        elif agent.mode == 2:  # Value-as-Barrier (Additive Bellman)
            target_qh = h_sa + agent.gamma * next_vh - (1 - agent.gamma) * agent.R
        elif agent.mode == 3:  # Reachability Constrained RL
            target_qh = jnp.maximum(h_sa, next_vh)
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
        # first_batch = jax.tree_util.tree_map(lambda x: x[:half_size], batch)
        # second_batch = jax.tree_util.tree_map(lambda x: x[half_size:], batch)
        
        # new_agent, _ = new_agent.update_actor(first_batch)
        new_agent, actor_info = new_agent.update_actor(batch)
        
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
