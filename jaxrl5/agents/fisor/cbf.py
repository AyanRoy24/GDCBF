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
from jaxrl5.networks import MLP, Ensemble, StateActionValue, StateValue, DDPM, FourierFeatures, cosine_beta_schedule, MLPResNet, get_weight_decay_mask, vp_beta_schedule, GaussianPolicy
from jaxrl5.networks.diffusion import dpm_solver_sampler_1st, vp_sde_schedule

def phi_fisor(x, y, gamma):
    return (1 - gamma) * x + gamma * jnp.maximum(x, y)

def expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff > 0, expectile, (1 - expectile))
    return weight * (diff**2)

def safe_expectile_loss(diff, expectile=0.8):
    weight = jnp.where(diff < 0, expectile, (1 - expectile))
    return weight * (diff**2)

def mish(x):
    return x * jnp.tanh(nn.softplus(x))

class CBF(Agent):
    score_model: TrainState
    target_score_model: TrainState
    critic: TrainState
    target_critic: TrainState
    value: TrainState
    safe_critic: TrainState
    safe_target_critic: TrainState
    safe_value: TrainState
    discount: float
    tau: float
    actor_tau: float
    critic_hyperparam: float
    cost_critic_hyperparam: float
    critic_objective: str = struct.field(pytree_node=False)
    critic_type: str = struct.field(pytree_node=False)
    actor_objective: str = struct.field(pytree_node=False)
    sampling_method: str = struct.field(pytree_node=False)
    extract_method: str = struct.field(pytree_node=False)
    act_dim: int = struct.field(pytree_node=False)
    T: int = struct.field(pytree_node=False)
    N: int #How many samples per observation
    M: int = struct.field(pytree_node=False) #How many repeat last steps
    clip_sampler: bool = struct.field(pytree_node=False)
    ddpm_temperature: float
    cost_temperature: float
    reward_temperature: float
    qc_thres: float
    cost_ub: float
    betas: jnp.ndarray
    alphas: jnp.ndarray
    alpha_hats: jnp.ndarray
    cbf_gamma: float #= 0.99
    cbf_expectile_tau: float #= 0.02
    cbf_admissibility_coef: float #= 1e-3
    unsafe_penalty_alpha: float #= 1.0
    r_min: float #= -1.0
    mask_unsafe_for_actor: bool #= False
    max_weight: float
    qh_penalty_scale: float

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Box,
        actor_architecture: str = 'mlp',
        actor_lr: Union[float, optax.Schedule] = 3e-4,
        critic_lr: float = 3e-4,
        value_lr: float = 3e-4,
        critic_hidden_dims: Sequence[int] = (256, 256),
        actor_hidden_dims: Sequence[int] = (256, 256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        critic_hyperparam: float = 0.8,
        cost_critic_hyperparam: float = 0.8,
        ddpm_temperature: float = 1.0,
        num_qs: int = 2,
        actor_num_blocks: int = 2,
        actor_weight_decay: Optional[float] = None,
        actor_tau: float = 0.001,
        actor_dropout_rate: Optional[float] = None,
        actor_layer_norm: bool = False,
        value_layer_norm: bool = False,
        cost_temperature: float = 3.0,
        reward_temperature: float = 3.0,
        T: int = 5,
        time_dim: int = 64,
        N: int = 64,
        M: int = 0,
        clip_sampler: bool = True,
        actor_objective: str = 'bc',
        critic_objective: str = 'expectile',
        critic_type: str = 'hj',
        sampling_method: str = 'ddpm',
        beta_schedule: str = 'vp',
        decay_steps: Optional[int] = int(2e6),
        extract_method: bool = False,
        cost_limit: float = 10.,
        env_max_steps: int = 1000,
        cost_ub: float = 200,
        cbf_gamma: float = 0.99,
        cbf_expectile_tau: float = 0.2,
        cbf_admissibility_coef: float = 1e-3,
        # safe_reward_mode: str = "piecewise"
        unsafe_penalty_alpha: float = 1.0,
        r_min: float = -1.0,
        mask_unsafe_for_actor: bool = False,
        max_weight: float = 100.0,  
        qh_penalty_scale: float = 1.0
       
    ):

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, value_key, safe_critic_key, safe_value_key = jax.random.split(rng, 6)
        actions = action_space.sample()
        observations = observation_space.sample()
        action_dim = action_space.shape[0]

        qc_thres = cost_limit * (1 - discount**env_max_steps) / (
            1 - discount) / env_max_steps
        
        preprocess_time_cls = partial(FourierFeatures,
                                      output_size=time_dim,
                                      learnable=True)

        cond_model_cls = partial(MLP,
                                hidden_dims=(128, 128),
                                activations=mish,
                                activate_final=False)
        
        print("Actor activation:", actor_architecture)
        if decay_steps is not None:
            actor_lr = optax.cosine_decay_schedule(actor_lr, decay_steps)

        if actor_architecture == 'mlp':
            base_model_cls = partial(MLP,
                                    hidden_dims=tuple(list(actor_hidden_dims) + [action_dim]),
                                    activations=mish,
                                    use_layer_norm=actor_layer_norm,
                                    activate_final=False)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        elif actor_architecture == 'ln_resnet':

            base_model_cls = partial(MLPResNet,
                                     use_layer_norm=actor_layer_norm,
                                     num_blocks=actor_num_blocks,
                                     dropout_rate=actor_dropout_rate,
                                     out_dim=action_dim,
                                     activations=mish)
            
            actor_def = DDPM(time_preprocess_cls=preprocess_time_cls,
                             cond_encoder_cls=cond_model_cls,
                             reverse_encoder_cls=base_model_cls)

        elif actor_architecture == 'gaussian':
            actor_def = GaussianPolicy(hidden_dims=actor_hidden_dims, action_dim=action_dim)

        else:
            raise ValueError(f'Invalid actor architecture: {actor_architecture}')
        
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

        if critic_type == 'qc':
            critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
            critic_def = Ensemble(critic_cls, num=num_qs)

        safe_critic_params = critic_def.init(safe_critic_key, observations, actions)["params"]
        safe_critic_params = FrozenDict(safe_critic_params)
        safe_critic = TrainState.create(
            apply_fn=critic_def.apply, params=safe_critic_params, tx=critic_optimiser
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

        if critic_type == 'qc':
            value_def = StateValue(base_cls=value_base_cls)
            # value_def = Relu_StateValue(base_cls=value_base_cls)

        safe_value_params = value_def.init(safe_value_key, observations)["params"]
        safe_value_params = FrozenDict(safe_value_params)

        safe_value = TrainState.create(apply_fn=value_def.apply,
                                  params=safe_value_params,
                                  tx=value_optimiser)

        if beta_schedule == 'cosine':
            betas = jnp.array(cosine_beta_schedule(T))
        elif beta_schedule == 'linear':
            betas = jnp.linspace(1e-4, 2e-2, T)
        elif beta_schedule == 'vp':
            betas = jnp.array(vp_beta_schedule(T))
        else:
            raise ValueError(f'Invalid beta schedule: {beta_schedule}')
#  fix beta 3 or 8 and tau 0.3 or 0.7
        alphas = 1 - betas
        alpha_hat = jnp.array([jnp.prod(alphas[:i + 1]) for i in range(T)])

        return cls(
            actor=None, # Base class attribute
            score_model=score_model,
            target_score_model=target_score_model,
            critic=critic,
            target_critic=target_critic,
            value=value,
            safe_critic=safe_critic,
            safe_target_critic=safe_target_critic,
            safe_value=safe_value,
            tau=tau,
            discount=discount,
            rng=rng,
            betas=betas,
            alpha_hats=alpha_hat,
            act_dim=action_dim,
            T=T,
            N=N,
            M=M,
            alphas=alphas,
            ddpm_temperature=ddpm_temperature,
            actor_tau=actor_tau,
            actor_objective=actor_objective,
            sampling_method=sampling_method,
            critic_objective=critic_objective,
            critic_type=critic_type,
            critic_hyperparam=critic_hyperparam,
            cost_critic_hyperparam=cost_critic_hyperparam,
            clip_sampler=clip_sampler,
            cost_temperature=cost_temperature,
            reward_temperature=reward_temperature,
            extract_method=extract_method,
            qc_thres=qc_thres,
            cost_ub=cost_ub,
            cbf_gamma=cbf_gamma,
            cbf_expectile_tau=cbf_expectile_tau,
            cbf_admissibility_coef=cbf_admissibility_coef,
            unsafe_penalty_alpha=unsafe_penalty_alpha,
            r_min=r_min,
            mask_unsafe_for_actor=mask_unsafe_for_actor,
            max_weight=max_weight,
            qh_penalty_scale=qh_penalty_scale,           
        )

    def update_cbf(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        def joint_cbf_loss_fn(params, batch):
            h_s = batch["costs"]
            next_v = self.safe_value.apply_fn({"params": params['safe_value']}, batch["next_observations"])
            target_qh = jnp.maximum(h_s, next_v)
            qh_pred = self.safe_critic.apply_fn({"params": params['safe_critic']}, batch["observations"], batch["actions"])
            qh_loss = ((qh_pred - target_qh) ** 2).mean()

            vh_pred = self.safe_value.apply_fn({"params": params['safe_value']}, batch["observations"])
            min_qh = qh_pred.min(axis=0)
            vh_loss = expectile_loss(min_qh - vh_pred, self.cbf_expectile_tau).mean()

            cbf_loss = qh_loss + vh_loss
            return cbf_loss, {
                "cbf_loss": cbf_loss,
            }

        params = {'safe_critic': self.safe_critic.params, 'safe_value': self.safe_value.params}
        grads, info = jax.grad(joint_cbf_loss_fn, has_aux=True)(params, batch)

        safe_critic = self.safe_critic.apply_gradients(grads=grads['safe_critic'])
        safe_value = self.safe_value.apply_gradients(grads=grads['safe_value'])
        self = self.replace(safe_critic=safe_critic, safe_value=safe_value)

        safe_target_critic_params = optax.incremental_update(
            safe_critic.params, self.safe_target_critic.params, self.tau
        )
        safe_target_critic = self.safe_target_critic.replace(params=safe_target_critic_params)

        return self.replace(safe_critic=safe_critic, safe_target_critic=safe_target_critic, safe_value=safe_value), info


    def reward_piecewise_target(self, batch):
        qh = self.safe_critic.apply_fn({"params": self.safe_critic.params}, batch["observations"], batch["actions"]).max(axis=0)
        next_vr = self.value.apply_fn({"params": self.value.params}, batch["next_observations"])
        mask_unsafe = (qh > 0)
        mask_safe = (qh <= 0)
        target = (
            mask_safe * (batch["rewards"] + self.discount * batch["masks"] * next_vr)
            + mask_unsafe * (self.r_min / (1 - self.discount) - qh*self.qh_penalty_scale)
        )
        return target
    

    def reward_loss_piecewise_fn(self, critic_params, batch):
        target_q = self.reward_piecewise_target(batch)
        qs = self.critic.apply_fn({"params": critic_params}, batch["observations"], batch["actions"])
        loss = ((qs - target_q) ** 2).mean()
        return loss, {"q_r_loss": loss, "q_r": qs.mean()}

    def update_reward_critic(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        grads, info = jax.grad(self.reward_loss_piecewise_fn, has_aux=True)(self.critic.params, batch)
        critic = self.critic.apply_gradients(grads=grads)
        self = self.replace(critic=critic)
        target_critic_params = optax.incremental_update(
            critic.params, self.target_critic.params, self.tau
        )
        target_critic = self.target_critic.replace(params=target_critic_params)
        return self.replace(critic=critic, target_critic=target_critic), info

    def value_loss_fn(self, value_params, batch):
        qs = self.target_critic.apply_fn({"params": self.target_critic.params}, batch["observations"], batch["actions"]).min(axis=0)
        v = self.value.apply_fn({"params": value_params}, batch["observations"])
        loss = expectile_loss(qs - v, self.critic_hyperparam).mean()
        return loss, {"v_r_loss": loss, "v_r": v.mean()}

    def update_value(self, batch: DatasetDict) -> Tuple[Agent, Dict[str, float]]:
        grads, info = jax.grad(self.value_loss_fn, has_aux=True)(self.value.params, batch)
        value = self.value.apply_gradients(grads=grads)
        return self.replace(value=value), info


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
        new_agent, reward_info = new_agent.update_reward_critic(batch)
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
