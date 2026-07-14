import functools
from dataclasses import dataclass
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import optax
import flax.nnx as nnx

from ..env import Game, env_reset, env_step
from .networks import ActorCritic
from .ppo import compute_gae, ppo_loss


@dataclass(frozen=True)
class TrainConfig:
    num_envs: int = 64
    num_steps: int = 128
    num_updates: int = 50
    update_epochs: int = 4
    num_minibatches: int = 4
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.0
    max_grad_norm: float = 0.5
    hidden: int = 64
    centralized_critic: bool = False


class Transition(NamedTuple):
    obs: jnp.ndarray
    critic_obs: jnp.ndarray
    actions: jnp.ndarray
    log_prob: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    alive: jnp.ndarray


def make_train(game: Game, config: Any, train_config: TrainConfig):
    num_agents = config.substrate.num_agents
    _, dummy_obs = env_reset(jax.random.PRNGKey(0), game, config)
    obs_dim = dummy_obs.shape[-1]
    act_dim = 2
    critic_in_dim = num_agents * obs_dim if train_config.centralized_critic else obs_dim
    batch_size = train_config.num_steps * train_config.num_envs * num_agents

    template_model = ActorCritic(obs_dim, act_dim, critic_in_dim, train_config.hidden, rngs=nnx.Rngs(0))
    graphdef, _ = nnx.split(template_model)

    tx = optax.chain(
        optax.clip_by_global_norm(train_config.max_grad_norm),
        optax.adam(train_config.lr),
    )

    def make_critic_obs(obs):
        if not train_config.centralized_critic:
            return obs
        num_envs = obs.shape[0]
        flat = obs.reshape(num_envs, num_agents * obs_dim)
        return jnp.broadcast_to(flat[:, None, :], (num_envs, num_agents, num_agents * obs_dim))

    def env_step_scan(carry, _):
        params, env_state, obs, rng = carry
        rng, act_rng, step_rng, reset_rng = jax.random.split(rng, 4)

        model = nnx.merge(graphdef, params)
        pi = model.actor(obs)
        actions = pi.sample(seed=act_rng)
        log_prob = pi.log_prob(actions)
        critic_obs = make_critic_obs(obs)
        value = model.critic(critic_obs)
        alive = env_state.substrate.alive

        step_rngs = jax.random.split(step_rng, train_config.num_envs)
        stepped_state, stepped_obs, reward, _agent_done, done = jax.vmap(
            functools.partial(env_step, game=game, config=config)
        )(step_rngs, env_state, actions)

        reset_rngs = jax.random.split(reset_rng, train_config.num_envs)
        reset_state, reset_obs = jax.vmap(functools.partial(env_reset, game=game, config=config))(reset_rngs)

        new_env_state = jax.tree_util.tree_map(
            lambda r, s: jnp.where(done.reshape((-1,) + (1,) * (s.ndim - 1)), r, s),
            reset_state,
            stepped_state,
        )
        new_obs = jnp.where(done[:, None, None], reset_obs, stepped_obs)

        transition = Transition(
            obs=obs,
            critic_obs=critic_obs,
            actions=actions,
            log_prob=log_prob,
            value=value,
            reward=reward,
            done=jnp.broadcast_to(done[:, None], reward.shape),
            alive=alive,
        )
        return (params, new_env_state, new_obs, rng), transition

    def update_step(runner_state, _):
        params, opt_state, env_state, obs, rng = runner_state

        (params, env_state, obs, rng), trajectory = jax.lax.scan(
            env_step_scan, (params, env_state, obs, rng), None, length=train_config.num_steps
        )

        model = nnx.merge(graphdef, params)
        last_value = model.critic(make_critic_obs(obs))

        advantages, returns = compute_gae(
            trajectory.reward,
            trajectory.value,
            trajectory.done,
            last_value,
            train_config.gamma,
            train_config.gae_lambda,
        )

        def flatten(x):
            return x.reshape((batch_size,) + x.shape[3:])

        batch = (
            flatten(trajectory.obs),
            flatten(trajectory.critic_obs),
            flatten(trajectory.actions),
            flatten(trajectory.log_prob),
            flatten(advantages),
            flatten(returns),
            flatten(trajectory.value),
            flatten(trajectory.alive.astype(jnp.float32)),
        )

        def update_epoch(carry, _):
            params, opt_state, rng = carry
            rng, perm_rng = jax.random.split(rng)
            perm = jax.random.permutation(perm_rng, batch_size)
            shuffled = jax.tree_util.tree_map(lambda x: x[perm], batch)
            minibatches = jax.tree_util.tree_map(
                lambda x: x.reshape((train_config.num_minibatches, -1) + x.shape[1:]), shuffled
            )

            def update_minibatch(carry, mb):
                params, opt_state = carry
                obs_mb, critic_obs_mb, actions_mb, log_prob_mb, adv_mb, ret_mb, val_mb, mask_mb = mb
                grad_fn = jax.value_and_grad(ppo_loss, has_aux=True)
                (_, metrics), grads = grad_fn(
                    params,
                    graphdef,
                    obs_mb,
                    critic_obs_mb,
                    actions_mb,
                    log_prob_mb,
                    adv_mb,
                    ret_mb,
                    val_mb,
                    mask_mb,
                    train_config.clip_eps,
                    train_config.vf_coef,
                    train_config.ent_coef,
                )
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                return (params, opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(update_minibatch, (params, opt_state), minibatches)
            return (params, opt_state, rng), metrics

        (params, opt_state, rng), epoch_metrics = jax.lax.scan(
            update_epoch, (params, opt_state, rng), None, length=train_config.update_epochs
        )

        metrics = jax.tree_util.tree_map(jnp.mean, epoch_metrics)
        metrics["mean_reward"] = jnp.mean(trajectory.reward)

        return (params, opt_state, env_state, obs, rng), metrics

    def init(rng):
        rng, net_rng, reset_rng = jax.random.split(rng, 3)
        model = ActorCritic(obs_dim, act_dim, critic_in_dim, train_config.hidden, rngs=nnx.Rngs(net_rng))
        _, params = nnx.split(model)
        opt_state = tx.init(params)

        reset_rngs = jax.random.split(reset_rng, train_config.num_envs)
        env_state, obs = jax.vmap(functools.partial(env_reset, game=game, config=config))(reset_rngs)

        return params, opt_state, env_state, obs, rng

    def update(runner_state):
        return update_step(runner_state, None)

    def train(rng):
        runner_state = init(rng)
        runner_state, metrics_history = jax.lax.scan(
            update_step, runner_state, None, length=train_config.num_updates
        )
        params, opt_state, env_state, obs, rng = runner_state
        return params, opt_state, metrics_history

    return init, update, train, graphdef
