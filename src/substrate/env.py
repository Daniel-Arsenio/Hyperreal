from typing import Any, Callable, NamedTuple, Optional, Tuple

import chex
import jax.numpy as jnp

from .dynamics import step as substrate_step, observe as substrate_observe


class Game(NamedTuple):
    reset_fn: Callable[[chex.PRNGKey, Any], Any]
    reward_fn: Callable[[Any, jnp.ndarray, Any, Any], jnp.ndarray]
    termination_fn: Callable[[Any, Any], Tuple[jnp.ndarray, jnp.ndarray]]
    dynamics_modifier: Optional[Callable[[chex.PRNGKey, Any, Any], Any]] = None
    observation_fn: Optional[Callable[[Any, Any], jnp.ndarray]] = None


def _observe(game: Game, state: Any, config: Any) -> jnp.ndarray:
    if game.observation_fn is not None:
        return game.observation_fn(state, config)
    return substrate_observe(state.substrate, config.substrate)


def env_reset(key: chex.PRNGKey, game: Game, config: Any):
    state = game.reset_fn(key, config)
    obs = _observe(game, state, config)
    return state, obs


def env_step(key: chex.PRNGKey, state: Any, actions: jnp.ndarray, game: Game, config: Any):
    substrate_state = substrate_step(state.substrate, actions, config.substrate)
    new_state = state.replace(substrate=substrate_state)
    if game.dynamics_modifier is not None:
        new_state = game.dynamics_modifier(key, new_state, config)

    obs = _observe(game, new_state, config)
    rewards = game.reward_fn(state, actions, new_state, config)
    agent_dones, done = game.termination_fn(new_state, config)
    return new_state, obs, rewards, agent_dones, done
