from dataclasses import dataclass

import chex
import jax
import jax.numpy as jnp
from flax import struct

from ..config import Config
from ..dynamics import reset as substrate_reset, observe as substrate_observe, ray_directions, raycast
from ..env import Game
from ..state import State


@dataclass(frozen=True)
class ForagingConfig:
    substrate: Config
    num_food: int
    consume_radius: float = 0.3


@struct.dataclass
class ForagingState:
    substrate: State
    food_pos: jnp.ndarray
    food_alive: jnp.ndarray


def reset_fn(key: chex.PRNGKey, config: ForagingConfig) -> ForagingState:
    agent_key, food_key = jax.random.split(key)
    substrate_state = substrate_reset(agent_key, config.substrate)
    food_pos = jax.random.uniform(
        food_key, (config.num_food, 2), minval=0.0, maxval=config.substrate.world_size
    )
    food_alive = jnp.ones((config.num_food,), dtype=bool)
    return ForagingState(substrate=substrate_state, food_pos=food_pos, food_alive=food_alive)


def dynamics_modifier(key: chex.PRNGKey, state: ForagingState, config: ForagingConfig) -> ForagingState:
    dist = jnp.linalg.norm(state.substrate.pos[:, None, :] - state.food_pos[None, :, :], axis=-1)
    in_range = (dist < config.consume_radius) & state.substrate.alive[:, None] & state.food_alive[None, :]
    consumed = jnp.any(in_range, axis=0)

    new_food_alive = state.food_alive & ~consumed
    return state.replace(food_alive=new_food_alive)


def reward_fn(
    prev_state: ForagingState, actions: jnp.ndarray, new_state: ForagingState, config: ForagingConfig
) -> jnp.ndarray:
    num_consumed = jnp.sum(prev_state.food_alive & ~new_state.food_alive).astype(jnp.float32)
    return jnp.full((config.substrate.num_agents,), num_consumed)


def termination_fn(state: ForagingState, config: ForagingConfig):
    agent_dones = ~state.substrate.alive
    done = ~jnp.any(state.food_alive)
    return agent_dones, done


def observation_fn(state: ForagingState, config: ForagingConfig) -> jnp.ndarray:
    """Base agent-sensing lidar plus a second channel sensing live food, since food
    isn't part of the substrate and would otherwise be completely unobservable.
    """
    substrate_cfg = config.substrate
    agent_ranges = substrate_observe(state.substrate, substrate_cfg)

    ray_dir = ray_directions(state.substrate.theta, substrate_cfg.num_rays)
    food_valid = state.food_alive[None, None, :]
    food_ranges = raycast(
        state.substrate.pos, ray_dir, state.food_pos, food_valid, config.consume_radius, substrate_cfg.lidar_range
    )
    return jnp.concatenate([agent_ranges, food_ranges], axis=-1)


FORAGING_GAME = Game(
    reset_fn=reset_fn,
    reward_fn=reward_fn,
    termination_fn=termination_fn,
    dynamics_modifier=dynamics_modifier,
    observation_fn=observation_fn,
)
