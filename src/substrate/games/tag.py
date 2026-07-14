from dataclasses import dataclass

import chex
import jax.numpy as jnp
from flax import struct

from ..config import Config
from ..dynamics import reset as substrate_reset, ray_directions, raycast
from ..env import Game
from ..state import State


@dataclass(frozen=True)
class TagConfig:
    substrate: Config
    num_pursuers: int
    capture_radius: float = 0.5


@struct.dataclass
class TagState:
    substrate: State
    is_pursuer: jnp.ndarray


def reset_fn(key: chex.PRNGKey, config: TagConfig) -> TagState:
    substrate_state = substrate_reset(key, config.substrate)
    is_pursuer = jnp.arange(config.substrate.num_agents) < config.num_pursuers
    return TagState(substrate=substrate_state, is_pursuer=is_pursuer)


def dynamics_modifier(key: chex.PRNGKey, state: TagState, config: TagConfig) -> TagState:
    pos = state.substrate.pos
    alive = state.substrate.alive
    dist = jnp.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)

    caught_by = (
        (dist < config.capture_radius)
        & state.is_pursuer[:, None]
        & ~state.is_pursuer[None, :]
        & alive[:, None]
        & alive[None, :]
    )
    evader_caught = jnp.any(caught_by, axis=0)

    new_alive = alive & ~evader_caught
    new_substrate = state.substrate.replace(alive=new_alive)
    return state.replace(substrate=new_substrate)


def reward_fn(prev_state: TagState, actions: jnp.ndarray, new_state: TagState, config: TagConfig) -> jnp.ndarray:
    newly_caught = prev_state.substrate.alive & ~new_state.substrate.alive
    num_caught = jnp.sum(newly_caught).astype(jnp.float32)

    is_pursuer = prev_state.is_pursuer
    pursuer_reward = jnp.where(is_pursuer, num_caught, 0.0)
    evader_reward = jnp.where(~is_pursuer & newly_caught, -1.0, 0.0)
    return pursuer_reward + evader_reward


def termination_fn(state: TagState, config: TagConfig):
    evader_alive = state.substrate.alive & ~state.is_pursuer
    agent_dones = ~state.substrate.alive
    done = ~jnp.any(evader_alive)
    return agent_dones, done


def observation_fn(state: TagState, config: TagConfig) -> jnp.ndarray:
    """Two-channel typed ring lidar: distance to nearest ally and nearest opponent,
    so a pursuer can tell a teammate from a target (plain distance alone can't).
    """
    substrate_cfg = config.substrate
    N = substrate_cfg.num_agents
    ray_dir = ray_directions(state.substrate.theta, substrate_cfg.num_rays)

    not_self = ~jnp.eye(N, dtype=bool)
    same_team = state.is_pursuer[:, None] == state.is_pursuer[None, :]
    alive = state.substrate.alive[None, None, :]

    ally_valid = (not_self & same_team)[:, None, :] & alive
    opp_valid = (not_self & ~same_team)[:, None, :] & alive

    pos = state.substrate.pos
    ally_ranges = raycast(pos, ray_dir, pos, ally_valid, substrate_cfg.agent_radius, substrate_cfg.lidar_range)
    opp_ranges = raycast(pos, ray_dir, pos, opp_valid, substrate_cfg.agent_radius, substrate_cfg.lidar_range)
    return jnp.concatenate([ally_ranges, opp_ranges], axis=-1)


TAG_GAME = Game(
    reset_fn=reset_fn,
    reward_fn=reward_fn,
    termination_fn=termination_fn,
    dynamics_modifier=dynamics_modifier,
    observation_fn=observation_fn,
)
