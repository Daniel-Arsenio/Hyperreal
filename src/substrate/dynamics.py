import chex
import jax
import jax.numpy as jnp

from .config import Config
from .state import State


def reset(key: chex.PRNGKey, config: Config) -> State:
    key_pos, key_theta = jax.random.split(key)
    N = config.num_agents
    state = State(
        pos=jax.random.uniform(key_pos, (N, 2), minval=0.0, maxval=config.world_size),
        vel=jnp.zeros((N, 2)),
        theta=jax.random.uniform(key_theta, (N,), minval=-jnp.pi, maxval=jnp.pi),
        energy=jnp.ones((N,)),
        alive=jnp.ones((N,), dtype=bool),
    )
    chex.assert_shape(state.pos, (N, 2))
    return state


def step(state: State, actions: jnp.ndarray, config: Config) -> State:
    N = config.num_agents
    chex.assert_shape(actions, (N, 2))

    accel_cmd = jnp.clip(actions[:, 0], -1.0, 1.0)
    turn_cmd = jnp.clip(actions[:, 1], -1.0, 1.0) * config.max_angular_vel

    theta = state.theta + turn_cmd * config.dt
    speed = jnp.clip(
        jnp.linalg.norm(state.vel, axis=-1) + accel_cmd * config.max_accel * config.dt,
        0.0,
        config.max_speed,
    )
    vel = speed[:, None] * jnp.stack([jnp.cos(theta), jnp.sin(theta)], axis=-1)
    pos = jnp.clip(state.pos + vel * config.dt, 0.0, config.world_size)

    alive2 = state.alive[:, None]
    new_state = state.replace(
        pos=jnp.where(alive2, pos, state.pos),
        vel=jnp.where(alive2, vel, state.vel),
        theta=jnp.where(state.alive, theta, state.theta),
    )
    chex.assert_shape(new_state.pos, (N, 2))
    return new_state


def ray_directions(theta: jnp.ndarray, num_rays: int) -> jnp.ndarray:
    """theta: (N,) heading per agent. Returns (N, num_rays, 2) unit vectors for a
    360-degree ring of rays evenly spaced around each agent's heading."""
    ray_theta = theta[:, None] + jnp.linspace(0.0, 2 * jnp.pi, num_rays, endpoint=False)[None, :]
    return jnp.stack([jnp.cos(ray_theta), jnp.sin(ray_theta)], axis=-1)


def raycast(
    origin: jnp.ndarray,
    ray_dir: jnp.ndarray,
    target_pos: jnp.ndarray,
    valid: jnp.ndarray,
    radius: float,
    max_range: float,
) -> jnp.ndarray:
    """origin: (N,2) ray origins. ray_dir: (N,K,2) unit vectors. target_pos: (M,2)
    disc centers of radius `radius`. valid: (N,K,M) or broadcastable, e.g. (N,1,M) -
    which (agent, ray, target) combinations may register a hit (handles self-exclusion,
    liveness, team masks etc, decided by the caller). Returns (N,K) nearest hit distance
    per ray, or max_range where nothing valid is hit.
    """
    oc = (target_pos[None, :, :] - origin[:, None, :])[:, None, :, :]  # (N, 1, M, 2)
    d = ray_dir[:, :, None, :]  # (N, K, 1, 2)

    t_closest = jnp.sum(oc * d, axis=-1)  # (N, K, M)
    perp_sq = jnp.sum(oc * oc, axis=-1) - t_closest**2
    disc = radius**2 - perp_sq
    t_hit = t_closest - jnp.sqrt(jnp.maximum(disc, 0.0))

    hit = (disc >= 0.0) & (t_hit >= 0.0) & (t_hit <= max_range) & valid
    ranges = jnp.min(jnp.where(hit, t_hit, max_range), axis=-1)
    return jnp.clip(ranges, 0.0, max_range)


def observe(state: State, config: Config) -> jnp.ndarray:
    """Ring lidar: each agent casts num_rays evenly spaced around its heading and
    reads back distance to the nearest other live agent (agent_radius disc), or
    lidar_range if nothing is hit. Does not sense world boundary walls, and does not
    distinguish agent identity/team - games with that need should layer their own
    typed sensing on top via Game.observation_fn.
    """
    N, K = config.num_agents, config.num_rays
    ray_dir = ray_directions(state.theta, K)
    valid = ~jnp.eye(N, dtype=bool)[:, None, :] & state.alive[None, None, :]
    ranges = raycast(state.pos, ray_dir, state.pos, valid, config.agent_radius, config.lidar_range)
    chex.assert_shape(ranges, (N, K))
    return ranges
