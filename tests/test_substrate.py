import functools

import chex
import jax
import jax.numpy as jnp

from substrate import Config, State, reset, step, observe


def make_state(config, pos, theta, alive=None, vel=None, energy=None):
    N = config.num_agents
    return State(
        pos=jnp.array(pos, dtype=jnp.float32),
        vel=jnp.zeros((N, 2)) if vel is None else jnp.array(vel, dtype=jnp.float32),
        theta=jnp.array(theta, dtype=jnp.float32),
        energy=jnp.ones((N,)) if energy is None else jnp.array(energy, dtype=jnp.float32),
        alive=jnp.ones((N,), dtype=bool) if alive is None else jnp.array(alive, dtype=bool),
    )


def test_reset_shapes_and_bounds():
    config = Config(num_agents=5, world_size=10.0)
    state = reset(jax.random.PRNGKey(0), config)
    chex.assert_shape(state.pos, (5, 2))
    chex.assert_shape(state.theta, (5,))
    assert jnp.all(state.alive)
    assert jnp.all((state.pos >= 0.0) & (state.pos <= config.world_size))


def test_step_dead_agent_frozen():
    config = Config(num_agents=2, world_size=20.0)
    state = make_state(config, pos=[[0.0, 0.0], [5.0, 5.0]], theta=[0.0, 0.0], alive=[True, False])
    actions = jnp.array([[1.0, 1.0], [1.0, 1.0]])

    new_state = step(state, actions, config)

    assert not jnp.allclose(new_state.pos[0], state.pos[0])
    assert jnp.allclose(new_state.pos[1], state.pos[1])
    assert jnp.allclose(new_state.theta[1], state.theta[1])


def test_step_respects_speed_cap():
    config = Config(num_agents=1, world_size=1000.0, max_speed=2.0, max_accel=4.0)
    state = make_state(config, pos=[[0.0, 0.0]], theta=[0.0])
    actions = jnp.array([[1.0, 0.0]])

    for _ in range(200):
        state = step(state, actions, config)

    speed = jnp.linalg.norm(state.vel, axis=-1)
    assert jnp.all(speed <= config.max_speed + 1e-5)
    assert jnp.allclose(speed, config.max_speed, atol=1e-3)


def test_step_is_jittable():
    config = Config(num_agents=4, world_size=10.0)
    state = reset(jax.random.PRNGKey(1), config)
    actions = jnp.zeros((4, 2))

    eager = step(state, actions, config)
    jitted = jax.jit(functools.partial(step, config=config))(state, actions)
    chex.assert_trees_all_close(eager.pos, jitted.pos)


def test_observe_head_on_hit_matches_analytic_distance():
    config = Config(num_agents=2, num_rays=4, world_size=20.0, agent_radius=0.1, lidar_range=3.0)
    state = make_state(config, pos=[[0.0, 0.0], [1.0, 0.0]], theta=[0.0, 0.0])

    ranges = observe(state, config)

    chex.assert_shape(ranges, (2, 4))
    assert jnp.isclose(ranges[0, 0], 1.0 - config.agent_radius, atol=1e-5)
    assert jnp.all(ranges[0, 1:] == config.lidar_range)


def test_observe_ignores_dead_agents():
    config = Config(num_agents=2, num_rays=4, world_size=20.0, agent_radius=0.1, lidar_range=3.0)
    state = make_state(config, pos=[[0.0, 0.0], [1.0, 0.0]], theta=[0.0, 0.0], alive=[True, False])

    ranges = observe(state, config)

    assert jnp.all(ranges[0] == config.lidar_range)


def test_observe_empty_scene_is_max_range():
    config = Config(num_agents=1, num_rays=8, lidar_range=3.0)
    state = make_state(config, pos=[[0.0, 0.0]], theta=[0.0])

    ranges = observe(state, config)

    assert jnp.all(ranges == config.lidar_range)


def test_vmap_over_env_batch():
    config = Config(num_agents=6, world_size=10.0)
    batch = 8
    keys = jax.random.split(jax.random.PRNGKey(2), batch)

    states = jax.vmap(functools.partial(reset, config=config))(keys)
    chex.assert_shape(states.pos, (batch, 6, 2))

    actions = jnp.zeros((batch, 6, 2))
    new_states = jax.vmap(functools.partial(step, config=config))(states, actions)
    chex.assert_shape(new_states.pos, (batch, 6, 2))

    ranges = jax.vmap(functools.partial(observe, config=config))(states)
    chex.assert_shape(ranges, (batch, 6, 16))
