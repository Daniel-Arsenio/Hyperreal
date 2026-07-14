import functools

import chex
import jax
import jax.numpy as jnp

from substrate import Config, State, env_reset, env_step
from substrate.games import TAG_GAME, TagConfig, TagState
from substrate.games import FORAGING_GAME, ForagingConfig, ForagingState
from substrate.games.tag import dynamics_modifier as tag_dynamics_modifier
from substrate.games.tag import reward_fn as tag_reward_fn
from substrate.games.tag import termination_fn as tag_termination_fn
from substrate.games.foraging import dynamics_modifier as forage_dynamics_modifier
from substrate.games.foraging import reward_fn as forage_reward_fn
from substrate.games.foraging import termination_fn as forage_termination_fn


def make_substrate_state(config, pos, theta):
    N = config.num_agents
    return State(
        pos=jnp.array(pos, dtype=jnp.float32),
        vel=jnp.zeros((N, 2)),
        theta=jnp.array(theta, dtype=jnp.float32),
        energy=jnp.ones((N,)),
        alive=jnp.ones((N,), dtype=bool),
    )


def test_tag_capture_kills_evader_and_rewards():
    config = Config(num_agents=2, world_size=20.0, agent_radius=0.1)
    tcfg = TagConfig(substrate=config, num_pursuers=1, capture_radius=0.5)
    substrate_state = make_substrate_state(config, pos=[[0.0, 0.0], [0.3, 0.0]], theta=[0.0, 0.0])
    state = TagState(substrate=substrate_state, is_pursuer=jnp.array([True, False]))
    actions = jnp.zeros((2, 2))

    new_state = tag_dynamics_modifier(jax.random.PRNGKey(0), state, tcfg)
    assert bool(new_state.substrate.alive[0])
    assert not bool(new_state.substrate.alive[1])

    rewards = tag_reward_fn(state, actions, new_state, tcfg)
    assert jnp.isclose(rewards[0], 1.0)
    assert jnp.isclose(rewards[1], -1.0)

    agent_dones, done = tag_termination_fn(new_state, tcfg)
    assert bool(agent_dones[1]) and not bool(agent_dones[0])
    assert bool(done)


def test_tag_no_capture_when_out_of_range():
    config = Config(num_agents=2, world_size=20.0, agent_radius=0.1)
    tcfg = TagConfig(substrate=config, num_pursuers=1, capture_radius=0.5)
    substrate_state = make_substrate_state(config, pos=[[0.0, 0.0], [10.0, 10.0]], theta=[0.0, 0.0])
    state = TagState(substrate=substrate_state, is_pursuer=jnp.array([True, False]))
    actions = jnp.zeros((2, 2))

    new_state = tag_dynamics_modifier(jax.random.PRNGKey(0), state, tcfg)
    assert jnp.all(new_state.substrate.alive)

    rewards = tag_reward_fn(state, actions, new_state, tcfg)
    assert jnp.all(rewards == 0.0)

    _, done = tag_termination_fn(new_state, tcfg)
    assert not bool(done)


def test_tag_env_step_wiring_and_vmap():
    config = Config(num_agents=4, world_size=10.0)
    tcfg = TagConfig(substrate=config, num_pursuers=2, capture_radius=0.5)
    batch = 8
    keys = jax.random.split(jax.random.PRNGKey(0), batch)

    states, obs = jax.vmap(functools.partial(env_reset, game=TAG_GAME, config=tcfg))(keys)
    chex.assert_shape(obs, (batch, 4, 2 * config.num_rays))

    actions = jnp.zeros((batch, 4, 2))
    new_states, obs, rewards, agent_dones, done = jax.vmap(
        functools.partial(env_step, game=TAG_GAME, config=tcfg)
    )(keys, states, actions)

    chex.assert_shape(rewards, (batch, 4))
    chex.assert_shape(agent_dones, (batch, 4))
    chex.assert_shape(done, (batch,))


def test_foraging_consumption_and_shared_reward():
    config = Config(num_agents=2, world_size=20.0)
    fcfg = ForagingConfig(substrate=config, num_food=2, consume_radius=0.3)
    substrate_state = make_substrate_state(config, pos=[[0.0, 0.0], [10.0, 10.0]], theta=[0.0, 0.0])
    state = ForagingState(
        substrate=substrate_state,
        food_pos=jnp.array([[0.1, 0.0], [15.0, 15.0]]),
        food_alive=jnp.array([True, True]),
    )
    actions = jnp.zeros((2, 2))

    new_state = forage_dynamics_modifier(jax.random.PRNGKey(0), state, fcfg)
    assert not bool(new_state.food_alive[0])
    assert bool(new_state.food_alive[1])

    rewards = forage_reward_fn(state, actions, new_state, fcfg)
    assert jnp.allclose(rewards, 1.0)

    agent_dones, done = forage_termination_fn(new_state, fcfg)
    assert not jnp.any(agent_dones)
    assert not bool(done)


def test_foraging_done_when_all_food_consumed():
    config = Config(num_agents=1, world_size=20.0)
    fcfg = ForagingConfig(substrate=config, num_food=1, consume_radius=0.3)
    substrate_state = make_substrate_state(config, pos=[[0.0, 0.0]], theta=[0.0])
    state = ForagingState(
        substrate=substrate_state, food_pos=jnp.array([[0.1, 0.0]]), food_alive=jnp.array([True])
    )

    new_state = forage_dynamics_modifier(jax.random.PRNGKey(0), state, fcfg)
    _, done = forage_termination_fn(new_state, fcfg)
    assert bool(done)


def test_foraging_env_step_wiring_and_vmap():
    config = Config(num_agents=3, world_size=10.0)
    fcfg = ForagingConfig(substrate=config, num_food=5, consume_radius=0.3)
    batch = 8
    keys = jax.random.split(jax.random.PRNGKey(1), batch)

    states, obs = jax.vmap(functools.partial(env_reset, game=FORAGING_GAME, config=fcfg))(keys)
    chex.assert_shape(obs, (batch, 3, 2 * config.num_rays))

    actions = jnp.zeros((batch, 3, 2))
    new_states, obs, rewards, agent_dones, done = jax.vmap(
        functools.partial(env_step, game=FORAGING_GAME, config=fcfg)
    )(keys, states, actions)

    chex.assert_shape(rewards, (batch, 3))
    chex.assert_shape(done, (batch,))
