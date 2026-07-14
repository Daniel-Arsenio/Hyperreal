import jax
import jax.numpy as jnp
import flax.nnx as nnx

from substrate import Config
from substrate.games import TAG_GAME, TagConfig, FORAGING_GAME, ForagingConfig
from substrate.rl import ActorCritic, TrainConfig, compute_gae, make_train
from substrate.rl.ppo import ppo_loss


def test_gae_matches_hand_computed_undiscounted_returns():
    rewards = jnp.array([1.0, 1.0, 1.0])
    values = jnp.zeros((3,))
    dones = jnp.zeros((3,))
    advantages, returns = compute_gae(rewards, values, dones, jnp.array(0.0), gamma=1.0, gae_lambda=1.0)

    assert jnp.allclose(advantages, jnp.array([3.0, 2.0, 1.0]))
    assert jnp.allclose(returns, advantages)


def test_gae_stops_bootstrap_across_episode_boundary():
    rewards = jnp.array([1.0, 1.0, 1.0])
    values = jnp.zeros((3,))
    dones = jnp.array([0.0, 1.0, 0.0])
    advantages, _ = compute_gae(rewards, values, dones, jnp.array(0.0), gamma=1.0, gae_lambda=1.0)

    assert jnp.allclose(advantages, jnp.array([2.0, 1.0, 1.0]))


def test_ppo_loss_mask_excludes_dead_agent_rows():
    model = ActorCritic(obs_dim=3, act_dim=2, critic_in_dim=3, hidden=8, rngs=nnx.Rngs(0))
    graphdef, params = nnx.split(model)
    live_model = nnx.merge(graphdef, params)

    obs = jax.random.normal(jax.random.PRNGKey(1), (5, 3))
    pi = live_model.actor(obs)
    actions = pi.sample(seed=jax.random.PRNGKey(2))
    log_prob = pi.log_prob(actions)
    values = live_model.critic(obs)
    returns = values + jax.random.normal(jax.random.PRNGKey(3), (5,))
    advantages = returns - values

    args_common = (0.2, 0.5, 0.01)

    loss_first_four, _ = ppo_loss(
        params, graphdef, obs[:4], obs[:4], actions[:4], log_prob[:4],
        advantages[:4], returns[:4], values[:4], jnp.ones((4,)), *args_common,
    )

    obs5 = obs.at[4].set(1e6)
    actions5 = actions.at[4].set(1e6)
    log_prob5 = log_prob.at[4].set(1e6)
    returns5 = returns.at[4].set(1e6)
    advantages5 = advantages.at[4].set(1e6)
    values5 = values.at[4].set(1e6)
    mask_drop_last = jnp.array([1.0, 1.0, 1.0, 1.0, 0.0])

    loss_masked, _ = ppo_loss(
        params, graphdef, obs5, obs5, actions5, log_prob5,
        advantages5, returns5, values5, mask_drop_last, *args_common,
    )

    assert jnp.isclose(loss_masked, loss_first_four, atol=1e-4)


def _assert_params_changed(before, after):
    leaves_before = jax.tree_util.tree_leaves(before)
    leaves_after = jax.tree_util.tree_leaves(after)
    changed = [not jnp.allclose(b, a) for b, a in zip(leaves_before, leaves_after)]
    assert any(changed)


def _assert_finite(metrics):
    for value in jax.tree_util.tree_leaves(metrics):
        assert jnp.all(jnp.isfinite(value))


def test_train_smoke_tag_ippo():
    game_config = TagConfig(substrate=Config(num_agents=4, world_size=6.0), num_pursuers=2, capture_radius=0.8)
    train_config = TrainConfig(num_envs=8, num_steps=16, num_updates=2, update_epochs=2, num_minibatches=2)
    init, _, train, _ = make_train(TAG_GAME, game_config, train_config)

    initial_params = init(jax.random.PRNGKey(0))[0]
    params, opt_state, metrics = jax.jit(train)(jax.random.PRNGKey(0))

    _assert_finite(metrics)
    _assert_params_changed(initial_params, params)


def test_train_smoke_tag_mappo():
    game_config = TagConfig(substrate=Config(num_agents=4, world_size=6.0), num_pursuers=2, capture_radius=0.8)
    train_config = TrainConfig(
        num_envs=8, num_steps=16, num_updates=2, update_epochs=2, num_minibatches=2, centralized_critic=True
    )
    _, _, train, _ = make_train(TAG_GAME, game_config, train_config)

    params, opt_state, metrics = jax.jit(train)(jax.random.PRNGKey(0))
    _assert_finite(metrics)


def test_train_smoke_foraging_ippo():
    game_config = ForagingConfig(substrate=Config(num_agents=3, world_size=6.0), num_food=6, consume_radius=0.5)
    train_config = TrainConfig(num_envs=8, num_steps=16, num_updates=2, update_epochs=2, num_minibatches=2)
    _, _, train, _ = make_train(FORAGING_GAME, game_config, train_config)

    params, opt_state, metrics = jax.jit(train)(jax.random.PRNGKey(0))
    _assert_finite(metrics)
