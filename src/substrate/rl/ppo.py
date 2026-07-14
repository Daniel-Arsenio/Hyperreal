from typing import Any

import jax
import jax.numpy as jnp
import flax.nnx as nnx


def compute_gae(
    rewards: jnp.ndarray,
    values: jnp.ndarray,
    dones: jnp.ndarray,
    next_value: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
):
    def scan_fn(carry, transition):
        gae, next_value = carry
        reward, value, done = transition
        delta = reward + gamma * next_value * (1.0 - done) - value
        gae = delta + gamma * gae_lambda * (1.0 - done) * gae
        return (gae, value), gae

    _, advantages = jax.lax.scan(
        scan_fn,
        (jnp.zeros_like(next_value), next_value),
        (rewards, values, dones),
        reverse=True,
    )
    returns = advantages + values
    return advantages, returns


def ppo_loss(
    params: Any,
    graphdef: nnx.GraphDef,
    obs: jnp.ndarray,
    critic_obs: jnp.ndarray,
    actions: jnp.ndarray,
    old_log_probs: jnp.ndarray,
    advantages: jnp.ndarray,
    returns: jnp.ndarray,
    old_values: jnp.ndarray,
    mask: jnp.ndarray,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
):
    """mask excludes agents that were already dead when the transition was collected
    (e.g. caught evaders in tag) from contributing to the loss; the episode itself
    keeps running for the other agents so these rows still exist in the batch.
    """
    model = nnx.merge(graphdef, params)
    mask_sum = jnp.sum(mask) + 1e-8

    pi = model.actor(obs)
    log_probs = pi.log_prob(actions)
    ratio = jnp.exp(log_probs - old_log_probs)

    adv_mean = jnp.sum(advantages * mask) / mask_sum
    adv_var = jnp.sum(((advantages - adv_mean) ** 2) * mask) / mask_sum
    norm_adv = (advantages - adv_mean) / (jnp.sqrt(adv_var) + 1e-8)

    pg_loss = -jnp.sum(
        jnp.minimum(ratio * norm_adv, jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * norm_adv) * mask
    ) / mask_sum

    value = model.critic(critic_obs)
    value_clipped = old_values + jnp.clip(value - old_values, -clip_eps, clip_eps)
    v_loss = 0.5 * jnp.sum(
        jnp.maximum((value - returns) ** 2, (value_clipped - returns) ** 2) * mask
    ) / mask_sum

    entropy = jnp.sum(pi.entropy() * mask) / mask_sum

    total_loss = pg_loss + vf_coef * v_loss - ent_coef * entropy
    metrics = {"pg_loss": pg_loss, "v_loss": v_loss, "entropy": entropy, "total_loss": total_loss}
    return total_loss, metrics
