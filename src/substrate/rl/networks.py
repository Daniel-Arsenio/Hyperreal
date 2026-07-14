import distrax
import jax.numpy as jnp
import flax.nnx as nnx


class ActorCritic(nnx.Module):
    def __init__(self, obs_dim: int, act_dim: int, critic_in_dim: int, hidden: int, rngs: nnx.Rngs):
        self.actor_l1 = nnx.Linear(obs_dim, hidden, rngs=rngs)
        self.actor_l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.actor_mean = nnx.Linear(hidden, act_dim, rngs=rngs)
        self.log_std = nnx.Param(jnp.zeros((act_dim,)))

        self.critic_l1 = nnx.Linear(critic_in_dim, hidden, rngs=rngs)
        self.critic_l2 = nnx.Linear(hidden, hidden, rngs=rngs)
        self.critic_out = nnx.Linear(hidden, 1, rngs=rngs)

    def actor(self, obs: jnp.ndarray) -> distrax.MultivariateNormalDiag:
        x = nnx.tanh(self.actor_l1(obs))
        x = nnx.tanh(self.actor_l2(x))
        mean = self.actor_mean(x)
        # unbounded log_std + an entropy bonus is a known PPO failure mode: nothing
        # stops std from growing without limit, so it can run away and drown out the
        # reward gradient. Clip to keep std in a sane range regardless of ent_coef.
        log_std = jnp.clip(self.log_std[...], -5.0, 2.0)
        return distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))

    def critic(self, critic_obs: jnp.ndarray) -> jnp.ndarray:
        x = nnx.tanh(self.critic_l1(critic_obs))
        x = nnx.tanh(self.critic_l2(x))
        return jnp.squeeze(self.critic_out(x), axis=-1)
