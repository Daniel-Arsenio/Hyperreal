import chex
import jax.numpy as jnp


def test_chex_harness_works():
    chex.assert_shape(jnp.zeros((4, 3)), (4, 3))
