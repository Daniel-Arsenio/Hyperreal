import jax.numpy as jnp
from flax import struct


@struct.dataclass
class State:
    pos: jnp.ndarray
    vel: jnp.ndarray
    theta: jnp.ndarray
    energy: jnp.ndarray
    alive: jnp.ndarray
