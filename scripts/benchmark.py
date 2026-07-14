import functools
import time

import jax
import jax.numpy as jnp

from substrate import Config, reset, step, observe

NUM_STEPS = 500
BATCH_SIZES = [256, 1024, 4096, 16384, 65536, 262144]


def rollout(key, config):
    reset_key, scan_key = jax.random.split(key)
    state = reset(reset_key, config)

    def body(state, key):
        obs = observe(state, config)
        noise = jax.random.uniform(key, (config.num_agents, 2), minval=-0.1, maxval=0.1)
        actions = jnp.clip(jnp.stack([obs.mean(-1), obs.std(-1)], axis=-1) + noise, -1.0, 1.0)
        state = step(state, actions, config)
        return state, None

    keys = jax.random.split(scan_key, NUM_STEPS)
    state, _ = jax.lax.scan(body, state, keys)
    return state


def bench(batch_size: int, config: Config) -> float:
    keys = jax.random.split(jax.random.PRNGKey(0), batch_size)
    batched_rollout = jax.jit(jax.vmap(functools.partial(rollout, config=config)))

    state = batched_rollout(keys)
    jax.block_until_ready(state)

    start = time.perf_counter()
    state = batched_rollout(keys)
    jax.block_until_ready(state)
    elapsed = time.perf_counter() - start

    return batch_size * NUM_STEPS / elapsed


def main():
    config = Config(num_agents=8)
    print(f"device: {jax.devices()[0]}")
    print(f"{'batch_size':>12} | {'steps/sec':>15}")
    for batch_size in BATCH_SIZES:
        try:
            rate = bench(batch_size, config)
        except Exception as exc:
            print(f"{batch_size:>12} | FAILED ({type(exc).__name__})")
            break
        print(f"{batch_size:>12} | {rate:>15,.0f}")


if __name__ == "__main__":
    main()
