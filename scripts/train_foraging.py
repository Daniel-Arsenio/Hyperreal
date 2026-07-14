import argparse
import time

import jax
import jax.numpy as jnp

from substrate import Config
from substrate.games import FORAGING_GAME, ForagingConfig
from substrate.rl import TrainConfig, make_train
from substrate.rl.checkpoint import save_checkpoint
from substrate.rl.logging import CsvLogger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--num-food", type=int, default=12)
    parser.add_argument("--world-size", type=float, default=8.0)
    parser.add_argument("--consume-radius", type=float, default=0.4)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--total-updates", type=int, default=200)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument("--centralized-critic", action="store_true")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, default="runs/foraging")
    args = parser.parse_args()

    game_config = ForagingConfig(
        substrate=Config(num_agents=args.num_agents, world_size=args.world_size),
        num_food=args.num_food,
        consume_radius=args.consume_radius,
    )
    train_config = TrainConfig(
        num_envs=args.num_envs, num_steps=args.num_steps, centralized_critic=args.centralized_critic
    )
    init, update, _, _ = make_train(FORAGING_GAME, game_config, train_config)

    logger = CsvLogger(f"{args.out_dir}/metrics.csv")

    if args.seeds > 1:
        rngs = jax.random.split(jax.random.PRNGKey(args.seed), args.seeds)
        runner_state = jax.jit(jax.vmap(init))(rngs)
        jitted_update = jax.jit(jax.vmap(update))
    else:
        runner_state = jax.jit(init)(jax.random.PRNGKey(args.seed))
        jitted_update = jax.jit(update)

    start = time.perf_counter()
    for i in range(1, args.total_updates + 1):
        runner_state, metrics = jitted_update(runner_state)
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        logger.log(i, metrics)

        if i % args.checkpoint_every == 0 or i == args.total_updates:
            save_checkpoint(f"{args.out_dir}/checkpoints/step_{i}", runner_state[0])
            print(
                f"update {i}/{args.total_updates}  reward={float(metrics['mean_reward']):.4f}  "
                f"elapsed={time.perf_counter() - start:.1f}s"
            )

    logger.close()


if __name__ == "__main__":
    main()
