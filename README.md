# substrate

A multi-agent reinforcement learning simulator written in JAX. The environment, the observation code, and the PPO training loop run on the GPU as one jit-compiled program, vectorised over parallel environment instances. I am building it as infrastructure for research on coordination and credit assignment in multi-agent systems.

<img width="600" height="600" alt="tag_random" src="https://github.com/user-attachments/assets/fb554a52-c101-429b-b9ca-0529f0beeb41" />

## Motivation

In most MARL codebases the environment steps on the CPU while the networks run on the GPU, and data crosses that boundary every step. For small environments the transfer dominates the step time. Writing the environment in JAX removes the boundary: reset, step, observation, and the PPO update compile into a single program under `jax.jit`, and `jax.vmap` batches it over environment instances. The design follows PureJaxRL and JaxMARL.

Throughput measured with `scripts/benchmark.py` on an RTX 5070 Ti: [N] agent-steps per second at [M] parallel environments. These are agent-steps rather than environment-steps; with k agents per environment the two differ by a factor of k.

[PLOT: steps per second against parallel environment count]

## Design

The code is split into a substrate and games. The substrate provides 2D agent kinematics, collision, and lidar sensing. A game is five pure functions (`reset_fn`, `reward_fn`, `termination_fn`, `dynamics_modifier`, `observation_fn`) closing over their own config and state. Cooperative, competitive, and mixed-motive settings share the same physics and differ only in these functions.

State is a `flax.struct.dataclass` holding position, velocity, heading, energy, and an alive mask, and `reset`, `step`, and `observe` are pure functions over it. The agent count is fixed per config, with the alive mask marking which slots are active, so all shapes are static under `jit`. Kinematics are a unicycle model (acceleration and turn-rate commands, clipped to speed and world bounds) computed in closed form each step. The default observation is a ring lidar: each agent casts a fixed number of rays and receives the distance to the nearest live agent along each. Games that need typed sensing supply their own observation function built on the shared `raycast` primitive.

## Contents

`src/substrate/` holds the config, state, and dynamics. Scaling has been verified to tens of thousands of parallel instances.

`src/substrate/games/` currently has two games and a stub:

- `tag.py`: pursuit and evasion. A fixed subset of agents are pursuers; catching an evader inside a capture radius removes it and ends its episode. The evader penalty is per-agent and the pursuer reward is shared across the team. Observations are a two-channel lidar (nearest ally, nearest opponent).
- `foraging.py`: cooperative. Agents consume food items, every consumption reward is shared, and the episode ends when the food is gone. Observations are the default lidar plus a channel for food.
- `warehouse.py`: empty stub, intended as a pickup-and-delivery game.

`src/substrate/rl/` is a PureJaxRL-style PPO implementation with rollout, GAE, and the update epochs inside one `lax.scan`. The actor-critic is written in `flax.nnx` with a Gaussian policy head (`distrax`), and `log_std` is clipped to [-5, 2]. Transitions collected from agents that were already dead are masked in the loss rather than dropped, since the episode continues for the remaining agents. `make_train` produces either IPPO or MAPPO depending on a `centralized_critic` flag, and `--seeds N` vmaps initialisation and updates over a seed axis. Checkpointing is through orbax and metrics go to CSV.

`scripts/` has the benchmark sweep, training entry points for both games, and `render_episode.py`, which rolls out a random or checkpointed policy and writes a GIF.

There are 21 tests: shape and dtype assertions with chex, jit-consistency checks, per-game reward and termination logic, GAE and PPO losses checked against hand-computed values, and end-to-end training smoke tests for IPPO and MAPPO on both games.


Developed under WSL2 on Ubuntu 24, which JAX requires for GPU support on Windows 11 with this card. Dependencies are managed with `uv`.
