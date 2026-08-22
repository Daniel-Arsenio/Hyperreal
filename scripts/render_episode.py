import argparse

import jax
import flax.nnx as nnx

from substrate import Config
from substrate.env import env_reset, env_step
from substrate.games import TAG_GAME, TagConfig, FORAGING_GAME, ForagingConfig
from substrate.render import render_video, render_tag_frame, render_foraging_frame
from substrate.rl import ActorCritic
from substrate.rl.checkpoint import load_checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", choices=["tag", "foraging"], required=True)
    parser.add_argument("--num-agents", type=int, default=6)
    parser.add_argument("--num-pursuers", type=int, default=3, help="tag only")
    parser.add_argument("--capture-radius", type=float, default=0.3, help="tag only")
    parser.add_argument("--num-food", type=int, default=10, help="foraging only")
    parser.add_argument("--consume-radius", type=float, default=0.4, help="foraging only")
    parser.add_argument("--world-size", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=0, help="picks/reproduces one episode")
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--checkpoint", type=str, default=None, help="orbax checkpoint dir; omit for random actions")
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--centralized-critic", action="store_true")
    parser.add_argument("--stochastic", action="store_true", help="sample actions instead of using the policy mean")
    parser.add_argument("--out", type=str, default="episode.gif")
    parser.add_argument("--fps", type=int, default=20)
    args = parser.parse_args()

    if args.game == "tag":
        game, render_frame = TAG_GAME, render_tag_frame
        game_config = TagConfig(
            substrate=Config(num_agents=args.num_agents, world_size=args.world_size),
            num_pursuers=args.num_pursuers,
            capture_radius=args.capture_radius,
        )
    else:
        game, render_frame = FORAGING_GAME, render_foraging_frame
        game_config = ForagingConfig(
            substrate=Config(num_agents=args.num_agents, world_size=args.world_size),
            num_food=args.num_food,
            consume_radius=args.consume_radius,
        )

    model = None
    if args.checkpoint is not None:
        _, dummy_obs = env_reset(jax.random.PRNGKey(0), game, game_config)
        obs_dim = dummy_obs.shape[-1]
        critic_in_dim = args.num_agents * obs_dim if args.centralized_critic else obs_dim
        template = ActorCritic(obs_dim, 2, critic_in_dim, args.hidden, rngs=nnx.Rngs(0))
        graphdef, params = nnx.split(template)
        params = load_checkpoint(args.checkpoint, params)
        model = nnx.merge(graphdef, params)

    root_key = jax.random.PRNGKey(args.seed)
    env_key, policy_key = jax.random.split(root_key)
    state, obs = env_reset(env_key, game, game_config)
    frames = [state]
    done = False

    for _ in range(args.max_steps):
        policy_key, act_key, step_key = jax.random.split(policy_key, 3)
        if model is not None:
            pi = model.actor(obs)
            actions = pi.sample(seed=act_key) if args.stochastic else pi.mean()
        else:
            actions = jax.random.uniform(act_key, (args.num_agents, 2), minval=-1.0, maxval=1.0)

        state, obs, reward, agent_dones, done = env_step(step_key, state, actions, game, game_config)
        frames.append(state)
        if bool(done):
            break

    print(f"episode ran {len(frames)} steps, done={bool(done)}")
    render_video(frames, lambda ax, s: render_frame(ax, s, game_config), args.world_size, args.out, fps=args.fps)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
