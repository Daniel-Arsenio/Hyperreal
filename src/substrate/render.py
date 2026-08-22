import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from .games.tag import TagState
from .games.foraging import ForagingState


def draw_agents(ax, pos, theta, colors, radius, alive=None):
    pos = np.asarray(pos)
    theta = np.asarray(theta)
    alive = np.ones(pos.shape[0], dtype=bool) if alive is None else np.asarray(alive)

    for i in range(pos.shape[0]):
        if not alive[i]:
            continue
        ax.add_patch(plt.Circle(pos[i], radius, color=colors[i], alpha=0.85, zorder=2))
        dx, dy = radius * 1.6 * np.cos(theta[i]), radius * 1.6 * np.sin(theta[i])
        ax.plot(
            [pos[i, 0], pos[i, 0] + dx], [pos[i, 1], pos[i, 1] + dy],
            color="black", linewidth=1, zorder=3,
        )


def render_tag_frame(ax, state: TagState, config):
    is_pursuer = np.asarray(state.is_pursuer)
    colors = np.where(is_pursuer, "crimson", "royalblue")
    draw_agents(ax, state.substrate.pos, state.substrate.theta, colors, config.substrate.agent_radius, state.substrate.alive)


def render_foraging_frame(ax, state: ForagingState, config):
    food_pos = np.asarray(state.food_pos)
    food_alive = np.asarray(state.food_alive)
    if np.any(food_alive):
        ax.scatter(
            food_pos[food_alive, 0], food_pos[food_alive, 1],
            marker="s", s=40, color="gold", edgecolors="darkgoldenrod", zorder=1,
        )
    colors = ["seagreen"] * state.substrate.pos.shape[0]
    draw_agents(ax, state.substrate.pos, state.substrate.theta, colors, config.substrate.agent_radius, state.substrate.alive)


def render_video(frames, draw_frame_fn, world_size, out_path, fps=20):
    fig, ax = plt.subplots(figsize=(6, 6))

    def update(i):
        ax.clear()
        ax.set_xlim(0, world_size)
        ax.set_ylim(0, world_size)
        ax.set_aspect("equal")
        ax.set_title(f"step {i}")
        draw_frame_fn(ax, frames[i])

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps)
    writer = "ffmpeg" if out_path.endswith(".mp4") else "pillow"
    anim.save(out_path, writer=writer, fps=fps)
    plt.close(fig)
