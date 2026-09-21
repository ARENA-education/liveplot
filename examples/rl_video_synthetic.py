# %%
"""
A synthetic RL run, laid out like `VPGTrainer` in ARENA's [2.2.2] (VPG on CartPole), to show what
liveplot looks like on that training loop without training anything: return and loss curves on the
left, and on the right a looping video of a 4x4 grid of rollouts, replaced by the newest set every
`video_every` updates while the curves keep updating. Episodes are short and reset constantly early on
(each reset flashes its cell pink, as the course's videos do) and balance for the whole rollout at the end.

Run it cell by cell in Jupyter, Colab or the VS Code interactive window to watch it live. As a script
it records `rl_video_synthetic.gif` instead (the video panel shows each video's first frame there).
Needs numpy and imageio-ffmpeg (`pip install imageio-ffmpeg`), which encodes the videos.
"""

import math
import sys
import time
from dataclasses import dataclass

import numpy as np

from liveplot import LivePlot

MAIN = __name__ == "__main__"

# %%
# The "environment": cart-poles drawn into a grid, one cell per environment, like the course's
# rl_utils._grid_frames does with each env's own draw().

CELL_W, CELL_H = 160, 120


def draw_cartpole(canvas, ox, oy, x, theta, flash):
    """One cell: track, cart and pole, with a pink background for a few frames after a reset."""
    cell = canvas[oy:oy + CELL_H, ox:ox + CELL_W]
    cell[:] = (255, 200, 200) if flash else (255, 255, 255)
    cell[[0, -1], :] = 220  # a thin grey border
    cell[:, [0, -1]] = 220
    track_y = int(CELL_H * 0.7)
    cell[track_y, :] = (120, 120, 120)
    cx = int(CELL_W / 2 + x * CELL_W / 5)
    cell[track_y - 8:track_y + 4, max(cx - 14, 0):min(cx + 14, CELL_W)] = (20, 20, 20)
    for r in np.linspace(0, 45, 60):  # the pole: a thick line from the cart, at angle theta from upright
        px, py = int(cx + r * math.sin(theta)), int(track_y - 8 - r * math.cos(theta))
        if 0 <= px < CELL_W - 2 and 0 <= py < CELL_H - 2:
            cell[py:py + 3, px:px + 3] = (70, 70, 200)


def rollout_video(skill, rng, n_envs=16, steps=250):
    """(steps, 480, 640, 3) uint8 frames of 16 cart-poles; `skill` in [0, 1] sets how long they balance."""
    frames = np.empty((steps, 4 * CELL_H, 4 * CELL_W, 3), np.uint8)
    x, theta, since_reset = rng.normal(0, 0.1, n_envs), rng.normal(0, 0.05, n_envs), np.full(n_envs, 99)
    drift = rng.normal(0, 1, n_envs)
    for t in range(steps):
        wobble = (1 - skill) * 0.08 + 0.004
        theta += wobble * drift + rng.normal(0, wobble, n_envs)
        x += 0.02 * np.sin(theta * 3) + rng.normal(0, 0.01, n_envs)
        drift = 0.9 * drift + rng.normal(0, 0.3, n_envs)
        fallen = (np.abs(theta) > 0.6) | (np.abs(x) > 2.2)
        x[fallen], theta[fallen], since_reset[fallen] = 0, rng.normal(0, 0.05, fallen.sum()), 0
        for i in range(n_envs):
            r, c = divmod(i, 4)
            draw_cartpole(frames[t], c * CELL_W, r * CELL_H, x[i], theta[i], since_reset[i] < 6)
        since_reset += 1
    return frames


# %%
# The trainer: the course's loop shape, with made-up numbers.


@dataclass
class Args:
    updates: int = 40
    env_steps_per_update: int = 16_000  # 32 envs x 500 steps, as in the course's fast config
    video_every: int = 10  # the course's video_log_freq
    seconds_per_update: float = 0.6  # the course's ~0.6 s per update on a GPU
    fps: int = 50
    seed: int = 0


def train(args: Args, **plot_kwargs) -> LivePlot:
    rng = np.random.default_rng(args.seed)
    total = args.updates * args.env_steps_per_update
    # liveplot: curves on the left, the rollout video on the right, a tqdm bar underneath
    plot, (ax_curves, ax_video) = LivePlot.subplots(1, 2, total=total, unit=" env steps", figsize=(12, 4.2), **plot_kwargs)
    ax_curves.plot("mean return", "max return")
    ax_curves.twinx().plot("loss")
    ax_curves.set_title("VPG on CartPole (synthetic)")
    ax_video.set_title("rollouts")
    for update in range(args.updates):
        skill = 1 / (1 + math.exp(-10 * (update / args.updates - 0.4)))
        mean_return = 20 + 480 * skill + rng.normal(0, 12)
        if update % args.video_every == 0 or update == args.updates - 1:
            ax_video.video(rollout_video(skill, rng), fps=args.fps)  # liveplot: replaces log_grid_video
        plot.log(**{"mean return": min(mean_return, 500), "max return": min(mean_return * 1.3 + 25, 500),
                    "loss": -0.6 * (1 - skill) + rng.normal(0, 0.05)})
        plot.update(args.env_steps_per_update)
        time.sleep(args.seconds_per_update)
    plot.finish()
    return plot


# %%
if MAIN:
    in_notebook = "ipykernel" in sys.modules
    plot = train(Args(), **({} if in_notebook else {"record": "rl_video_synthetic.gif", "refresh_seconds": 0.3}))
