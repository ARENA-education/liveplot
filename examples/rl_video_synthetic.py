# %%
"""
A synthetic RL run, laid out like `VPGTrainer.train` in ARENA's [2.2] deep RL day, to show a looping
video of recent rollouts beside the training curves, in one live figure. Nothing is learned: 16
CartPoles (gym's physics, in numpy) are driven by a controller that gets steadier as "training" goes
on, so the poles fall over less and less, and the returns climb.

Every `video_log_freq` updates the rollout's 16 environments go to the right-hand panel as a 4x4 grid
video, which loops until the next one replaces it. A cell flashes pink when its pole falls and the
environment resets, as in the course's `rl_utils.log_grid_video`.

Run it cell by cell in the VS Code / Cursor interactive window or Jupyter to see it live. Run it as a
script (`python examples/rl_video_synthetic.py [out.gif]`) and nothing is drawn on screen, but the
run is recorded to `rl_video_synthetic.gif`. Needs only numpy and Pillow (both come with matplotlib).

The parts to copy into the real trainer are marked `# liveplot:`.
"""

import math
import sys
import time
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from liveplot import LivePlot

MAIN = __name__ == "__main__"


# %%
# The environment: gym's CartPole-v1 dynamics for a batch of carts, and a `draw` that renders one of
# them into a small picture, like `gpu_env.CartPole.draw` (which draws with cv2 into a shared canvas).


class CartPoles:
    gravity, masscart, masspole, length, force_mag, tau = 9.8, 1.0, 0.1, 0.5, 10.0, 0.02
    theta_threshold, x_threshold = 12 * 2 * math.pi / 360, 2.4

    def __init__(self, num_envs: int, seed: int = 0):
        self.num_envs, self.rng = num_envs, np.random.default_rng(seed)

    def reset(self) -> np.ndarray:
        self.state = self.rng.uniform(-0.05, 0.05, (self.num_envs, 4))
        return self.state.copy()

    def step(self, actions: np.ndarray):
        x, x_dot, theta, theta_dot = self.state.T
        force = np.where(actions == 1, self.force_mag, -self.force_mag)
        total, pml = self.masspole + self.masscart, self.masspole * self.length
        cos, sin = np.cos(theta), np.sin(theta)
        temp = (force + pml * theta_dot**2 * sin) / total
        theta_acc = (self.gravity * sin - cos * temp) / (self.length * (4 / 3 - self.masspole * cos**2 / total))
        x_acc = temp - pml * theta_acc * cos / total
        self.state = np.stack([x + self.tau * x_dot, x_dot + self.tau * x_acc,
                               theta + self.tau * theta_dot, theta_dot + self.tau * theta_acc], axis=1)
        dones = (np.abs(self.state[:, 0]) > self.x_threshold) | (np.abs(self.state[:, 2]) > self.theta_threshold)
        self.state[dones] = self.rng.uniform(-0.05, 0.05, (dones.sum(), 4))  # auto-reset, as the GPU envs do
        return self.state.copy(), np.ones(self.num_envs), dones

    def draw(self, o: np.ndarray, w: int = 96, h: int = 72, flash: bool = False) -> np.ndarray:
        """One environment's state -> an (h, w, 3) uint8 picture: the cart on its rail and the pole."""
        img = Image.new("RGB", (w, h), (255, 200, 200) if flash else (255, 255, 255))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w - 1, h - 1], outline=(220, 220, 220))
        scale = w / (2 * self.x_threshold * 1.1)
        cw, ch, cy = max(int(scale * 0.5), 6), 5, int(h * 0.62)
        cx = min(max(int(w / 2 + o[0] * scale), cw), w - cw)
        d.line([0, cy, w, cy], fill=(200, 200, 200))
        d.rectangle([cx - cw, cy - ch, cx + cw, cy + ch], fill=(0, 0, 0))
        tip = (cx + scale * math.sin(o[2]), cy - scale * math.cos(o[2]))
        d.line([cx, cy, *tip], fill=(60, 80, 200), width=3)
        return np.asarray(img)


def draw_rollout(envs: CartPoles, obs: np.ndarray, dones: np.ndarray, flash_frames: int = 6) -> np.ndarray:
    """(num_envs, T, 4) observations -> (num_envs, T, H, W, 3) uint8 clips; a cell flashes pink for a few
    frames after its episode ends, like `rl_utils._grid_frames`."""
    flash = np.zeros(dones.shape, bool)
    for i, t in zip(*np.nonzero(dones)):
        flash[i, t: t + flash_frames] = True
    return np.stack([np.stack([envs.draw(o, flash=f) for o, f in zip(obs_i, flash_i)]) for obs_i, flash_i in zip(obs, flash)])


# %%
# The trainer: the shape of the course's VPGTrainer.train, with the policy and the gradient step faked.


@dataclass
class VPGArgs:
    num_envs: int = 16
    num_steps_per_rollout: int = 128
    num_updates: int = 80
    video_log_freq: int = 16  # every N updates, show the latest rollout as a 4x4 grid video
    seconds_per_update: float = 0.12  # stands in for the gradient steps
    seed: int = 0


class SyntheticVPGTrainer:
    def __init__(self, args: VPGArgs):
        self.args = args
        self.envs = CartPoles(args.num_envs, args.seed)
        self.rng = np.random.default_rng(args.seed + 1)

    def policy(self, obs: np.ndarray, skill: float) -> np.ndarray:
        """A balancing controller behind noise that fades with `skill` in [0, 1]: random at 0, steady near 1."""
        push = obs @ np.array([0.05, 0.3, 4.0, 1.0])
        noise = self.rng.normal(0, 1, len(obs)) * (1 - skill) * 3
        return (push * skill + noise > 0).astype(int)

    def gen_rollout(self, skill: float):
        obs, all_obs, all_dones = self.envs.reset(), [], []  # full_reset=True, as in the course
        for _ in range(self.args.num_steps_per_rollout):
            all_obs.append(obs)
            obs, _, dones = self.envs.step(self.policy(obs, skill))
            all_dones.append(dones)
        return np.stack(all_obs, 1), np.stack(all_dones, 1)  # (num_envs, T, 4), (num_envs, T)

    def train(self, **plot_kwargs) -> LivePlot:
        args = self.args
        # liveplot: the curves on the left, the rollout video on the right, the tqdm bar underneath (this
        # replaces `LiveVideo()`, `log_grid_video(..., live=True)` and the `with tqdm(...)` block)
        plot, (ax_curves, self.ax_video) = LivePlot.subplots(
            1, 2, total=args.num_updates * args.num_steps_per_rollout * args.num_envs,
            unit="env steps", figsize=(12, 4.5), **plot_kwargs,
        )
        ax_curves.plot("return").set_ylim(0, args.num_steps_per_rollout * 1.05)
        ax_curves.twinx().plot("loss")
        ax_curves.set_title("mean first-episode return (left), loss (right)")

        for update in range(args.num_updates):
            skill = 1 - math.exp(-2.5 * update / args.num_updates)
            obs, dones = self.gen_rollout(skill)
            first_episode = (np.cumsum(dones, 1) - dones) == 0  # no `done` strictly before this step
            episode_return = first_episode.sum(1)  # reward 1 per step
            mean_return = float(episode_return.mean())

            if update % args.video_log_freq == 0 or update == args.num_updates - 1:
                clips = draw_rollout(self.envs, obs, dones)  # (16, T, H, W, 3), like wandb.Video's (B, T, C, H, W)
                # liveplot: was log_grid_video(tau.obs, self.envs.draw, dones=..., live=True, slot=..., caption=...)
                self.ax_video.video(clips, fps=50)  # 50 fps is CartPole's real time (tau = 0.02 s)
                self.ax_video.set_title(f"rollout at update {update}: mean return {mean_return:.0f}")

            time.sleep(args.seconds_per_update)  # the gradient steps
            loss = -mean_return * (1 + 0.3 * self.rng.normal()) / 50  # a made-up surrogate objective
            # liveplot: log the rollout's numbers and advance the bar by the env steps collected (was pbar.update)
            plot.log({"return": mean_return, "loss": loss})
            plot.update(args.num_steps_per_rollout * args.num_envs)

        plot.finish()
        return plot


# %%
# Run it. In a notebook or interactive window you get the live figure with a bar underneath; as a
# script, the run is recorded to a GIF instead (pass a path to choose where).

if MAIN:
    in_notebook = "ipykernel" in sys.modules
    gif = None if in_notebook else (sys.argv[1] if len(sys.argv) > 1 else "rl_video_synthetic.gif")
    plot = SyntheticVPGTrainer(VPGArgs()).train(**({"record": gif, "refresh_seconds": 0.5, "dpi": 80} if gif else {}))
    if gif:
        print(f"{len(plot.frames)} frames -> {gif}")
