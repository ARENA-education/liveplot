# Looping rollout videos in a liveplot figure: design notes

Status: prototype on branch `live-video` (stacked on `live-imshow-dev`, PR #14). Not a PR yet.

![rl_video](https://raw.githubusercontent.com/ARENA-education/liveplot/live-video/docs/rl_video.gif)

*The synthetic demo, `python examples/rl_video_synthetic.py docs/rl_video.gif`: return and loss
on the left, and on the right a looping 4x4 grid of the latest rollout's CartPoles, pink when one
falls over and resets. The GIF is made by `record=`.*

## The goal

During an RL run, one liveplot figure shows the loss/return curves on the left and, on the right, a
looping video of a grid of recent rollouts. The video keeps looping until the next set of rollouts
replaces it, and after the run ends too.

## What ARENA does today

Paths are relative to `ARENA_3.0/worktrees/0.5-numstable/`.

- **The trainer.** `infrastructure/chapters/chapter2_rl/section_2_deep_rl/master_2_2_2.py`:
  - `VPGArgs.video_log_freq = 50` (l.627-628): every N gradient steps, a 4x4 grid video of the latest
    rollout; `live_viz = True` (l.660) shows it inline, `overwrite_video = True` (l.661) makes each
    video replace the last.
  - `VPGTrainer.train` makes one `LiveVideo` slot per run (l.1212), and every `video_log_freq` steps
    calls `log_grid_video(tau.obs, self.envs.draw, dones=tau.dones, step=env_steps_consumed,
    live=self.args.live_viz, slot=video_slot, caption=caption)` on the rollout it already has, with
    no extra environment steps (l.1247-1254). The caption is the rollout length and mean return.
  - The curves are not plotted at all: the tqdm postfix carries `G = mean ± ci (max)`, `joy`, `lr`
    (l.1276-1283). The markdown at l.1090-1097 and l.1392 describes the video and the pink flash.
- **The rendering.** `chapter2_rl/exercises/rl_utils.py`:
  - `_grid_frames` (l.291-326): takes `obs` of shape (n_envs, T, ...), keeps the first 16
    environments, subsamples to at most `max_frames=250`, and draws every frame of a 640x480 canvas
    (a 4x4 grid of 160x120 cells) by calling `draw_cell(obs[i, t], canvas, ox, oy, w, h)` for each
    cell. **The pink flash:** for `flash_frames=6` frames after each `done`, the cell's background is
    painted (255, 200, 200) before the environment draws on it. Each cell gets a 1 px grey (220) border.
  - Environments draw themselves: `GPUEnv.draw` (`chapter2_rl/exercises/gpu_env.py` l.107-114) is an
    optional per-cell renderer, and `CartPole.draw` (l.470-481) draws the rail, a black cart and a
    blue pole with cv2 directly into the shared canvas. `AtariEnvs.draw` (rl_utils l.96-103) blits the
    last stacked frame; Brax has its own path (l.230-265).
  - `frames_to_video_html` (l.329-339): `imageio.mimsave(..., fps=50)` to a temporary **MP4**
    (imageio-ffmpeg, H.264), base64 into
    `<video autoplay loop controls muted src="data:video/mp4;base64,..." width="640">`, as an
    `IPython.display.HTML`. `render_rollout_grid_html` (l.342-347) glues the two with `fps=50`.
    128 steps per rollout at 50 fps is a 2.6 s loop.
  - `LiveVideo` (l.370-391): the first `show` does `display(video, display_id=True)`, later ones
    `handle.update(video)`, so each video replaces the previous one in place; the caption is an HTML
    `<div>` inside the same output.
  - `log_grid_video` / `log_video_html` (l.403-435): render, then `wandb.log({key: wandb.Html(...)})`
    if a run is active, then show it in the slot. Any exception is printed, never raised.

So today there are two separate outputs in the cell: a tqdm bar with numbers, and an HTML `<video>`
below it. There is no curve at all, and the video relies on the frontend playing an H.264 MP4 from a
data URI.

## Why a video doesn't fit liveplot as it stands

liveplot's display is one PNG, drawn by a spawned render process, swapped into one output with a
`display_id` update at most once per `refresh_seconds`. Any output update replaces the output's DOM.
So whatever the video is, the curves redrawing every second would restart it every second.

## Options

**(a) The render process cycles the frames itself**, redrawing the figure at the video's fps and
sending each frame as a PNG update.
- Everything stays a still PNG; `record=` captures the video with no changes.
- Every video frame is a whole figure through the kernel's output channel: ~60-100 kB at 50 fps
  is 3-5 MB/s, and even with blitting (redraw only the image axes) a redraw plus PNG encode is
  ~20-50 ms, so the real rate is 10-25 fps at best. Classic Notebook (< 7) would hit its 1 MB/s
  `iopub_data_rate_limit`.
- The frames are shown by the collector thread in the kernel, so the video stutters whenever the
  training loop holds the GIL or the kernel is busy.
- It stops when the render process exits: after `finish()`, and in a saved notebook, the video is a
  still frame.
- A second clock besides `refresh_seconds`.

**(b) Composite HTML output**: the figure PNG plus a `<video>` (or GIF) beside it or positioned over
it, in one HTML output.
- Every curve redraw replaces the HTML, so the video restarts from frame 0 each `refresh_seconds`
  and the whole MP4 is re-sent each time. Putting the video in a *second* output (its own
  `display_id`) avoids the restart but it is no longer one figure, and it can only go below the
  curves, not beside them.
- H.264 MP4: VS Code's Electron build is said not to ship proprietary codecs, so the MP4 may not
  play in the VS Code interactive window; I couldn't check. WebM would need an encoder we don't
  have (imageio-ffmpeg isn't a liveplot dependency and shouldn't become one).
- `record=` can't see an HTML video; it would have to re-composite.
- An overlay needs CSS positioning that each frontend sanitises in its own way.
- A variant: the whole figure as an animated **GIF** each redraw. That gives one image, but 256
  colours, a costly quantisation every redraw, and it still restarts.

**(c) Recommended: the figure as an animated PNG (APNG), frames rotated to "now".**
- An APNG *is* a PNG: the same signature and the same `image/png` MIME type, and viewers that
  can't animate it show the first frame. So nothing in the display path changes: it is still
  `IPython.display.Image(data=png)` in one `display_id` output. Every browser engine animates APNG
  in an `<img>` (Chromium/Electron since 2017, Firefox since 2008, Safari since 2014), and every
  frontend we care about draws `image/png` output as an `<img>`.
- The first APNG frame is the whole figure. The later frames repaint **only the video panel's pixel
  box** (APNG `fcTL` offsets), so the curves are encoded once per redraw.
- **Each video frame's box is compressed once, when the video arrives**, and cached. A redraw then
  encodes the full first frame and splices the cached chunks behind it: renumbering chunks and
  recomputing CRCs is cheap (`liveplot/_video.py`, ~100 lines, stdlib `struct`/`zlib` plus Pillow).
- **No restarts.** A browser starts a new image at its first frame, so each redraw rotates the frames
  to start where the video has got to, `(now - arrival) * fps`. The video plays on across redraws.
  The one visible cost is a small hitch of encode-plus-transport latency, about 0.1-0.3 s, at each swap.
- **It keeps looping forever**, after `finish()`, in a saved notebook, in nbconvert HTML: there is
  no process behind it.
- The wire cost is one APNG per redraw instead of one per video frame. For the demo figure
  (12x4.5 in, dpi 100, 128 frames of a 4x4 CartPole grid) that is ~430 kB per redraw, measured.
  The render worker keeps the output under 500 kB/s by waiting at least `len(png) / 500 kB` between
  frames that carry a video (`_WIRE_BYTES_PER_SECOND`). The video loops in the browser
  meanwhile; only the curves wait a little longer.
- `record=`: `save_gif` expands each recorded APNG into the stretch of video that played until the
  next frame replaced it (at most 25 fps), plus a full loop at the end. One shared palette as before.

## The API (prototype)

wandb's name for the object and its layouts, matplotlib's pattern for the call:

```python
ax.video(frames, fps=4, *, rows=None, cols=None, griddim=None, vmin=None, vmax=None,
         scale_each=False, channels=None, cmap="gray", padding=0, pad_value=0,
         max_images=64, max_cols=8)
plot.video(frames, **kwargs)   # like plot.imshow: uses/creates the plot's image panel
```

- `frames` uses `wandb.Video`'s layouts, which are also TensorBoard's `add_video` layouts: `(T, C, H, W)`
  for one clip, `(B, T, C, H, W)` for a batch, which is tiled into a near-square grid frame by frame
  like `imshow`'s batches. Also accepted: channels-last `(T, H, W, C)` / `(B, T, H, W, C)` and
  grayscale `(T, H, W)`. `fps` is wandb's keyword with wandb's default, 4.
- The grid and range keywords are `imshow`'s. The range is taken over the whole clip, not frame by
  frame, so the brightness doesn't flicker.
- Tiling and uint8 conversion run on the calling thread (`_images.to_video`). ARENA-sized input
  (16 clips x 128 frames of 160x120 RGB, 118 MB) takes 0.05 s.
- A later `video()` or `imshow()` on the panel replaces the video. There is one video per figure.
- `plot.figure()` shows the first frame.

### The VPG loop with it

```python
plot, (ax_curves, ax_video) = LivePlot.subplots(
    1, 2, total=args.total_timesteps, unit="env steps", figsize=(12, 4.5))
ax_curves.plot("return")
ax_curves.twinx().plot("loss")

for update_num in range(num_updates):
    rollout = self.agent.gen_rollout(rollout)
    tau = rollout.get()
    ...  # mean_return as now
    if self.args.video_log_freq and train_steps >= next_video_at:
        # the course's own renderer, pink flashes and all: (T, 480, 640, 3) uint8
        frames = _grid_frames(tau.obs, self.envs.draw, dones=tau.dones)
        ax_video.video(frames, fps=50)  # replaces LiveVideo + log_grid_video(live=True)
        ax_video.set_title(f"rollout {rollout.timestep} steps | mean return {mean_return:.1f}")
        if wandb.run is not None:
            wandb.log({"rollout_video": wandb.Video(frames.transpose(0, 3, 1, 2), fps=50)}, step=env_steps_consumed)
        next_video_at += self.args.video_log_freq
    for batch in rollout_batches:
        loss, info = self.compute_loss(batch)
        ...
        plot.log({"return": mean_return, "loss": loss.item()})
    plot.update(self.args.num_steps_per_rollout * self.args.num_envs)  # was pbar.update
plot.finish()
```

`env.draw` draws into a shared canvas, so `_grid_frames` has already tiled the frames, and
`ax.video` takes the tiled `(T, H, W, C)` directly. An environment that returned one picture per
state could pass `(num_envs, T, H, W, C)` and let liveplot tile it, as the demo does.

## What was verified, and how

- **Tests** (`tests/test_video.py`, 11 tests; full suite 76 passed): layouts; each video frame equals
  `to_grid` of that time step; range over the whole clip; the renderer's output is a PNG that
  Pillow reads as an endlessly looping APNG with T frames 1/fps apart, the size of the still figure;
  the later frames change pixels only inside the AxesImage's window extent; the phase rotation (a
  redraw 2.5 frames later starts on frame 2); `imshow` replaces a video; panel rules; `figure()`;
  `save_gif` replays the video at real pace and caps it at 25 fps; and process mode end to end, where
  the final output is still animated after `finish()`.
- **JupyterLab 4.2.5 in Chromium 153** (headless, driven by Playwright): the demo notebook ran live.
  Screenshots of the output `<img>` every 0.6 s showed the curves half changing during training
  and the video half changing throughout. After `finish()` the `src` stayed fixed (428 kB) while
  the video half kept changing for the 25 s I watched. So it plays, and it keeps looping when
  nothing is left running.
- **Not verified:** Colab, the VS Code / Cursor interactive window, Notebook 7, Firefox and Safari.
  They should work, because all of them show `image/png` output in an `<img>` and all of those
  engines animate APNG, but nobody has looked. The risk to check first is a frontend that
  re-encodes or thumbnails PNG output: it would show the still first frame, which is a graceful
  failure.
- Demo: `examples/rl_video_synthetic.py`, run as a script, writes the GIF above. I checked sampled
  GIF frames by eye: pink flashes early in the run, steady poles late, and curves and title in step
  with the video.

## Costs and limits found

- A new video's first redraw compresses every frame's box: ~1 s for 128 CartPole frames. Later
  redraws take ~0.1 s.
- **High-entropy video is expensive.** A 128-frame 336x336 noise-like video (a stand-in for Atari
  at its worst) gave a 23 MB APNG and a 5 s encode. The wire budget would then hold the curves back
  to one redraw every ~45 s. Real Atari frames are much flatter than noise, but this is where the
  design is weakest. Possible levers: fewer frames, a lower fps, or a smaller figure. APNG is
  lossless, so there is no quality knob.
- One video per figure. Several would need a common frame clock in one APNG.
- In thread mode (the fallback when the render process can't start), the APNG is built on the
  training thread: ~0.1 s per redraw, ~1 s when a video arrives.

## Open questions for David

1. **Name.** `ax.video(frames, fps=...)` is wandb's noun used as a verb. Alternatives:
   `ax.imshow(frames, fps=...)` (matplotlib's name, with `fps` switching the time axis on, but then
   a 4-d array is ambiguous: a batch (B, C, H, W) or a clip (T, C, H, W)?), or wandb's shape exactly,
   `plot.log(rollout=liveplot.Video(frames, fps=50))`.
2. **`fps` default.** wandb's 4 (current), or none, so it must be given? The course always passes 50.
3. **Heavy videos.** Should liveplot cap them itself, e.g. `max_frames` as `rl_utils._grid_frames`
   has, or at most 25 fps by dropping frames? Or leave that to the caller?
   Is the 500 kB/s budget right? Colab's output limits are unknown to me.
4. **The pink reset flash** stays in the course's drawing code (`_grid_frames`), not in liveplot
   (`dones=`)? I'd keep it out: it is how an environment is drawn, not how a figure is plotted.
5. **One video per figure**: acceptable for now?
6. The video panel's title shows the step it was logged at, like `imshow`'s. With a caption as long
   as the course's, that's a long title. Keep it?
7. Could someone with **Colab and VS Code** open the demo notebook and confirm that it animates, and
   keeps animating after the cell finishes?
