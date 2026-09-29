# %%
"""
liveplot tour. A cell-separated file: run it cell by cell in the VS Code / Cursor interactive
window, open it in Colab (the README has the link: CI turns this file into a notebook on the
`demo` branch), or run it as a plain script, which prints instead of drawing (except the
recording cell, which renders).
"""

# Setup: on Colab, install liveplot; anywhere else it is already installed.
import subprocess
import sys

try:
    import google.colab  # noqa: F401

    IN_COLAB = True
except ImportError:
    IN_COLAB = False
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "liveplot[notebook] @ git+https://github.com/ARENA-education/liveplot.git"], check=True)

# %%
import math
import random
import time

import numpy as np

from liveplot import LivePlot

MAIN = __name__ == "__main__"


def slow(seconds=0.01):
    """The toy models below train in well under a second; this makes the plots watchable."""
    time.sleep(seconds)


# %%
# 1. The one-liner: wrap the range like tqdm, log keyword arguments. Metrics are discovered from
#    what you log and, with no layout given, share a single panel with a legend. A tqdm bar sits
#    under the plot with the latest values as its postfix.

if MAIN:
    plot = LivePlot(range(300))
    for step in plot:
        plot.log(loss=math.exp(-step / 80) + 0.05 * random.random(), val_loss=math.exp(-step / 80) + 0.1)
        slow()

# %%
# 2. Layout strings and metrics logged at different rates. Each string is a panel; names separated
#    by spaces share the left axis, names after "|" go on a right-hand axis. The eval metrics are
#    logged only every 50 steps: they simply get fewer points, on the same x-axis.
#    This is a real (tiny) model: a 2-16-1 MLP on an XOR-ish problem, trained with hand-written SGD.


def make_data(n, rng):
    x = rng.normal(size=(n, 2))
    return x, (x[:, 0] * x[:, 1] > 0).astype(float)


def forward(params, x):
    w1, b1, w2, b2 = params
    h = np.tanh(x @ w1 + b1)
    return h, 1 / (1 + np.exp(-(h @ w2 + b2)))


def bce(p, y):
    return -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))


def train_classifier(steps=600, batch_size=64, lr=0.5, eval_every=50, seed=0, pace_seconds=0.0, **plot_kwargs):
    rng = np.random.default_rng(seed)
    x_train, y_train = make_data(4096, rng)
    x_eval, y_eval = make_data(1024, rng)
    params = [rng.normal(scale=0.5, size=(2, 16)), np.zeros(16), rng.normal(scale=0.5, size=16), 0.0]

    plot = LivePlot(range(steps), "loss | eval_loss eval_acc", unit="examples", unit_scale=batch_size, **plot_kwargs)
    for step in plot:
        idx = rng.integers(0, len(x_train), batch_size)
        x, y = x_train[idx], y_train[idx]
        h, p = forward(params, x)
        dz = (p - y) / batch_size  # backprop by hand: it's a demo of the plot, not of autograd
        dh = np.outer(dz, params[2]) * (1 - h**2)
        for prm, grad in zip(params, (x.T @ dh, dh.sum(0), h.T @ dz, dz.sum())):
            prm -= lr * grad
        plot.log(loss=bce(p, y))  # every step
        if step % eval_every == 0 or step == steps - 1:  # rarely
            _, p_eval = forward(params, x_eval)
            plot.log(eval_loss=bce(p_eval, y_eval), eval_acc=np.mean((p_eval > 0.5) == y_eval))
        if pace_seconds:
            time.sleep(pace_seconds)
    return plot


if MAIN:
    plot = train_classifier(pace_seconds=0.01)
    print("final:", {k: round(v, 3) for k, v in plot.latest.items()})

# %%
# 3. Nested loops (epochs x batches). Create the plot once and wrap the INNER loop with plot(...):
#    the count, and so the x-axis, continues across epochs and you get one tqdm bar per epoch,
#    exactly as with tqdm. `total` is in items, like tqdm's, and fixes the x range up front.
#    Titles, labels and limits use matplotlib's own setter names, addressed by metric or by panel.

if MAIN:
    epochs, loader = 3, [None] * 60
    plot = LivePlot("loss", "acc", total=epochs * len(loader))
    plot["acc"].set_ylim(0, 1)                  # plot[metric] is the y-axis holding that metric
    plot["acc"].set_ylabel("test accuracy")
    plot.panels[0].set_title("training loss")
    plot.set(xlabel="batches")                  # every panel, like Axes.set(**kwargs)
    for epoch in range(epochs):
        for batch in plot(loader, desc=f"epoch {epoch}"):
            plot.log(loss=math.exp(-plot.n / 60) + 0.05 * random.random())
            slow()
        plot.log(acc=1 - 0.5 * math.exp(-(epoch + 1)))  # once per epoch, lands at the end of the epoch
    plot.finish()

# %%
# 4. Two left-axis curves plus a right-axis one on the same panel ("lossD lossG | acc"), and a
#    metric that no layout string mentions (lr) appearing mid-run: it gets a panel of its own.
#    This cell uses the context-manager form with explicit steps, for loops you want to own.

if MAIN:
    with LivePlot("lossD lossG | acc", total=400) as plot:
        for step in range(400):
            plot.log(step, lossD=math.exp(-step / 150) + 0.05 * random.random(), lossG=1.5 * math.exp(-step / 250) + 0.05 * random.random())
            if step % 40 == 0:
                plot.log(step, acc=min(1.0, step / 300))
            if step >= 200:
                plot.log(step, lr=1e-3 * (400 - step) / 200)
            if step == 200:
                plot.axvline(label="lr decay starts")     # a vertical line on every panel, at the current x
                plot["acc"].axhline(0.9, "target")        # a horizontal reference line on acc's axis
            slow()

# %%
# 5. Your own tqdm bar. Pass a tqdm object as the iterable and it is reused (postfix included)
#    rather than wrapped in a second bar.

if MAIN:
    from tqdm.auto import tqdm

    bar = tqdm(range(200), desc="my own bar", unit="batch")
    plot = LivePlot(bar, "loss")
    for step in plot:
        plot.log(loss=1 / (1 + step / 30))
        slow()

# %%
# 6. Interrupts are safe. Run this cell and press "interrupt" (or Ctrl-C) while it's going: the
#    plot freezes at its last frame, `plot.data` keeps everything logged so far, and no render
#    process is left behind. Re-run the cell and you get a fresh plot.

if MAIN:
    plot = LivePlot(range(1000), "loss")  # ~20 s: plenty of time to interrupt it
    try:
        for step in plot:
            plot.log(loss=1 / (1 + step / 100))
            slow(0.02)
    except KeyboardInterrupt:
        pass
    print(f"kept {len(plot.data['loss'][0])} points")

# %%
# 7. Recording. record=True keeps every rendered frame in plot.frames; record="file.gif" writes an
#    animated GIF at finish() that replays at the pace of the run. This works outside a notebook
#    too, which is how docs/demo.gif is made (see examples/make_gif.py).

if MAIN:
    plot = train_classifier(pace_seconds=0.01, record="tour.gif", refresh_seconds=0.15, cell_size=(4.2, 3.0), dpi=80)
    print(f"{len(plot.frames)} frames written to tour.gif")

# %%
# 8. After training, the full history is in plot.data as {metric: (xs, values)} for whatever you
#    want to do next, and plot.latest has the last value of each metric.

if MAIN:
    xs, losses = plot.data["loss"]
    print(f"{len(xs)} loss points, x from {xs[0]} to {xs[-1]} examples; final eval acc {plot.latest['eval_acc']:.3f}")

# %%
# 9. Images beside curves: LivePlot.subplots mirrors plt.subplots, and a panel can hold a picture
#    instead of lines. `ax.plot("name", ...)` says which metrics live on an axis (matplotlib spells
#    the same idea `ax.plot("name", data=d)`); `ax.imshow(tensor)` draws a batch as a grid and
#    replaces it on every call. This is the shape a GAN training loop wants: loss curves updating
#    every step, generated samples every so often, in one figure.
#
#    A batch is tiled for you: `rows=` or `cols=` infers the other, and values are scaled to the
#    full range unless you fix it with vmin/vmax -- worth doing here, since a generator that ends
#    in tanh always produces [-1, 1] and a fixed range keeps the black point still between frames.


def fake_generator(step, rng, n=8, size=24):
    """Stand-in for a generator: blobs that sharpen as `step` grows, in tanh range like a real one."""
    y, x = np.mgrid[0:size, 0:size] / size
    out = np.empty((n, 3, size, size))
    for i in range(n):
        cx, cy, sharp = rng.random(), rng.random(), 0.02 + 0.5 / (1 + step / 40)
        blob = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / sharp)
        out[i] = np.stack([blob, blob * (0.3 + 0.7 * cx), blob * (0.3 + 0.7 * cy)])
    return out * 2 - 1  # [-1, 1], as a tanh output would be


if MAIN:
    rng = np.random.default_rng(0)
    plot, (ax_loss, ax_samples) = LivePlot.subplots(1, 2, total=400, figsize=(11, 4))
    ax_loss.plot("lossD", "lossG")
    ax_loss.twinx().plot("D(x)")  # matplotlib's own spelling for a right-hand axis
    ax_samples.set_title("generator samples")

    for step in plot(range(400)):
        plot.log(lossD=1 / (1 + step / 50) + 0.05 * rng.random(),
                 lossG=0.4 + 0.3 * rng.random(),
                 **{"D(x)": 0.5 + 0.3 / (1 + step / 80)})
        if step % 25 == 0:
            ax_samples.imshow(fake_generator(step, rng), rows=2, vmin=-1, vmax=1)
        slow()
    plot.finish()

# %%
# 10. One image on its own, overwritten in place: plot.imshow() makes the panel on first use. The
#     accepted layouts are (H, W), (C, H, W), (H, W, C), (B, H, W), (B, C, H, W) and (B, H, W, C).
#     (3, H, W) and (4, H, W) are the ambiguous ones -- one colour image, or that many grayscale? --
#     and they raise, telling you which `channels=` to pass.

if MAIN:
    plot = LivePlot()
    for step in range(40):
        plot.imshow(np.random.default_rng(step).random((6, 1, 28, 28)), rows=2)  # replaces the last
        slow(0.05)
    plot.finish()
    try:
        plot.imshow(np.zeros((3, 8, 8)))
    except ValueError as e:
        print(e)
