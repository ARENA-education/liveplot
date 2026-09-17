# Guide

The one-liner first, then everything it can grow into.

```python
from liveplot import LivePlot

plot = LivePlot(range(num_steps))          # a tqdm bar + a live plot
for step in plot:
    loss, acc = train_step()
    plot.log(loss=loss, acc=acc)           # step is implicit; the bar's postfix shows the latest values
```

Metrics are discovered from what you log. With no layout given they all share one panel, with a legend. To split them up, give panel strings:

```python
plot = LivePlot(range(num_steps), "loss", "return | entropy", "lossD lossG | acc")
```

Each string is one panel. Names separated by spaces share the left y-axis; names after a `|` go on a right-hand y-axis. Anything you log that no string mentions gets a panel of its own. Every panel has a legend.

For nested loops, create the plot once and wrap the inner loop with `plot(...)`; the count, and so the x-axis, continues across epochs and you get one tqdm bar per epoch, as with tqdm:

```python
plot = LivePlot("loss", "acc", total=epochs * len(loader))
for epoch in range(epochs):
    for imgs, labels in plot(loader, desc=f"epoch {epoch}"):   # kwargs go to tqdm
        plot.log(loss=train_step(imgs, labels))
    plot.log(acc=evaluate())                                    # metrics can have different cadences
plot.finish()                                                   # or wrap the whole thing in `with LivePlot(...) as plot:`
```

If you'd rather supply the x values yourself, use the context manager and pass the step to `log`:

```python
with LivePlot("loss", "acc", total=num_steps) as plot:
    for step in range(num_steps):
        plot.log(step, loss=train_step())
```

Already have a tqdm bar? Pass it as the iterable and it is reused instead of wrapped: `LivePlot(tqdm(loader), "loss")`.

## Metrics logged at different rates

Nothing special is needed. Every metric keeps its own list of `(x, value)` points, and each `log` call stamps its metrics with the current x. A metric logged every step gets a point per step; one logged every 50 steps, or once per epoch after the inner loop, gets a point wherever the count was at the time and is drawn as a line through those points. So in `"loss | eval_loss eval_acc"` the left axis fills in continuously while the right axis grows a point every 50 steps, all against the same x.

## The x-axis

It follows tqdm: the plot counts items consumed and never looks at their values.

    x = initial + n * unit_scale

`n` is 0 inside the loop body for the first item, like `for step in range(N)`. `initial` shifts the start (`initial=1` for 1-based, `initial=10` to plot `range(10, 20)` at its values). `unit` and `unit_scale` relabel and rescale it: `unit="examples", unit_scale=batch_size` plots against examples seen, and the tqdm bar shows the same numbers. `total`, in items like tqdm's, fixes the x range; the single-loop form takes it from `len(iterable)`. An explicit `plot.log(step, ...)` uses that x for that call only, like wandb's `step=`.

## How it works

The training thread only appends numbers (about 40 µs per `log`). A separate render process owns the matplotlib figure, redraws it at most once per `refresh_seconds` (default 0.2 s; `0` means on every arrival), and sends back PNG bytes that get swapped into a fixed output cell. The output is a plain image, so it behaves identically in Jupyter, Colab, VS Code and Cursor: no widgets, no JavaScript, no CDN.

Interrupting the cell is safe. Jupyter sends its interrupt to every process the kernel started; the render process ignores it, so you get a frozen plot with `plot.data` intact. A plot that is dropped without `finish()` shuts its process down when garbage collected, and the process exits by itself if the notebook kernel dies.

## Starting instantly

A plot's first frame waits for its render process to start, which is mostly a fresh Python importing matplotlib: about 0.6 s. Call `liveplot.warm()` once, in a setup cell, and a render process is kept ready in the background, so every plot from then on appears as soon as its first point is logged.

```python
import liveplot
liveplot.warm()
```

It is optional and off by default; the cost is one idle process that exits with the kernel.

## Options

| | |
|---|---|
| `total`, `initial`, `unit`, `unit_scale` | tqdm's arguments, with tqdm's meaning; they define the x-axis (see above) |
| `refresh_seconds` | minimum time between redraws (default 0.2, the same interval fastprogress uses for its live bars). Points arriving in between are batched into the next frame. A frame itself takes about 0.15 s to render, so going lower mostly just keeps the renderer busy. `0` redraws whenever new data arrives, as fast as rendering allows (roughly 0.15 s per frame at a few thousand points), and costs nothing while idle. |
| `max_cols`, `rows`, `cols` | grid shape; `max_cols=None` gives a near-square grid |
| `progress`, `desc` | disable the bundled tqdm bars, or give the single-loop form's bar a description |
| `cell_size`, `dpi` | size of each panel in inches, and PNG resolution |
| `record` | `True` keeps every rendered frame in `plot.frames`; a path such as `"run.gif"` also writes an animated GIF at `finish()`. `plot.save_gif(path, speedup=1.0)` does it on demand. Works outside a notebook too. |

## Titles, labels, limits: matplotlib's names

Panels and axes are addressed and configured with the setters you already know from matplotlib, before or during the loop:

```python
plot = LivePlot(loader, "loss | acc", "lr")

plot["acc"].set_ylim(0, 1)                     # plot[metric] is the y-axis holding that metric (left or right)
plot["acc"].set_ylabel("test accuracy")
plot["loss"].axhline(0.1, label="target", color="red")   # extra kwargs go to the matplotlib artist

plot.panels[0].set_title("training")           # a panel: set_title, set_xlabel, set_xlim, set_smooth, axvline
plot.panels[0].set_xlim(0, 10_000)
plot.panels[0].axvline(label="lr drop")        # at the current x

plot.set(xlabel="examples", smooth=0.9)        # on the plot itself: every panel, like Axes.set(**kwargs)
```

`plot[metric]` finds the right-hand axis too, so there is no separate spelling for it; `plot.panels[i].left` / `.right` name the two axes explicitly. Everything on an axis takes matplotlib's name and signature: `set_ylabel`, `set_ylim` (two arguments or a tuple), `set_yscale("log")`, `axhline`, `set(**kwargs)`, and the panel's own setters are reachable from it as they would be on an `Axes`. A setter called mid-run just triggers a re-layout on the next frame.

The dict form, `{"metrics": ["acc"], "ylim": (0, 1), ...}`, is still accepted in the layout list but the setters are the intended way.

`plot.figure()` returns a matplotlib Figure of the plot as it stands, built independently of the live renderer, for `fig.savefig(...)`, a title, or any other tweak.

`plot.log` accepts keywords, an explicit step (`plot.log(step, loss=...)`), or a dict (`plot.log(step, {"loss": ...})`). Values can be anything `float()` accepts, including one-element tensors. `plot.data` holds the full history as `{metric: (steps, values)}` and `plot.latest` the most recent value of each.

## Reference lines

Matplotlib's `axhline` / `axvline`, on an axis, a panel, or the whole plot. Horizontal lines get a legend entry; extra kwargs go to the artist.

```python
plot["loss"].axhline(math.log(d_vocab), "uniform")        # the level a curve should beat
plot["return"].axhline(500, "solved", color="green")
plot.axvline(label="lr drop")                             # every panel, at the current x
plot.panels[1].axvline(2000, "checkpoint", linestyle="-")  # one panel, at a given x
```

## Smoothing and log axes

Per-step losses are noisy. `set_smooth(0.9)` draws each curve of a panel through wandb's default smoothing, the [time-weighted exponential moving average](https://docs.wandb.ai/models/app/features/panels/line-plot/smoothing), with the same 0 to 1 weight as wandb's smoothing slider and the raw values faded behind. On the plot it applies to every panel; `set_smooth(0)` turns it off. `set_yscale("log")` on an axis, a panel, or the plot gives log axes.

```python
plot.set_smooth(0.9)
plot["loss"].set_yscale("log")
```

