# API

One class. Everything else is a method on it, on a panel, or on an axis, and the names are tqdm's, wandb's or matplotlib's wherever one exists.

## `LivePlot(*args, **options)`

```python
LivePlot([iterable,] *panels, total=None, initial=0, unit="step", unit_scale=1,
         refresh_seconds=1.0, smooth=None, max_cols=3, rows=None, cols=None,
         cell_size=(5, 3.5), dpi=100, progress=True, desc=None, record=False)
```

| argument | meaning |
|---|---|
| `iterable` | anything you would wrap with tqdm: a range, a DataLoader, an existing tqdm bar. Iterating the plot yields its items, shows a tqdm bar under the plot, and finishes the plot when the loop ends. Leave it out for nested loops and use `plot(inner)` instead. |
| `*panels` | layout strings, one per panel: `"loss"`, `"return | entropy"`, `"lossD lossG | acc"`. Names separated by spaces share the left y-axis; names after `|` go on a right-hand axis. With no strings, every metric shares one panel. Metrics no string mentions get a panel of their own. |
| `total`, `initial`, `unit`, `unit_scale` | tqdm's arguments, with tqdm's meaning. They define the x-axis: `x = initial + n * unit_scale`, where `n` counts items consumed. `total` is in items and fixes the x range; the single-loop form takes it from `len(iterable)`. |
| `refresh_seconds` | minimum time between redraws, default 0.2 s. `0` redraws on every arrival, as fast as rendering allows, and costs nothing while idle. |
| `smooth` | default smoothing weight for every panel, wandb's time-weighted EMA in `[0, 1)`. |
| `max_cols`, `rows`, `cols` | grid shape. `max_cols=None` gives a near-square grid. |
| `progress`, `desc` | switch the tqdm bars off, or give the single-loop form's bar a description. |
| `record` | `True` keeps every rendered frame in `plot.frames`; a path such as `"run.gif"` also writes an animated GIF at `finish()`. |

## Logging

| call | meaning |
|---|---|
| `plot.log(loss=0.3, acc=0.9)` | record metrics at the current x. Values are anything `float()` accepts, one-element tensors included. |
| `plot.log(step, loss=0.3)` | an explicit x for this call only, like wandb's `step=`. |
| `plot.log(step, {"loss": 0.3})` | the dict form. |
| `for batch in plot(loader, desc="epoch 3")` | wrap an inner loop: a tqdm bar per loop, the x count continues across loops. |
| `plot.finish()` | draw the final frame and shut the renderer down. Called for you by the iterator form and by `with`. |

## Configuring, with matplotlib's names

| handle | how to get it | setters |
|---|---|---|
| an axis | `plot["acc"]` is the y-axis holding that metric, left or right; `plot.panels[i].left` / `.right` | `set_ylabel`, `set_ylim(lo, hi)` or `set_ylim((lo, hi))`, `set_yscale("log")`, `axhline(y, label=..., **kwargs)`, `set(**kwargs)` |
| a panel | `plot.panels[i]`, or `plot["acc"].panel` | `set_title`, `set_xlabel`, `set_xlim`, `set_smooth(weight)`, `axvline(x=None, label=..., **kwargs)` on this panel, `set(**kwargs)`, plus the left axis's setters |
| the plot | `plot` | the same setters applied to every panel (and to panels created later); `axhline(y, label, metric=None)` on every left axis or on one metric's axis; `axvline(x=None, label=...)` on every panel |

A setter called during the run re-lays the figure out on the next frame. Extra kwargs on `axhline` / `axvline` go to the matplotlib artist (`color`, `linestyle`, `linewidth`, `alpha`, ...).

## Afterwards

| attribute / call | meaning |
|---|---|
| `plot.data` | `{metric: (xs, values)}`, the full history. |
| `plot.latest` | the last value of each metric. |
| `plot.figure()` | a matplotlib `Figure` of the current state, to title, tweak or `savefig`. |
| `plot.save_gif(path, speedup=1.0)` | the recorded frames as an animated GIF at the pace of the run (needs `record=True`). |
| `plot.frames` | the recorded `(time, png_bytes)` frames. |

## Interrupts

Interrupting a cell is safe: the render process ignores the interrupt, the plot freezes at its last frame with `plot.data` intact, and re-running the cell starts a fresh plot. A plot dropped without `finish()` shuts its process down when garbage collected, and the process exits by itself if the kernel dies.
