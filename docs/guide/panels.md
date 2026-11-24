# Panels and layout strings

The figure is a grid of panels. A panel holds curves on one or two y-axes.

## No layout

Metrics exist once logged. With no layout, all share one panel with a legend:

```python
plot = LivePlot(range(300))
for step in plot:
    plot.log(loss=..., val_loss=...)    # one panel, two curves
```

## Layout strings

One string per panel, after the iterable if any:

```python
plot = LivePlot(loader, "loss", "return | entropy", "lossD lossG | acc")
```

- Space-separated names share the left y-axis.
- Names after `|` go on a right-hand y-axis. At most one `|`.

| String | Left | Right |
|---|---|---|
| `"loss"` | `loss` | — |
| `"return | entropy"` | `return` | `entropy` |
| `"lossD lossG | acc"` | `lossD`, `lossG` | `acc` |

Names in a string can't contain spaces or `|`.

A logged metric that no string mentions gets a new panel at the end:

```python
plot = LivePlot(loader, "loss")
plot.log(loss=0.3, lr=1e-3)     # lr gets its own panel
```

## Defaults

- Title: the metric names joined by " / ".
- x label: the plot's `unit` (`"step"`).
- y labels: only on two-axis panels, the metric names of each side.
- A legend on every panel, covering both axes.

Change any of them with the setters in [Customising axes](axes.md).

## The grid

Panels fill row by row, at most 3 columns by default.

| Option | Effect |
|---|---|
| `max_cols` | Widest grid. Default 3; `None` for near-square. |
| `rows` | Fix the number of rows. |
| `cols` | Fix the number of columns. |
| `cell_size` | `(width, height)` of one panel, inches. Default `(5, 3.5)`. |
| `dpi` | Image resolution. Default 100. |

```python
plot = LivePlot(loader, "loss", "acc", "lr", "grad_norm", cols=2, cell_size=(4, 3))
```

For a fixed grid you fill panel by panel, like `plt.subplots`, use [`LivePlot.subplots`][liveplot.LivePlot.subplots]; see [Images](images.md).

## The dict form

A dict also works in place of a string: `{"metrics": ["acc"], "ylim": (0, 1), "ylabel": "test accuracy"}`. It's the older form; prefer the [setters](axes.md).
