# Panels and layout strings

The figure is a grid of panels. Each panel holds one or more curves and has one or two y-axes.

## No layout: one shared panel

Metrics are created by logging them. With no layout, they all share one panel, with a legend:

```python
plot = LivePlot(range(300))
for step in plot:
    plot.log(loss=..., val_loss=...)    # one panel, two curves
```

## Layout strings

Pass one string per panel, after the iterable if there is one:

```python
plot = LivePlot(loader, "loss", "return | entropy", "lossD lossG | acc")
```

Within a string:

- Names separated by spaces share the left y-axis.
- Names after a `|` go on a right-hand y-axis. A string can have at most one `|`.

So the example has three panels:

| String | Left axis | Right axis |
|---|---|---|
| `"loss"` | `loss` | none |
| `"return | entropy"` | `return` | `entropy` |
| `"lossD lossG | acc"` | `lossD`, `lossG` | `acc` |

Metric names cannot contain spaces or `|` in a layout string.

### Common cases

Two metrics on one axis:

```python
plot = LivePlot(loader, "train_loss val_loss")
```

A metric on the right-hand axis:

```python
plot = LivePlot(loader, "loss | acc")
```

Several panels:

```python
plot = LivePlot(loader, "loss", "acc", "lr")
```

### Metrics that no string mentions

With a layout, a metric you log that no string mentions gets a new panel of its own, added to the end of the grid:

```python
plot = LivePlot(loader, "loss")
...
plot.log(loss=0.3, lr=1e-3)     # lr gets its own panel
```

## What a panel shows by default

- The title is the metric names joined by " / ".
- The x label is the plot's `unit`, `"step"` by default.
- A panel with two y-axes labels each axis with its metric names. A panel with one y-axis has no y label.
- Every panel has a legend. It lists the curves of both y-axes.

All of these can be changed. See [Customising axes](axes.md).

## The grid

Panels fill the grid row by row. By default the grid is at most three columns wide.

| Option | Effect |
|---|---|
| `max_cols` | Widest grid. The default is 3. `None` gives a near-square grid. |
| `rows` | Fix the number of rows. |
| `cols` | Fix the number of columns. |
| `cell_size` | `(width, height)` of one panel in inches. The default is `(5, 3.5)`. |
| `dpi` | Resolution of the image. The default is 100. |

```python
plot = LivePlot(loader, "loss", "acc", "lr", "grad_norm", cols=2, cell_size=(4, 3))
```

For a fixed grid that you fill panel by panel, as with `plt.subplots`, use [`LivePlot.subplots`][liveplot.LivePlot.subplots]. See [Images with subplots and imshow](images.md).

## The dict form

A panel can also be given as a dict in place of a string, for example `{"metrics": ["acc"], "ylim": (0, 1), "ylabel": "test accuracy"}`. This form is older. The setters in [Customising axes](axes.md) do the same job and are the intended way.
