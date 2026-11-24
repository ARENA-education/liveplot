# Common questions

## Is `total` needed?

No. `total` is tqdm's: the length of the run. It sets the progress bar's length when the bar can't get one from `len()`, and, with `fixed_x=True`, fixes the x-axis to the whole run. `fixed_x` is off by default, so the figure is the same with or without `total`.

`LivePlot(loader)` takes `total` from `len(loader)`. Bars made by `plot(loader)` always use `len(loader)`. See [The x-axis and total](guide/x-axis.md#total-and-fixed_x).

## Why wrap the dataloader?

Wrapping moves x. [`plot.log`][liveplot.LivePlot.log] records values but doesn't advance x, so with nothing advancing it every point lands at `x = 0`: a vertical line, not a curve.

Three ways to advance x:

```python
# 1. Wrap the loop (also gives a progress bar)
for batch in plot(loader):
    plot.log(loss=train_step(batch))

# 2. Call update() once per step
for batch in loader:
    plot.log(loss=train_step(batch))
    plot.update()

# 3. Pass the step yourself
for step, batch in enumerate(loader):
    plot.log(step, loss=train_step(batch))
```

Call `plot.finish()` after each. See [The x-axis and total](guide/x-axis.md).

## Two metrics on one axis?

One string, names separated by a space: `LivePlot(loader, "train_loss val_loss")`. With no strings at all, every metric shares one panel.

## A metric on the right-hand axis?

Put it after `|`: `LivePlot(loader, "loss | acc")`. Then `plot["acc"]` is the right-hand axis. Each side can hold several names: `"lossD lossG | acc"`.

## Several panels?

One string per panel:

```python
LivePlot(loader, "loss", "acc", "lr")                  # three panels
LivePlot(loader, "train_loss val_loss | acc", "lr")    # two, the first with two y-axes
```

A metric no string mentions gets its own panel. See [Panels and layout strings](guide/panels.md).

## Why is nothing drawn?

The plot only shows in a notebook (Jupyter, Colab, VS Code / Cursor interactive window). In a script nothing is drawn, silently. `plot.data` and [`plot.figure()`][liveplot.LivePlot.figure] still work.

## Do I need `finish()`?

Not for `for batch in plot` or `with LivePlot(...) as plot:`; both finish for you. After `plot(loader)` or `plot.update()` loops, call [`plot.finish()`][liveplot.LivePlot.finish]: it draws the last frame, closes the bar and stops the render process.
