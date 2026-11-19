# Common questions

## Is `total` needed? What does it change?

No. You can leave it out.

`total` is the number of items in the whole run, with the same meaning as tqdm's `total`. It does two things:

- It sets the length of the progress bar when the bar cannot work it out: the bar that [`plot.update()`][liveplot.LivePlot.update] opens, and the bar for an iterable that has no `len()`. With a length, the bar shows a percentage and a time remaining.
- With `fixed_x=True`, it fixes the x-axis to the whole run from the first frame.

`fixed_x` is `False` by default. The x-axis then grows with the data, and the figure looks the same with or without `total`.

If you wrap something that has a length, such as `LivePlot(loader)` or `LivePlot(range(n))`, `total` is taken from `len()`. Bars made by `plot(loader)` always use `len(loader)` and ignore `total`.

See [The x-axis and total](guide/x-axis.md#total-and-fixed_x).

## Why wrap the dataloader?

Because wrapping is what moves the x-axis. `LivePlot(loader)` and `plot(loader)` count the items they yield, and that count is the x value of every point you log.

[`plot.log`][liveplot.LivePlot.log] records values. It does not advance x. If nothing advances x, every point is logged at `x = 0` and you get a vertical line instead of a curve:

```python
plot = LivePlot()
for batch in loader:
    plot.log(loss=train_step(batch))   # every point at x = 0
```

There are three ways to advance x. Use whichever fits your loop.

```python
# 1. Wrap the loop
plot = LivePlot("loss")
for batch in plot(loader):
    plot.log(loss=train_step(batch))
plot.finish()

# 2. Call update() once per step
plot = LivePlot("loss")
for batch in loader:
    plot.log(loss=train_step(batch))
    plot.update()
plot.finish()

# 3. Pass the step yourself
plot = LivePlot("loss")
for step, batch in enumerate(loader):
    plot.log(step, loss=train_step(batch))
plot.finish()
```

Wrapping also gives you the progress bar. See [The x-axis and total](guide/x-axis.md).

## How do I put two metrics on one axis?

Put both names in one layout string, separated by a space:

```python
plot = LivePlot(loader, "train_loss val_loss")
```

They share a panel and a y-axis, and the legend tells them apart. With no layout strings at all, every metric you log shares one panel.

## How do I put a metric on the right-hand axis?

Put its name after a `|`:

```python
plot = LivePlot(loader, "loss | acc")
plot["acc"].set_ylim(0, 1)     # plot["acc"] is the right-hand axis
```

`loss` uses the left y-axis and `acc` the right one. Each side can hold several names: `"lossD lossG | acc"`.

## How do I get several panels?

Give one string per panel:

```python
plot = LivePlot(loader, "loss", "acc", "lr")                  # three panels
plot = LivePlot(loader, "train_loss val_loss | acc", "lr")    # two panels, the first with two y-axes
```

A metric you log that no string mentions gets a panel of its own. See [Panels and layout strings](guide/panels.md).

## Why is nothing drawn?

The live plot only appears in a notebook: Jupyter, Colab, or the VS Code / Cursor interactive window. In a plain script nothing is drawn and no message is shown. The metrics are still collected in `plot.data`, and [`plot.figure()`][liveplot.LivePlot.figure] still returns a figure you can save.

## Do I need to call `finish()`?

Not when you iterate the plot itself (`for batch in plot`) or use `with LivePlot(...) as plot:`. Both finish the plot for you. Call [`plot.finish()`][liveplot.LivePlot.finish] yourself after loops that use `plot(loader)` or `plot.update()`. It draws the last frame, closes the progress bar and stops the render process.
