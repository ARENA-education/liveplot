# The x-axis and `total`

## x is a step counter

The plot keeps a counter of items consumed, `plot.n`, like tqdm's `n`. The x value of a logged point is

```
x = initial + n * unit_scale
```

at the time of the [`log`][liveplot.LivePlot.log] call. With the defaults (`initial=0`, `unit_scale=1`) that is just `n`. The plot never looks at the items themselves.

`n` is 0 inside the loop body for the first item, like `for step in range(N)`. It goes up by one after each item.

`log` does not change `n`. Something else has to.

## Three ways to advance x

### Wrap the loop

A single loop: pass the iterable to the constructor and iterate the plot.

```python
plot = LivePlot(loader, "loss")
for batch in plot:
    plot.log(loss=train_step(batch))
```

This shows one progress bar and finishes the plot when the loop ends.

Nested loops: create the plot once and wrap the inner loop with [`plot(...)`][liveplot.LivePlot.__call__]. The count continues from one epoch to the next. Each wrapped loop gets its own progress bar, and keyword arguments such as `desc` go to tqdm.

```python
plot = LivePlot("loss", "acc")
for epoch in range(epochs):
    for batch in plot(loader, desc=f"epoch {epoch}"):
        plot.log(loss=train_step(batch))
    plot.log(acc=evaluate())
plot.finish()
```

If you already have a tqdm bar, pass it as the iterable. It is reused instead of wrapped in a second bar: `LivePlot(tqdm(loader), "loss")`.

### Call `update()`

[`plot.update()`][liveplot.LivePlot.update] advances the count by one, like tqdm's `update`. Use it when you do not want to wrap the loop. Call it after logging the step's metrics.

```python
plot = LivePlot("loss", total=epochs * len(loader))
for epoch in range(epochs):
    plot.set_description(f"epoch {epoch}")
    for batch in loader:
        plot.log(loss=train_step(batch))
        plot.update()
plot.finish()
```

The first `update()` opens one progress bar for the whole run.

### Pass the step yourself

Give the x value as the first argument of `log`:

```python
with LivePlot("loss") as plot:
    for step in range(num_steps):
        plot.log(step, loss=train_step())
```

Nothing is counted here and no progress bar is shown. The step can be any number, for example examples seen or seconds elapsed.

Inside a wrapped loop, an explicit step applies to that one `log` call. Without a wrapped loop, it also becomes the x of later `log` calls that give no step.

### What happens if you do none of these

Every point is logged at `x = initial`, which is 0 by default:

```python
plot = LivePlot()
for batch in loader:
    plot.log(loss=train_step(batch))

plot.data["loss"][0]   # [0, 0, 0, ...]
```

The figure shows a vertical line.

## `total` and `fixed_x`

`total` is the number of items in the whole run. It is optional.

| You write | `total` is |
|---|---|
| `LivePlot(range(1000))`, `LivePlot(loader)` | taken from `len()` |
| `LivePlot("loss", total=epochs * len(loader))` | what you gave |
| `LivePlot("loss")` | `None` |

What it changes:

- **The progress bar**, when the bar cannot get a length from what it wraps. That is the bar opened by `plot.update()`, and the bar for an iterable with no `len()`, such as a generator. With a total the bar shows a percentage and a time remaining. Without one it shows a plain count.
- **The x range**, but only with `fixed_x=True`.

Bars made by `plot(loader)` use `len(loader)`. They do not use `total`.

`fixed_x` is `False` by default: the x-axis grows with the data, and `total` does not change the figure. With `fixed_x=True` and a known total, the x-axis is fixed to

```
(initial, initial + total * unit_scale)
```

from the first frame, and the curve grows into it from the left.

```python
plot = LivePlot(range(1000), "loss", fixed_x=True)                      # x-axis is 0 to 1000 throughout
plot = LivePlot("loss", total=epochs * len(loader), fixed_x=True)       # the same for nested loops
```

`fixed_x=True` without a total has no effect. To fix the x range of one panel by hand, use [`set_xlim`][liveplot.liveplot.Panel.set_xlim].

## `initial`, `unit` and `unit_scale`

These are tqdm's arguments with tqdm's meaning.

- `initial` is the x of the first item. Use `initial=1` for a 1-based axis, or `initial=10` to plot `range(10, 20)` at its own values.
- `unit` names one item. It is the default x label and the unit on the progress bar. The default is `"step"`.
- `unit_scale` is how far x moves per item.

To plot against examples seen instead of batches:

```python
plot = LivePlot(loader, "loss", unit="examples", unit_scale=batch_size)
```

The x values are then 0, `batch_size`, `2 * batch_size`, and so on. The progress bar shows the same numbers.

## Reading the current x

[`plot.step`][liveplot.LivePlot.step] is the x that a `log` call would get right now. `plot.n` is the raw item count.

## `si_xticks`

By default the x tick labels are matplotlib's: numbers are written in full up to a million. On a long run counted in examples, such as 300,000, those labels can run into each other. `si_xticks=True` writes them with SI prefixes instead:

```python
plot = LivePlot("loss", total=epochs * len(loader), unit="examples", unit_scale=batch_size, si_xticks=True)
# x ticks: 0, 50k, 100k, ... 300k
```

The prefix is chosen once per axis, from its largest tick: k, M, G, then T. An axis whose ticks stay below 10,000 is labelled as before. From 10¹⁵ on, the axis uses matplotlib's scientific notation, with the power at the end of the axis. `LivePlot.subplots` takes the same argument.
