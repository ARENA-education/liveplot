# The x-axis and `total`

## x is a step counter

`plot.n` counts items consumed, like tqdm's `n`. A point's x is

```
x = initial + n * unit_scale
```

when [`log`][liveplot.LivePlot.log] is called. `n` is 0 for the first item and goes up by one after each. The items themselves are never looked at. `log` doesn't change `n`.

## Three ways to advance x

### Wrap the loop

Single loop: pass the iterable and iterate the plot. One progress bar; the plot finishes when the loop ends.

```python
plot = LivePlot(loader, "loss")
for batch in plot:
    plot.log(loss=train_step(batch))
```

Nested loops: create the plot once, wrap the inner loop with [`plot(...)`][liveplot.LivePlot.__call__]. The count carries across epochs. Each wrapped loop gets its own bar; keyword arguments such as `desc` go to tqdm.

```python
plot = LivePlot("loss", "acc")
for epoch in range(epochs):
    for batch in plot(loader, desc=f"epoch {epoch}"):
        plot.log(loss=train_step(batch))
    plot.log(acc=evaluate())
plot.finish()
```

An existing tqdm bar passed as the iterable is reused, not wrapped: `LivePlot(tqdm(loader), "loss")`.

### Call `update()`

[`plot.update()`][liveplot.LivePlot.update] adds one to the count, like tqdm's. Call it after logging the step. The first call opens one bar for the whole run.

```python
plot = LivePlot("loss", total=epochs * len(loader))
for epoch in range(epochs):
    plot.set_description(f"epoch {epoch}")
    for batch in loader:
        plot.log(loss=train_step(batch))
        plot.update()
plot.finish()
```

### Pass the step yourself

```python
with LivePlot("loss") as plot:
    for step in range(num_steps):
        plot.log(step, loss=train_step())
```

No counting, no bar. The step can be any number (examples seen, seconds). Inside a wrapped loop it applies to that call only; otherwise later calls without a step reuse it.

### None of these

Every point lands at `x = initial` (0), and the figure is a vertical line: `plot.data["loss"][0] == [0, 0, 0, ...]`.

## `total` and `fixed_x`

`total` is the number of items in the run. Optional.

| You write | `total` is |
|---|---|
| `LivePlot(range(1000))`, `LivePlot(loader)` | `len()` |
| `LivePlot("loss", total=epochs * len(loader))` | what you gave |
| `LivePlot("loss")` | `None` |

It sets:

- **The bar's length**, when the bar can't get one from `len()`: the bar from `plot.update()`, or a generator. Then the bar shows a percentage and ETA. Bars from `plot(loader)` use `len(loader)`, not `total`.
- **The x range**, only with `fixed_x=True`: `(initial, initial + total * unit_scale)` from the first frame, filled from the left.

`fixed_x` is off by default: the x-axis grows with the data. Without a `total` it does nothing. For one panel, use [`set_xlim`][liveplot.liveplot.Panel.set_xlim].

```python
plot = LivePlot(range(1000), "loss", fixed_x=True)                  # x is 0 to 1000 throughout
plot = LivePlot("loss", total=epochs * len(loader), fixed_x=True)   # same, nested loops
```

## `initial`, `unit`, `unit_scale`

tqdm's arguments, tqdm's meaning:

- `initial`: x of the first item. `initial=1` for 1-based; `initial=10` plots `range(10, 20)` at its values.
- `unit`: the name of one item. Default x label and bar unit. Default `"step"`.
- `unit_scale`: x per item.

Plot against examples seen:

```python
plot = LivePlot(loader, "loss", unit="examples", unit_scale=batch_size)
```

x is then 0, `batch_size`, `2 * batch_size`, …; the bar shows the same numbers.

## Reading the current x

[`plot.step`][liveplot.LivePlot.step]: the x a `log` call would get now. `plot.n`: the raw count.

## `si_xticks`

Default tick labels are matplotlib's, written in full up to a million, so long runs get crowded labels. `si_xticks=True` uses SI prefixes:

```python
plot = LivePlot("loss", total=epochs * len(loader), unit="examples", unit_scale=batch_size, si_xticks=True)
# x ticks: 0, 50k, 100k, ... 300k
```

One prefix per axis, from its largest tick: k, M, G, T. Axes below 10,000 are unchanged. From 10¹⁵, matplotlib's scientific notation. `LivePlot.subplots` takes it too.
