# Saving and recording

## A matplotlib figure

[`plot.figure()`][liveplot.LivePlot.figure] returns a matplotlib `Figure` of the current state: panels, curves, reference lines, images.

```python
fig = plot.figure()
fig.suptitle("run 12")
fig.savefig("run12.png", dpi=150)
```

![a figure exported with plot.figure() and saved as a PNG](../figure_export.png)

Built from `plot.data` on each call and separate from the live display. Works during or after the run, in a script, and with `show=False`.

## Recording a GIF

`record="run.gif"` keeps every drawn frame and writes a GIF at `finish()`, replaying at the run's pace. Works in scripts too.

```python
plot = LivePlot(range(steps), "loss | acc", record="run.gif")
for step in plot:
    plot.log(loss=train_step())
```

`record=True` keeps frames without writing; write later with [`plot.save_gif`][liveplot.LivePlot.save_gif]:

```python
plot = LivePlot(range(steps), record=True)
...
plot.save_gif("run.gif", speedup=2.0)     # twice as fast as the run
```

- `plot.frames`: `(seconds since creation, PNG bytes)` pairs.
- `plot.last_png`: the latest frame.
- One frame per redraw, so `refresh_seconds` sets the frame count.
