# Examples

## The tour, in Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARENA-education/liveplot/blob/demo/examples/demo.ipynb)

[`examples/demo.py`](https://github.com/ARENA-education/liveplot/blob/main/examples/demo.py) walks through the features cell by cell: the one-liner, layout strings and metrics logged at different rates, nested epoch/batch loops, matplotlib-style configuration, your own tqdm bar, interrupts, recording, and reading the data back. CI converts it to a notebook on the `demo` branch on every push to `main`, which is what the badge opens. It also runs as a plain script, and cell by cell in the VS Code / Cursor interactive window.

## What it looks like

A tiny numpy classifier logging its training loss every step and its eval loss and accuracy every 50 steps, laid out as `"loss | eval_loss eval_acc"`. Recorded by the library itself with `record="demo.gif"`:

![demo](demo.gif)

## Recording your own

```python
plot = LivePlot(range(steps), "loss | acc", record="run.gif")
for step in plot:
    plot.log(loss=train_step())
    if step % 50 == 0:
        plot.log(acc=evaluate())
```

`record="run.gif"` writes an animated GIF at `finish()` that replays at the real pace of the run; `record=True` just keeps the frames for `plot.save_gif(path, speedup=2.0)` later. Recording works outside a notebook too, so a script can produce the GIF for a README.
