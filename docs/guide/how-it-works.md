# How it works

## A separate render process

`plot.log` only appends numbers to lists and puts them on a queue. It does no drawing.

A separate process owns the matplotlib figure. It redraws at most once every `refresh_seconds` (1.0 by default), and only when new data has arrived. Points that arrive between redraws are drawn together in the next frame. `refresh_seconds=0` redraws whenever new data arrives, as fast as rendering allows.

Each frame comes back as PNG bytes and replaces the image in the notebook's output cell. The output is a plain image. It needs no widgets and no JavaScript, so it behaves the same in Jupyter, Colab, VS Code and Cursor.

A small background thread shows each frame as soon as it is ready, so a loop that logs rarely still sees its frames on time.

If the render process cannot start, liveplot warns and draws inside `log` instead. `plot.mode` tells you which is in use: `"process"`, `"thread"`, or `"off"` when nothing is drawn.

## The progress bar

The bar is tqdm's (`tqdm.auto`), shown under the plot. Its postfix shows the latest value of each metric. Pass an existing tqdm bar as the iterable to use yours instead. `progress=False` turns the bar off.

## Interrupting a cell

Interrupting is safe. Jupyter sends the interrupt to every process the kernel started, and the render process ignores it. You are left with the last frame, and `plot.data` holds everything logged so far. Running the cell again starts a new plot.

A plot that is dropped without `finish()` stops its render process when it is garbage collected. The render process also exits on its own if the notebook kernel dies.

## Outside a notebook

In a plain script nothing is drawn, and no message is shown. `plot.data` and `plot.latest` are still filled in, the progress bar still works, and [`plot.figure()`][liveplot.LivePlot.figure] and [recording](saving.md#recording-a-gif) still work.

## Dependencies

`import liveplot` imports nothing heavy. matplotlib, numpy and pillow are imported when they are first needed. IPython and tqdm are optional: without IPython nothing is displayed, and without tqdm there is no progress bar.
