# How it works

## A separate render process

`plot.log` appends numbers and queues them; it never draws.

A separate process owns the matplotlib figure. It redraws at most every `refresh_seconds` (default 1.0), only when there's new data; points in between share a frame. `refresh_seconds=0` redraws on every arrival.

Each frame comes back as a PNG and replaces the image in the output cell. No widgets, no JavaScript, so Jupyter, Colab, VS Code and Cursor all behave the same. A background thread shows each frame as soon as it's ready, so rarely-logging loops still update on time.

If the render process can't start, liveplot warns and draws inside `log` instead. `plot.mode` is `"process"`, `"thread"`, or `"off"` (nothing drawn).

## Progress bar

tqdm's (`tqdm.auto`), under the plot, with the latest metrics as its postfix. Pass your own tqdm bar as the iterable to reuse it. `progress=False` turns it off.

## Interrupting

Safe. Jupyter interrupts every process the kernel started; the render process ignores it. You keep the last frame and `plot.data`. Rerunning the cell starts a new plot.

A plot dropped without `finish()` stops its render process when garbage-collected. The render process also exits if the kernel dies.

## Outside a notebook

Nothing is drawn, silently. `plot.data`, `plot.latest`, the bar, [`plot.figure()`][liveplot.LivePlot.figure] and [recording](saving.md#recording-a-gif) still work.

## Dependencies

`import liveplot` imports nothing heavy; matplotlib, numpy and pillow load on first use. IPython and tqdm are optional: without IPython nothing is shown, without tqdm there's no bar.
