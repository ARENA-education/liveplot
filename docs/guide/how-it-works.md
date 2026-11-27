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

Nothing is drawn, silently, unless you serve it. `plot.data`, `plot.latest`, the bar, [`plot.figure()`][liveplot.LivePlot.figure] and [recording](saving.md#recording-a-gif) still work.

### In a browser: `serve`

`serve=True` draws the plot anyway and serves it as a web page. The link is printed when the plot is created; the tqdm bars stay in the terminal as usual.

```
$ python train.py
liveplot: live plot at http://localhost:8765/
epoch 0: 100%|██████████| 200/200 [00:04<00:00, 48.9step/s, loss=0.278, acc=0.528]
```

To serve a script without editing it, set `LIVEPLOT_SERVE`: `LIVEPLOT_SERVE=1 python train.py`. It is read only when `serve` isn't given.

| `serve=` / `LIVEPLOT_SERVE=` | |
|---|---|
| `True` / `1` | `localhost`, port 8765 or the next free one |
| `9000` | that port |
| `"0.0.0.0:9000"` | every interface: anyone who can reach the machine can see the plot |

Every plot in one process shares the server, and the page shows the newest one, so a script that trains several runs keeps one link. The server stops when the script exits; the page keeps the last frame. On a remote machine, VS Code and Cursor forward the port for you; otherwise `ssh -L 8765:localhost:8765 host`. `show=False` turns serving off too.

## Dependencies

`import liveplot` imports nothing heavy; matplotlib, numpy and pillow load on first use. IPython and tqdm are optional: without IPython nothing is shown, without tqdm there's no bar.
