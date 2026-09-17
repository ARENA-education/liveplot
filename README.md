# liveplot

Live training curves in Jupyter, Colab and the VS Code / Cursor interactive window, with a tqdm bar underneath, at (almost) no cost to the training loop.

[![docs](https://img.shields.io/badge/docs-arena--education.github.io%2Fliveplot-blue)](https://arena-education.github.io/liveplot/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ARENA-education/liveplot/blob/demo/examples/demo.ipynb) [![tests](https://github.com/ARENA-education/liveplot/actions/workflows/tests.yml/badge.svg)](https://github.com/ARENA-education/liveplot/actions/workflows/tests.yml)

![training loss every step, eval loss and accuracy every 50 steps, on one panel with two y-axes](docs/demo.gif)

```
pip install git+https://github.com/ARENA-education/liveplot.git
```

```python
from liveplot import LivePlot

plot = LivePlot(loader, "loss | acc", "lr")       # wrap the loop like tqdm; "|" puts acc on a right-hand axis
plot["acc"].set_ylim(0, 1)                         # configure like matplotlib
for batch in plot:
    loss, acc = train_step(batch)
    plot.log(loss=loss, acc=acc, lr=lr)            # log like wandb
```

It wraps iterables like tqdm, logs like wandb and is configured like matplotlib, so the names are ones you already know. Rendering happens in a separate process and the output is a plain image, so it behaves the same everywhere with no widgets or JavaScript, and interrupting a cell is safe.

**[Documentation](https://arena-education.github.io/liveplot/)**: the [guide](https://arena-education.github.io/liveplot/guide/) (nested loops, the x-axis, reference lines, smoothing, recording GIFs), the [API](https://arena-education.github.io/liveplot/api/), and the [examples](https://arena-education.github.io/liveplot/examples/). The tour in [`examples/demo.py`](examples/demo.py) runs cell by cell in the interactive window or [in Colab](https://colab.research.google.com/github/ARENA-education/liveplot/blob/demo/examples/demo.ipynb).

MIT. Per-panel option names and the grid rule are adapted from Tyler Lum's [live_plotter](https://github.com/tylerlum/live_plotter); see `THIRD_PARTY_LICENSES.md`.
