# Guide

Each page explains one feature, with examples. Start with the x-axis page if you are new. The others can be read in any order.

- [The x-axis and `total`](x-axis.md): x counts the items the loop has consumed, like tqdm's `n`. The three ways to advance it, `total` and `fixed_x`, `initial`, `unit` and `unit_scale`, and `si_xticks`.
- [Panels and layout strings](panels.md): one string per panel, names that share a y-axis, `|` for a right-hand axis, the grid, and the dict form.
- [Logging at different rates](logging.md): what `log` accepts, metrics logged every step beside others logged every epoch, reading the values back, and collecting without drawing.
- [Customising axes](axes.md): titles, labels, limits and scales, set with matplotlib's method names, and which object to call them on.
- [Reference lines](reference-lines.md): `axhline` for a level to reach and `axvline` for a moment in the run.
- [Images with `subplots` and `imshow`](images.md): generated samples beside the loss curves, in one figure.
- [Smoothing and log axes](smoothing.md): a smoothed curve with the raw values faded behind it, and log-scale y-axes.
- [Saving and recording](saving.md): a matplotlib figure of the current state, and a GIF of the whole run.
- [How it works](how-it-works.md): the render process, the progress bar, interrupting a cell, and running outside a notebook.

The [API reference](../api.md) lists every class and method.
