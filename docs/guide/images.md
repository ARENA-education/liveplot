# Images with `subplots` and `imshow`

A panel can hold a picture instead of curves: loss curves every step beside generated samples now and then.

![a loss curve beside a grid of generated samples](../imshow.gif)

## `LivePlot.subplots`

[`LivePlot.subplots`][liveplot.LivePlot.subplots] follows `plt.subplots`: it returns the plot and a fixed grid of empty panels, which you then fill.

```python
plot, (ax_loss, ax_samples) = LivePlot.subplots(1, 2, figsize=(11, 4))
ax_loss.plot("lossD", "lossG")            # these metrics on the left axis
ax_loss.twinx().plot("D_real")            # this one on the right
ax_samples.set_title("generator samples")

for epoch in range(epochs):
    for imgs, _ in plot(loader, desc=f"epoch {epoch}"):
        lossD, lossG, d_real = train_step(imgs)
        plot.log(lossD=lossD, lossG=lossG, D_real=d_real)                     # every step
        if plot.n % 250 == 0:
            ax_samples.imshow(netG(fixed_noise), rows=2, vmin=-1, vmax=1)     # replaces the last picture
plot.finish()
```

- `ax.plot(...)` takes metric names, not data; values come from `plot.log`.
- `ax.twinx()` is the right-hand axis.
- `figsize` is the whole figure, in inches. `width_ratios` / `height_ratios` as in matplotlib.
- Other kwargs go to `LivePlot` (`total`, `smooth`, …). For a single loop, pass `iterable=loader`.

The grid never grows. An unplaced metric joins the first empty panel, else the first curve panel.

### What `subplots` returns

matplotlib's squeeze rules:

| Grid | `axes` is |
|---|---|
| 1 x 1 | one panel |
| one row or column | a flat list |
| larger | a 2-d grid: `axes[0][1]`, `axes[0, 1]`, `axes.flat` |

`squeeze=False` always gives the 2-d grid.

## `plot.imshow`

For one picture, [`plot.imshow`][liveplot.LivePlot.imshow] adds an image panel on first call; later calls replace the picture.

```python
plot = LivePlot(range(steps))
for step in plot:
    plot.log(loss=train_step())
    if step % 100 == 0:
        plot.imshow(model(holdout))
```

On a `subplots` plot, call `imshow` on a panel instead.

## Accepted input

Torch tensors (detached and moved to CPU for you; torch is never imported), numpy arrays, array-likes.

| Shape | Read as |
|---|---|
| `(H, W)`, `(1, H, W)` | one grayscale image |
| `(H, W, 3)`, `(H, W, 4)` | one colour image, channels last |
| `(B, H, W)` | `B` grayscale images |
| `(B, 1|3|4, H, W)` | `B` images, channels first (torch) |
| `(B, H, W, 1|3|4)` | `B` images, channels last |

`(3, H, W)` and `(4, H, W)` are ambiguous and raise: pass `channels="first"` (one colour image) or `channels="none"` (grayscale batch). A grayscale batch of images 3 or 4 pixels wide also needs `channels="none"`.

## Grids

- `rows=2` or `cols=5` fixes one side; the other is inferred.
- `griddim=(rows, cols)` fixes both; blanks fill, extras are dropped.
- Default: near-square, at most `max_cols` (8) wide.
- At most `max_images` (64) shown, with a warning; `max_images=None` for all.
- Edge to edge by default; `padding=2` adds 2 px of `pad_value` around each.

## Scaling

Default: batch min is black, batch max white.

- `vmin` / `vmax` fix the range. Do this for live views, or brightness jumps between frames.
- `scale_each=True` scales each image alone.
- `uint8` is shown as is, unless `vmin` / `vmax` is given.

Unlike matplotlib's `imshow`, `vmin` / `vmax` apply to colour images too. For a `tanh` generator use `vmin=-1, vmax=1`.

## Step in the title

Image panel titles show the step the picture was drawn at, e.g. "generator samples (step 250)", so you can tell how old it is.

Full example: [`examples/dcgan_synthetic.py`](https://github.com/ARENA-education/liveplot/blob/main/examples/dcgan_synthetic.py), a fake DCGAN run.
