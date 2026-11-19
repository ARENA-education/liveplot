# Smoothing and log axes

## Smoothing

Per-step losses are noisy. `set_smooth` draws a smoothed curve and keeps the raw values as a faded line behind it.

![noisy loss curves with a smoothed line drawn over the faded raw values](../smoothing.gif)

```python
plot.set_smooth(0.9)                  # every panel
plot.panels[0].set_smooth(0.9)        # one panel
plot["loss"].set_smooth(0.9)          # the panel that holds loss
plot = LivePlot(loader, smooth=0.9)   # the default for every panel, set at creation
```

- The weight is a number from 0 up to, but not including, 1. Larger is smoother. 0 turns smoothing off.
- Smoothing belongs to a panel. All curves on the panel are smoothed, on both y-axes.
- `plot.data` always holds the raw values. Smoothing only changes what is drawn.

The method is wandb's default, the [time-weighted exponential moving average](https://docs.wandb.ai/models/app/features/panels/line-plot/smoothing), with the same meaning of the weight as wandb's smoothing slider.

As in wandb, the amount of smoothing depends on how the points are spread over the x range of the plot, not on how many points there are. By default the x range is the range of the data so far, so a short run with few points is barely smoothed, and the smoothed curve can shift a little as the run grows. With `fixed_x=True` and a `total`, the x range is the whole run from the start.

## Log axes

```python
plot["loss"].set_yscale("log")        # the axis that holds loss
plot.set_yscale("log")                # the left axis of every panel
```

The scale is `"linear"` or `"log"`. Only the y-axis has a scale setting.
