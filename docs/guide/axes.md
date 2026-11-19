# Customising axes

Titles, labels, limits and scales are set with matplotlib's method names. You can call the setters before the loop or during it. The figure is laid out again on the next frame.

![a panel with a title, a fixed right-hand axis and a log-scale panel](../axes_api.gif)

## Which object to call

| Object | How to get it | What its setters change |
|---|---|---|
| An axis | `plot["acc"]` | One y-axis: the one holding that metric, left or right |
| A panel | `plot.panels[0]`, `plot["acc"].panel` | One panel, and its left y-axis |
| The plot | `plot` | Every panel, including panels created later |

```python
plot = LivePlot(loader, "loss | acc", "lr")

plot["acc"].set_ylim(0, 1)                   # the right-hand axis of the first panel
plot["acc"].set_ylabel("test accuracy")
plot["lr"].set_yscale("log")

plot.panels[0].set_title("training")         # one panel
plot.panels[0].set_xlim(0, 10_000)

plot.set_xlabel("batches")                   # every panel
```

`plot[metric]` finds the axis whichever side it is on, so a right-hand axis needs no special spelling. `plot.panels[i].left` and `plot.panels[i].right` name a panel's two axes directly.

## The setters

On an axis ([reference](../api.md#liveplot.liveplot._Axis)):

| Setter | Effect |
|---|---|
| `set_ylabel(text)` | The y label |
| `set_ylim(low, high)` | Fix the y range. Also accepts one `(low, high)` pair. Both limits are required. |
| `set_yscale("log")` | `"linear"` or `"log"` |
| `axhline(y, label)` | A horizontal [reference line](reference-lines.md) |

On a panel ([reference][liveplot.liveplot.Panel]):

| Setter | Effect |
|---|---|
| `set_title(text)` | The title |
| `set_xlabel(text)` | The x label |
| `set_xlim(low, high)` | Fix the x range |
| `set_smooth(weight)` | [Smoothing](smoothing.md) for the panel's curves |
| `legend(**kwargs)` | Move or restyle the legend. The keyword arguments go to matplotlib's `Axes.legend`. |
| `axvline(x, label)` | A vertical [reference line](reference-lines.md) |

The panel setters are also available on an axis, where they act on the axis's panel. So `plot["acc"].set_title("evaluation")` works. The y setters are also available on a panel, where they act on the left axis.

On the plot, `set_title`, `set_xlabel`, `set_xlim`, `set_ylabel`, `set_ylim`, `set_yscale` and `set_smooth` apply to every panel. The y setters reach left axes only.

## Several settings at once

`set(**kwargs)` calls several setters in one go, like matplotlib's `Axes.set`. Each keyword `name=value` calls `set_name(value)`:

```python
plot["acc"].set(ylabel="test accuracy", ylim=(0, 1))
plot.panels[0].set(title="training", xlabel="examples")
plot.set(xlabel="examples", smooth=0.9)       # every panel
```

## Moving the legend

```python
plot.panels[0].legend(loc="upper left")
plot.panels[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=3)   # under the panel
```

## Limits and scales that liveplot does not support

`set_ylim` and `set_xlim` need both limits. `set_yscale` accepts `"linear"` and `"log"` only. For anything else, build a matplotlib figure with [`plot.figure()`][liveplot.LivePlot.figure] and change it there.
