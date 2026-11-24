# API reference

Generated from the docstrings in [`liveplot/liveplot.py`](https://github.com/ARENA-education/liveplot/blob/main/liveplot/liveplot.py). One public class, [`LivePlot`][liveplot.LivePlot]; panels and axes are handles you get from it:

| Object | Get it with |
|---|---|
| [`LivePlot`][liveplot.LivePlot] | `LivePlot(...)`, `LivePlot.subplots(...)` |
| [`Panel`][liveplot.liveplot.Panel] | `plot.panels[i]`, `plot[metric].panel`, `LivePlot.subplots` |
| [Axis](#liveplot.liveplot._Axis) | `plot[metric]`, `panel.left`, `panel.right`, `panel.twinx()` |

## LivePlot

::: liveplot.LivePlot
    options:
      heading_level: 3

## Panel

::: liveplot.liveplot.Panel
    options:
      heading_level: 3

## Axis

`_Axis` in the source; never created directly.

::: liveplot.liveplot._Axis
    options:
      heading_level: 3

## Panel grid

`_PanelGrid` in the source; what `LivePlot.subplots` returns for grids of at least 2 x 2.

::: liveplot.liveplot._PanelGrid
    options:
      heading_level: 3
