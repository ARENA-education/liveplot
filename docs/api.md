# API reference

This page is generated from the docstrings in [`liveplot/liveplot.py`](https://github.com/ARENA-education/liveplot/blob/main/liveplot/liveplot.py).

The library has one public class, [`LivePlot`][liveplot.LivePlot]. Its panels and axes are small handle objects that you get from the plot:

| Object | How to get it |
|---|---|
| [`LivePlot`][liveplot.LivePlot] | `LivePlot(...)` or `LivePlot.subplots(...)` |
| [`Panel`][liveplot.liveplot.Panel] | `plot.panels[i]`, `plot[metric].panel`, or the second value returned by `LivePlot.subplots` |
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

The class is named `_Axis` in the source. You never create one yourself.

::: liveplot.liveplot._Axis
    options:
      heading_level: 3

## Panel grid

The class is named `_PanelGrid` in the source. `LivePlot.subplots` returns one for a grid with at least two rows and two columns.

::: liveplot.liveplot._PanelGrid
    options:
      heading_level: 3
