# Reference lines

Reference lines mark a level a curve should reach, or a moment in the run. The method names are matplotlib's: `axhline` for a horizontal line and `axvline` for a vertical one.

![horizontal reference lines for uniform loss and unigram entropy, and a vertical line at a learning rate drop](../reference_lines.gif)

## Horizontal lines

Call `axhline` on the axis of the metric the line belongs with:

```python
plot["loss"].axhline(math.log(d_vocab), "uniform")       # the loss of a uniform guess
plot["return"].axhline(500, "solved", color="green")
```

- The line is dashed grey by default. Extra keyword arguments (`color`, `linestyle`, `linewidth`, `alpha`) go to matplotlib.
- The second argument is the label. It appears in the panel's legend. Without a label, the legend shows the number.
- On a panel with two y-axes, the line uses the scale of the axis you called it on. `plot["acc"].axhline(0.9)` is drawn against the right-hand axis if `acc` is on the right.

`plot.axhline(y, label)` on the plot itself draws the line on the left axis of every panel that exists at that moment. Panels created later do not get it. `plot.axhline(y, label, metric="loss")` is the same as `plot["loss"].axhline(y, label)`.

## Vertical lines

```python
plot.axvline(label="lr drop")                   # every panel, at the current step
plot.panels[1].axvline(2000, "checkpoint")      # one panel, at x = 2000
```

- Without an `x`, the line is drawn at the current step, [`plot.step`][liveplot.LivePlot.step], at the time of the call. So you can mark an event at the point in the loop where it happens.
- The line is dotted grey by default. Extra keyword arguments go to matplotlib.
- The label is written along the line. It does not appear in the legend.
- `plot.axvline` draws on every panel that holds curves, including panels created later. Image panels are skipped.

```python
for step in plot:
    if step == warmup_steps:
        plot.axvline(label="warmup ends")
    plot.log(loss=train_step())
```
