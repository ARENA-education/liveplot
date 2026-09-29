"""
Live training curves that cost the training loop (almost) nothing.

    from liveplot import LivePlot

    for step in LivePlot(range(num_steps)):          # a tqdm bar + a live plot, in one
        loss, acc = train_step()
        plot.log(loss=loss, acc=acc)                 # -> hmm, where does `plot` come from? see below

The ways to hold on to the plot object:

    plot = LivePlot(range(num_steps))                # 1. tqdm style: iterate it
    for step in plot:
        plot.log(loss=train_step())                  #    x is implicit: the iteration count

    plot = LivePlot("loss", "acc", total=epochs * len(loader))
    for epoch in range(epochs):                      # 2. nested loops: wrap the inner one, the
        for imgs, labels in plot(loader, desc=f"epoch {epoch}"):   # count (and x) continues
            plot.log(loss=train_step(imgs, labels))  #    across epochs; one tqdm bar per epoch
        plot.log(acc=evaluate())
    plot.finish()                                    #    (or use `with LivePlot(...) as plot:`)

    with LivePlot(total=num_steps) as plot:          # 3. you own the loop and the x values
        for step in range(num_steps):
            plot.log(step, loss=train_step())        #    x is explicit

The x-axis follows tqdm: it counts items consumed and never looks at their values.
x = initial + n * unit_scale, where n is 0 inside the body for the first item (like
`for step in range(N)`), `initial` shifts the start (tqdm's argument of the same
name; use initial=1 for 1-based, initial=10 to plot range(10, 20) at its values),
and `unit` / `unit_scale` relabel and rescale it (e.g. unit="examples",
unit_scale=batch_size plots against examples seen, and the tqdm bar shows the same
numbers). `total` (in items, like tqdm's) fixes the x range; the single-loop form
takes it from len(iterable). An explicit `plot.log(step, ...)` uses that x for that
call only, like wandb's `step=`.

Metrics are discovered from what you log: with no layout given, every metric goes
on ONE panel with a legend. To split them up, give panel strings:

    LivePlot(range(N), "loss", "return | entropy", "lossD lossG | acc")

Each string is one panel. Names separated by spaces share the left y-axis; a `|`
puts the names after it on a right-hand y-axis. Metrics you log that no string
mentions get a panel of their own. Every panel has a legend. For labels or fixed
ranges, use a dict instead of a string:

    {"metrics": ["acc"], "ylim": (0, 1), "ylabel": "test accuracy", "xlabel": "epoch"}

(the dict form; the setters below are the nicer way). The plot is the figure and its
panels are the Axes, configured with matplotlib's own names and signatures, before
or during the loop:

    plot["acc"].set_ylim(0, 1)                 # the y-axis holding a metric (left or right)
    plot["loss"].set_ylim(bottom=0)            # one end fixed, the other follows the data
    plot["acc"].set_ylabel("test accuracy", fontsize=9)
    plot["loss"].axhline(0.1, color="red", label="target")   # a label puts it in the legend
    plot.axes[0].set_title("training")         # plot.axes, like Figure.axes
    plot.axes[0].axvline(plot.step, label="lr drop")
    plot.axes[0].right.set_yscale("log")       # .left / .right are the panel's y-axes
    plot.suptitle("run 3")                     # Figure.suptitle / supxlabel / supylabel
    plot.set_all(xlabel="examples", smooth=0.9)   # every panel, including ones made later

Available on an axis: plot, set_ylabel, set_ylim, set_yscale, axhline, set(**kw),
lines; on a panel: set_title, set_xlabel, set_xlim, set_xscale, axvline, legend,
twinx, imshow, set_smooth, set(**kw), plus the left axis's setters. Arguments are
matplotlib's (text kwargs, scale kwargs, one-sided limits, Line2D kwargs on
reference lines), and each call is tried on a scratch matplotlib Axes first, so a
mistake raises matplotlib's own error where it was made. Smoothing is wandb's
time-weighted EMA with the same 0 to 1 weight (`smooth=0.9`), raw values faded
behind.

Images beside curves: `LivePlot.subplots` mirrors `plt.subplots` (numpy array of
panels, sharex / sharey, width_ratios, gridspec_kw, subplot_kw), and a panel can
hold a picture instead of lines -- loss curves updating every step next to samples
from a generator, in one figure:

    plot, (ax_loss, ax_samples) = LivePlot.subplots(1, 2, total=n_steps, figsize=(11, 4))
    ax_loss.plot("lossD")                      # a metric's name where matplotlib takes data
    ax_loss.plot("lossG", "--", label="generator")   # fmt strings and Line2D kwargs, as in matplotlib
    ax_loss.twinx().plot("D(x)")               # matplotlib's spelling for a right-hand axis
    ...
    ax_samples.imshow(netG(noise), rows=2, vmin=-1, vmax=1)   # replaces the last one

`ax.plot("loss")` is matplotlib's `ax.plot("loss", data=d)` with the data supplied
later by `log()`; like matplotlib it takes one series per call and returns `[line]`
(two names would mean x and y there, so they are an error here).

`plot.imshow(x)` does the same on a plot of its own, making the panel on first use.
A batch is tiled for you (`rows` / `cols` / `griddim=(r, c)`, padding or dropping
to fit, with make_grid's 2-pixel gaps); (H, W), (C, H, W), (H, W, C), (B, H, W), (B, C, H, W) and (B, H, W, C) are
all understood, and the two genuinely ambiguous shapes, (3, H, W) and (4, H, W),
raise and name the `channels=` to pass. Values are scaled to the batch's full range
unless `vmin` / `vmax` fix it (worth doing live: otherwise the black point moves
every frame), or `scale_each=True` scales each image alone, as make_grid does.
Unlike matplotlib, which ignores vmin/vmax for colour data and clips it to [0, 1],
these apply to colour images too.

How it works: the training thread only appends numbers (~40 us per `log`). A
separate *render process* owns the matplotlib figure, redraws it at most once per
`refresh_seconds` (default 1.0; points arriving in between are batched into the
next frame; 0 means redraw on every arrival, as fast as rendering allows), and
sends back PNG bytes that get swapped into a fixed output cell. The output is a plain image, so it behaves the same in Jupyter, Colab, VS
Code and Cursor: no widgets, no CDN, no JavaScript. The progress bar is tqdm
(`tqdm.auto`), shown below the plot with the latest logged values as its postfix;
pass an existing tqdm object as the iterable to reuse yours instead.

Frames are shown by a small background thread as soon as the renderer produces
them, so a loop that logs rarely still sees each frame when it is ready.

Interrupts: the render process ignores SIGINT (Jupyter's "interrupt kernel" is sent
to the whole process group), exits by itself if the notebook process dies, and is
terminated when the LivePlot object is garbage collected, so an interrupted cell
leaves a frozen plot with `plot.data` intact and no stray process. Outside a
notebook nothing is drawn; `plot.data` still collects everything.

`plot.figure()` returns a matplotlib Figure of the current state, for saving or tweaking.

Recording: `LivePlot(..., record=True)` keeps every rendered frame in `plot.frames`
(as PNG bytes with timestamps) and `plot.save_gif("run.gif")` stitches them into an
animated GIF that replays at the real pace; `record="run.gif"` does that at
`finish()`. Recording also works outside a notebook, so a script can produce the GIF.

This module deliberately imports nothing heavy (no torch): the render process
imports it afresh, so keeping it light keeps the renderer's startup ~1 s.

The per-panel label/limit options and the automatic grid-shape rule are borrowed
from Tyler Lum's `live_plotter` (https://github.com/tylerlum/live_plotter, MIT
License, Copyright (c) 2023 Tyler Lum) -- see THIRD_PARTY_LICENSES.md.
"""

from __future__ import annotations

import io
import math
import multiprocessing as mp
import os
import queue
import signal
import sys
import threading
import time
import warnings
import weakref

_PANEL_KEYS = {"title", "metrics", "secondary", "xlabel", "ylabel", "ylabel2", "xlim", "ylim", "ylim2", "axhlines", "axhlines2",
               "axvlines", "smooth", "xscale", "yscale", "yscale2", "kind", "cmap", "legend", "styles", "kw"}
_REF_LINE_STYLE = {"linestyle": "--", "linewidth": 1, "color": "0.45"}  # defaults for axhline / axvline artists


def _normalise_axhlines(spec) -> list:
    """
    Reference lines for a panel, in any of these forms, -> a list of matplotlib `Axes.axhline` kwargs:
        {"uniform": 10.8, "unigram": 7.35}           label -> y
        [10.8, 7.35]                                  y values (labelled with the number)
        [dict(y=10.8, label="uniform", color="red")]  full matplotlib kwargs per line
    """
    if not spec:
        return []
    if isinstance(spec, dict):
        return [{"y": float(y), "label": str(label)} for label, y in spec.items()]
    out = []
    for item in spec:
        if isinstance(item, dict):
            assert "y" in item, f"axhline needs a y: {item!r}"
            out.append({**item, "y": float(item["y"]), "label": str(item.get("label", f"{float(item['y']):g}"))})
        else:
            out.append({"y": float(item), "label": f"{float(item):g}"})
    return out


# --------------------------------------------------------------------------- panel specs


def _parse_panel_string(spec: str) -> dict:
    """'lossD lossG | acc'  ->  {"metrics": ["lossD", "lossG"], "secondary": ["acc"]}"""
    left, _, right = spec.partition("|")
    assert "|" not in right, f"at most one '|' per panel: {spec!r}"
    return {"metrics": left.split(), "secondary": right.split()}


def _check_smooth(weight):
    """A smoothing weight is wandb's: in [0, 1), where 0 means off. None means 'no opinion'."""
    assert weight is None or 0 <= weight < 1, f"smooth must be a weight in [0, 1), got {weight!r}"
    return weight


def _normalise_panel(spec, allow_empty: bool = False) -> dict:
    """`allow_empty` is for the panels `LivePlot.subplots` hands out before anything is drawn on them."""
    if isinstance(spec, str):
        spec = _parse_panel_string(spec)
    unknown = set(spec) - _PANEL_KEYS
    assert not unknown, f"unknown panel keys {sorted(unknown)}; allowed: {sorted(_PANEL_KEYS)}"
    kind = spec.get("kind", "curve")
    assert kind in ("curve", "image"), f'kind must be "curve" or "image", got {kind!r}'
    metrics, secondary = list(spec.get("metrics", [])), list(spec.get("secondary", []))
    assert metrics or secondary or kind == "image" or allow_empty, f"a panel needs at least one metric: {spec!r}"
    for lim in ("xlim", "ylim", "ylim2"):
        if spec.get(lim) is not None:
            assert len(spec[lim]) == 2, f"{lim} must be a (low, high) pair (either may be None)"
    return {
        "title": spec.get("title") or " / ".join(metrics + secondary),
        "metrics": metrics,
        "secondary": secondary,
        "xlabel": spec.get("xlabel"),  # None -> the plot-wide unit ("step" by default)
        "ylabel": spec.get("ylabel"),
        "ylabel2": spec.get("ylabel2"),
        "xlim": spec.get("xlim"),
        "ylim": spec.get("ylim"),
        "ylim2": spec.get("ylim2"),
        "axhlines": _normalise_axhlines(spec.get("axhlines")),  # reference lines on the left axis (see _normalise_axhlines)
        "axhlines2": _normalise_axhlines(spec.get("axhlines2")),  # ... and on the right axis
        "axvlines": [dict(kw) for kw in spec.get("axvlines") or []],  # vertical lines on this panel only (axvline kwargs)
        "smooth": _check_smooth(spec.get("smooth")),  # TWEMA weight in [0, 1); None = plot-wide default; 0 = off
        "xscale": spec.get("xscale", "linear"),  # any matplotlib scale name: "linear", "log", "symlog", ...
        "yscale": spec.get("yscale", "linear"),  # ... left axis
        "yscale2": spec.get("yscale2", "linear"),  # ... right axis
        "kind": kind,  # "curve" (metric lines) or "image" (one picture, overwritten in place)
        "cmap": spec.get("cmap", "gray"),  # image panels only; matplotlib ignores it for RGB data
        "legend": dict(spec.get("legend") or {}),  # Axes.legend kwargs (loc, ncols, bbox_to_anchor, ...)
        "styles": {k: dict(v) for k, v in (spec.get("styles") or {}).items()},  # metric -> Line2D kwargs, from ax.plot
        "kw": {k: dict(v) for k, v in (spec.get("kw") or {}).items()},  # extra matplotlib kwargs for the setters:
        # "title", "xlabel", "ylabel", "ylabel2" (Text kwargs), "xscale", "yscale", "yscale2", "imshow"
    }


_TWEMA_VIEWPORT_SCALE = 1000  # x is measured in thousandths of the plotted range, as in wandb


def _twema(xs, ys, weight: float, x_range: float):
    """
    wandb's default smoothing, the time-weighted exponential moving average, exactly as documented at
    docs.wandb.ai/models/app/features/panels/line-plot/smoothing: the smoothing weight is
    min(sqrt(param), 0.999); each step decays the running sum by weight ** dx, where dx is the gap to
    the previous point in thousandths of the x-range (so the result depends on where points sit, not
    on how many there are); a debiasing accumulator stops early values leaning towards zero.
    """
    import numpy as np

    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    w = min(math.sqrt(max(weight, 0.0)), 0.999)
    dx = np.diff(x, prepend=x[0]) / max(x_range, 1e-12) * _TWEMA_VIEWPORT_SCALE
    decay = w**dx
    out, last, debias = np.empty_like(y), 0.0, 0.0
    for i in range(len(y)):
        last = last * decay[i] + y[i]
        debias = debias * decay[i] + 1.0
        out[i] = last / debias
    return out


def _grid_shape(n_panels: int, max_cols: int | None, rows: int | None, cols: int | None) -> tuple[int, int]:
    """
    Rows x cols for `n_panels` panels. Explicit `rows` / `cols` win; otherwise the grid is at
    most `max_cols` wide; with neither, use the near-square rule from live_plotter's
    `compute_n_rows_n_cols` (rows = ceil(sqrt(n)), cols = ceil(n / rows)).
    """
    assert n_panels > 0
    if rows is not None and cols is not None:
        assert rows * cols >= n_panels, f"{rows}x{cols} grid can't hold {n_panels} panels"
        return rows, cols
    if rows is not None:
        return rows, math.ceil(n_panels / rows)
    if cols is not None:
        return math.ceil(n_panels / cols), cols
    if max_cols is not None:
        cols = min(max_cols, n_panels)
        return math.ceil(n_panels / cols), cols
    rows = math.ceil(math.sqrt(n_panels))
    return rows, math.ceil(n_panels / rows)


# --------------------------------------------------------------------------- the figure


class _FigureRenderer:
    """
    Owns one matplotlib figure and the metric history it draws. Built on the
    object-oriented API (no pyplot), so it has no global state and works the
    same inside the render process or on the calling thread. `set_layout`
    rebuilds the figure for a new panel list, keeping the history.
    """

    def __init__(self, panels, xlim, layout, hist=None, images=None):
        # xlim = default x range or None; layout = (max_cols, rows, cols, cell_size, dpi, xlabel[, figure_kw])
        self.xlim, self.layout = xlim, layout
        self.hist = hist if hist is not None else {}
        self.images = dict(images) if images else {}  # panel index -> (the uint8 array it is showing, its x)
        self.set_layout(panels)

    def set_layout(self, panels):
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.cbook import normalize_kwargs
        from matplotlib.figure import Figure
        from matplotlib.lines import Line2D

        max_cols, rows_opt, cols_opt, cell_size, dpi, xlabel, *rest = self.layout
        figure_kw = rest[0] if rest else {}  # subplots()' grid options, and suptitle / supxlabel / supylabel
        gridspec_kw = dict(figure_kw.get("gridspec_kw") or {})
        n = len(panels)
        rows, cols = _grid_shape(n, max_cols, rows_opt, cols_opt)
        self.panels = panels
        fig_width = cell_size[0] * cols
        # One row with pictures in it, and no width_ratios chosen: size each image panel to its picture
        # at the row's height, and give the curves what is left, so neither is framed in white space.
        self.fitted_aspects = None
        if rows == 1 and "width_ratios" not in gridspec_kw and any(p["kind"] == "image" for p in panels):
            widths, self.fitted_aspects = self._fitted_widths(panels, cell_size, fig_width)
            fig_width = sum(widths)
            gridspec_kw = {**gridspec_kw, "width_ratios": widths + [widths[-1]] * (cols - n)}
        self.fig = Figure(figsize=(fig_width, cell_size[1] * rows))
        FigureCanvasAgg(self.fig)
        self.dpi = dpi
        axes = self.fig.subplots(rows, cols, squeeze=False, gridspec_kw=gridspec_kw,
                                 sharex=figure_kw.get("sharex", False), sharey=figure_kw.get("sharey", False)).flatten()
        for method in ("suptitle", "supxlabel", "supylabel"):
            if figure_kw.get(method):
                text, kw = figure_kw[method]
                getattr(self.fig, method)(text, **kw)
        palette = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        self.lines = {}
        self.raw_lines = {}  # metric -> faded line of the unsmoothed values, on smoothed panels
        self.smooth = {}  # metric -> TWEMA weight in (0, 1) or None
        self.limits = {}  # axes -> (xlim, ylim) as the user set them: None, (lo, hi), or one-sided with a None
        self.panel_titles = {i: (p["title"], p["kw"].get("title", {})) for i, p in enumerate(panels)}
        self.panel_imshow = {i: {"cmap": p["cmap"], "interpolation": "nearest", **p["kw"].get("imshow", {})}
                             for i, p in enumerate(panels)}
        self.image_axes = {}  # panel index -> its Axes, for the image panels
        self.image_artists = {}  # ... and the AxesImage drawn on it, so a redraw can set_data in place
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            kw = panel["kw"]
            if panel["kind"] == "image":
                ax.set_title(panel["title"], **kw.get("title", {}))
                ax.set_axis_off()  # pixel indices along the edge of a tiled grid are just noise
                self.image_axes[i] = ax
                continue
            names = panel["metrics"] + panel["secondary"]
            ax2 = ax.twinx() if panel["secondary"] else None
            ax.set_xscale(panel["xscale"], **kw.get("xscale", {}))
            ax.set_yscale(panel["yscale"], **kw.get("yscale", {}))
            if ax2 is not None:
                ax2.set_yscale(panel["yscale2"], **kw.get("yscale2", {}))
            window = panel["smooth"] if panel["smooth"] else None  # TWEMA weight in (0, 1)
            assert window is None or 0 < window < 1, f"smooth must be a weight in (0, 1), got {window!r}"
            for j, name in enumerate(names):
                target = ax2 if name in panel["secondary"] else ax
                style = {"color": palette[j % len(palette)], "linewidth": 1.2, "label": name,
                         **normalize_kwargs(panel["styles"].get(name, {}), Line2D)}
                if window:
                    (self.raw_lines[name],) = target.plot([], [], color=style["color"], linewidth=0.8, alpha=0.25,
                                                          linestyle=style.get("linestyle", "-"), label="_raw")
                (self.lines[name],) = target.plot([], [], **style)
                self.smooth[name] = window
                self.hist.setdefault(name, ([], []))
            for target, key in ((ax, "axhlines"), (ax2, "axhlines2")):
                for line_kw in panel[key] if target is not None else []:
                    target.axhline(**{**_REF_LINE_STYLE, **line_kw})
            for line_kw in panel["axvlines"]:
                ax.axvline(**{**_REF_LINE_STYLE, "linestyle": ":", **line_kw})
            ax.set_title(panel["title"], **kw.get("title", {}))
            ax.set_xlabel(panel["xlabel"] or xlabel, **kw.get("xlabel", {}))
            xlim = panel["xlim"] or self.xlim
            self.limits[ax] = (xlim, panel["ylim"])
            if ax2 is not None:
                ax.set_ylabel(panel["ylabel"] or ", ".join(panel["metrics"]), **kw.get("ylabel", {}))
                ax2.set_ylabel(panel["ylabel2"] or ", ".join(panel["secondary"]), **kw.get("ylabel2", {}))
                self.limits[ax2] = (xlim, panel["ylim2"])
            elif panel["ylabel"]:
                ax.set_ylabel(panel["ylabel"], **kw.get("ylabel", {}))
            for target in (ax, ax2):
                if target is not None:
                    self._apply_limits(target)
            # One legend per panel, for the curves on both y-axes and the labelled reference lines, as
            # matplotlib collects it: an artist whose label starts with "_" is left out. The metrics come
            # first, then the reference lines.
            handles, labels = ax.get_legend_handles_labels()
            if ax2 is not None:
                handles2, labels2 = ax2.get_legend_handles_labels()
                handles, labels = handles + handles2, labels + labels2
            curves = set(map(id, self.lines.values()))
            order = sorted(range(len(handles)), key=lambda k: id(handles[k]) not in curves)
            if handles:  # (nothing logged yet, or the "waiting for data" placeholder)
                ax.legend([handles[k] for k in order], [labels[k] for k in order],
                          **{"loc": "best", "fontsize": 8, **panel["legend"]})
        for ax in axes[n:]:
            ax.set_visible(False)
        self.axes = list(axes[:n])
        for index in self.images:  # a re-layout rebuilds the figure; put the pictures back
            if index in self.image_axes:
                self._draw_image(index)
        self.fig.tight_layout(pad=0.6)

    def _apply_limits(self, ax, autoscale: bool = False):
        """
        The user's limits on `ax`. Both ends given: fixed, and never autoscaled. One end given, as in
        matplotlib's `set_ylim(bottom=0)`: the other end follows the data, so autoscale first and then
        pin the given end.
        """
        xlim, ylim = self.limits[ax]
        whole = [lim is not None and None not in lim for lim in (xlim, ylim)]
        if autoscale:
            ax.relim()
            ax.autoscale_view(scalex=not whole[0], scaley=not whole[1])
        for lim, full, setter in ((xlim, whole[0], ax.set_xlim), (ylim, whole[1], ax.set_ylim)):
            if lim is not None:
                setter(*lim) if full else setter(*lim, auto=None)  # auto=None: keep autoscaling the free end

    def _fitted_widths(self, panels, cell_size, fig_width):
        """
        Panel widths in inches for a one-row figure: an image panel is as wide as its picture is at the
        row's height (less room for the title), the curve panels share the rest of the figure's width,
        or keep their own if the pictures leave too little. Also returns the aspect ratio each image
        panel was sized for, so a picture of another shape can trigger a re-layout.
        """
        cell_w, cell_h = cell_size
        aspects = {}
        for i, panel in enumerate(panels):
            if panel["kind"] == "image":
                arr = self.images.get(i, (None,))[0]
                aspects[i] = None if arr is None else arr.shape[1] / arr.shape[0]
        image_h = cell_h - 0.55  # the title above, and tight_layout's margins
        widths = [0.2 + image_h * aspects[i] if aspects.get(i) else cell_w for i in range(len(panels))]
        curves = [i for i, panel in enumerate(panels) if panel["kind"] != "image"]
        if curves:
            spare = (fig_width - sum(widths[i] for i in aspects)) / len(curves)
            for i in curves:
                widths[i] = max(spare, 0.6 * cell_w)
        return widths, aspects

    def add_image(self, index: int, arr, x=None):
        self.images[index] = (arr, x)
        if index in self.image_axes:
            self._draw_image(index)

    def _draw_image(self, index: int):
        arr, x = self.images[index]
        if self.fitted_aspects is not None and self.fitted_aspects.get(index) != arr.shape[1] / arr.shape[0]:
            return self.set_layout(self.panels)  # a picture of a new shape: re-fit the panel widths (draws it too)
        ax = self.image_axes[index]
        if x is not None:  # say how old the picture is: it is only redrawn when imshow is called
            (title, title_kw), when = self.panel_titles[index], f"{self.layout[5]} {x:g}"
            ax.set_title(f"{title} ({when})" if title else when, **title_kw)
        artist = self.image_artists.get(index)
        if artist is not None and artist.get_array().shape == arr.shape:
            artist.set_data(arr)  # same size as last time: no new artist, no rescale
            return
        if artist is not None:
            artist.remove()
        self.image_artists[index] = ax.imshow(arr, **self.panel_imshow[index])

    def add(self, step, metrics: dict):
        for name, value in metrics.items():
            xs, ys = self.hist.setdefault(name, ([], []))
            xs.append(step)
            ys.append(float(value))

    def render(self) -> bytes:
        for name, line in self.lines.items():
            xs, ys = self.hist[name]
            if self.smooth[name] and len(ys) > 1:
                self.raw_lines[name].set_data(xs, ys)
                x_range = (self.xlim[1] - self.xlim[0]) if self.xlim else (max(xs) - min(xs))
                line.set_data(xs, _twema(xs, ys, self.smooth[name], x_range))
            else:
                line.set_data(xs, ys)
        for ax in self.limits:
            self._apply_limits(ax, autoscale=True)
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", dpi=self.dpi)
        return buf.getvalue()


# --------------------------------------------------------------------------- the render process


def _render_worker(panels, inbox, outbox, refresh_seconds, layout, xlim, parent_pid):
    """
    Loop: collect messages from `inbox`, redraw at most once per `refresh_seconds`
    while there is new data, put PNG bytes on `outbox`. Messages: ("data", step,
    metrics); ("layout", panels, layout) to rebuild the figure; None to finish (draw one
    last frame, put None, exit). Exits on its own if the parent process is gone.
    """
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Jupyter interrupts the whole process group; not our business
    renderer = _FigureRenderer(panels, xlim, layout)
    dirty, last_draw, running = False, 0.0, True
    while running:
        if os.getppid() != parent_pid:  # notebook kernel died or restarted: nobody is listening
            return
        now = time.monotonic()
        if dirty and now - last_draw >= refresh_seconds:
            outbox.put(renderer.render())  # points that arrive during this render go into the next frame
            dirty, last_draw = False, time.monotonic()
            continue
        # Block until something arrives. With points pending, wake when the redraw floor is reached;
        # otherwise sleep (checking for a dead parent every half second). refresh_seconds=0 means
        # "redraw whenever there is new data" and must not spin while idle.
        wait = max(refresh_seconds - (now - last_draw), 0.001) if dirty else 0.5
        try:
            msg = inbox.get(timeout=wait)
        except queue.Empty:
            continue
        pending = [msg]
        while True:  # drain whatever else is queued so one redraw covers many steps
            try:
                pending.append(inbox.get_nowait())
            except queue.Empty:
                break
        for item in pending:
            if item is None:
                running = False
            elif item[0] == "layout":
                renderer.layout = item[2]  # grid options can change after start (subplots' width_ratios)
                renderer.set_layout(item[1])
                dirty = True
            elif item[0] == "image":
                renderer.add_image(*item[1:])
                dirty = True
            else:
                renderer.add(item[1], item[2])
                dirty = True
    if dirty:
        outbox.put(renderer.render())  # the final frame
    outbox.put(None)


def _terminate(proc, inbox):
    """Finalizer: shut the render process down without blocking (used when a LivePlot is dropped)."""
    if proc is not None and proc.is_alive():
        try:
            inbox.put_nowait(None)
        except Exception:  # noqa: BLE001
            pass
        proc.join(timeout=0.5)
        if proc.is_alive():
            proc.terminate()


def _collect_loop(plot_ref, outbox, proc, done):
    """
    Background thread of a LivePlot: show each frame as the renderer produces it (only the newest
    if several queued up); stop after the renderer's final None, when the renderer is gone, or when
    the LivePlot itself has been garbage collected.
    """
    try:
        while True:
            try:
                item = outbox.get(timeout=0.5)
            except queue.Empty:
                plot = plot_ref()
                if plot is None or plot.mode != "process" or not proc.is_alive():
                    break
                continue
            if item is None:
                break
            while True:
                try:
                    newer = outbox.get_nowait()
                except queue.Empty:
                    break
                if newer is None:
                    plot = plot_ref()
                    if plot is not None:
                        plot._show(item)
                    return
                item = newer
            plot = plot_ref()
            if plot is None:
                break
            plot._show(item)
            del plot  # don't hold the reference while blocked on the queue
    finally:
        done.set()


# --------------------------------------------------------------------------- panels and axes, addressed like matplotlib


def _lim(lo, hi, lo_alias=None, hi_alias=None):
    """
    matplotlib's limit arguments -> (lo, hi), either of which may be None to leave that end to the
    data: set_ylim(0, 1), set_ylim((0, 1)), set_ylim(bottom=0), set_ylim(top=1), set_ylim(ymin=0).
    Both None (a bare `set_ylim()`) -> None, i.e. autoscale again.
    """
    if hi is None and isinstance(lo, (tuple, list)):
        lo, hi = lo
    for name, value, alias in (("lower", lo, lo_alias), ("upper", hi, hi_alias)):
        assert value is None or alias is None, f"give the {name} limit once, not under two names"
    lo = lo if lo is not None else lo_alias
    hi = hi if hi is not None else hi_alias
    if lo is None and hi is None:
        return None
    return (None if lo is None else float(lo), None if hi is None else float(hi))


_FMT_LINESTYLES = ("--", "-.", "-", ":")
_FMT_MARKERS = set(".,ov^<>12348spP*hH+xXDd|_")
_FMT_COLORS = set("bgrcmykw")


def _parse_fmt(fmt: str) -> dict | None:
    """
    matplotlib's plot format string, '[marker][line][color]' in any order ("r--", "o-", "C2:", "k"),
    or a lone colour of any kind ("tab:orange", "#1f77b4") -> Line2D kwargs; None if `fmt` is none
    of these (then it is taken as a metric name). A marker with no line style means markers only.
    """
    style, i = {}, 0
    while i < len(fmt):
        two = fmt[i:i + 2]
        if two in ("--", "-.") and "linestyle" not in style:
            style["linestyle"], i = two, i + 2
            continue
        c = fmt[i]
        if c in "-:" and "linestyle" not in style:
            style["linestyle"] = c
        elif c == "C" and fmt[i + 1:i + 2].isdigit() and "color" not in style:
            j = i + 1
            while j < len(fmt) and fmt[j].isdigit():
                j += 1
            style["color"], i = fmt[i:j], j
            continue
        elif c in _FMT_COLORS and "color" not in style:
            style["color"] = c
        elif c in _FMT_MARKERS and "marker" not in style:
            style["marker"] = c
        else:
            from matplotlib.colors import is_color_like

            return {"color": fmt} if is_color_like(fmt) else None
        i += 1
    if "marker" in style and "linestyle" not in style:
        style["linestyle"] = "None"
    return style or None


_scratch_axes = None


def _try(method: str, *args, **kwargs):
    """
    Call `Axes.<method>(*args, **kwargs)` on a scratch matplotlib Axes and throw the result away. The
    real drawing happens later, in the render process, where a misspelt keyword or an unknown scale
    would only kill the renderer; trying it here first raises matplotlib's own error at the line that
    made the mistake.
    """
    global _scratch_axes
    if _scratch_axes is None:
        from matplotlib.figure import Figure

        _scratch_axes = Figure().add_subplot()
    try:
        getattr(_scratch_axes, method)(*args, **kwargs)
    finally:
        _scratch_axes.cla()


_DATA_KWARGS = {"data", "xdata", "ydata", "transform", "axes", "figure"}  # liveplot's to set, not the caller's


class _Line:
    """
    One metric's line, as `ax.plot` returns it (in a one-element list, like matplotlib's, so
    `(line,) = ax.plot("loss")` works). It takes Line2D's setters for restyling later:
    `line.set_color("red")`, `line.set_linestyle("--")`, `line.set(alpha=0.5, label="train")`.
    """

    def __init__(self, plot, index: int, name: str):
        self._plot, self._index, self._name = plot, index, name

    @property
    def _styles(self) -> dict:
        return self._plot._specs[self._index]["styles"]

    def set(self, **kwargs):
        """Like `Line2D.set`: any Line2D property, e.g. `line.set(color="k", linewidth=2)`."""
        style = {**self._styles.get(self._name, {}), **kwargs}
        _check_line_style(style)
        self._styles[self._name] = style
        self._plot._send_layout()

    def get_label(self) -> str:
        return self._styles.get(self._name, {}).get("label", self._name)

    def __getattr__(self, attr):
        if attr.startswith("set_"):  # set_color(c), set_linewidth(w), ...: Line2D's setters, via set()
            return lambda value: self.set(**{attr[4:]: value})
        raise AttributeError(attr)

    def __repr__(self):
        return f"<liveplot line {self._name!r}>"


def _check_line_style(style: dict):
    """Line2D kwargs as `ax.plot` would take them, checked on a scratch Axes (see `_try`)."""
    taken = sorted(_DATA_KWARGS & set(style))
    assert not taken, f"{taken}: liveplot sets these itself; a metric's data comes from log()"
    _try("plot", [], [], **style)


def _text_kw(kwargs: dict, **named) -> dict:
    """matplotlib's text setters' named arguments (fontdict, loc, pad, labelpad), where given, and the rest."""
    return {**{k: v for k, v in named.items() if v is not None}, **kwargs}


class _Axis:
    """
    One y-axis of a panel (left, or the right-hand `secondary` one), with matplotlib's `Axes` names:
    plot / set_ylabel / set_ylim / set_yscale / axhline / set(**kwargs). Panel-level setters (title,
    xlabel, xlim, xscale, axvline, legend, smooth) are available here too, as they are on a matplotlib
    Axes.
    """

    def __init__(self, plot, index: int, right: bool):
        self._plot, self._index, self._right = plot, index, right

    @property
    def panel(self):
        return Panel(self._plot, self._index)

    @property
    def metrics(self) -> list:
        return list(self._plot._specs[self._index]["secondary" if self._right else "metrics"])

    @property
    def lines(self) -> list:
        """Like `Axes.lines`: a handle on each metric's line on this axis, for restyling."""
        return [_Line(self._plot, self._index, name) for name in self.metrics]

    def _key(self, key):
        return key + ("2" if self._right else "")

    def _set(self, key, value, kw=None):
        spec = self._plot._specs[self._index]
        spec[self._key(key)] = value
        if kw is not None:
            spec["kw"][self._key(key)] = kw
        self._plot._send_layout()

    def plot(self, *args, **kwargs) -> list:
        """
        Like `Axes.plot`, with a metric's name where matplotlib takes data: `ax.plot("loss")` draws the
        metric `loss` against the step, as matplotlib's `ax.plot("loss", data=d)` draws d["loss"]
        against its index; `log()` supplies the data later. A format string and Line2D kwargs style
        it: `ax.plot("loss", "r--", label="train loss", alpha=0.8)`. Returns `[line]`, like matplotlib,
        and the line takes Line2D's setters: `(line,) = ax.plot("loss"); line.set_color("k")`.

        One metric per call. matplotlib reads `ax.plot("a", "b", data=d)` as b against a, so two
        names are an error here rather than two curves.
        """
        assert args, 'plot() needs a metric name: ax.plot("loss")'
        name, *rest = args
        assert isinstance(name, str), (
            f"plot() takes a metric's name, not data (got {type(name).__name__}): the data comes from "
            f'log(). Name the metric here, e.g. ax.plot("loss"), then plot.log(loss=...) in the loop'
        )
        style = _parse_fmt(rest[0]) if len(rest) == 1 and isinstance(rest[0], str) else None
        if rest and style is None:
            names = [a for a in args if isinstance(a, str)]
            raise TypeError(
                f"ax.plot({', '.join(map(repr, args))}): one metric per call. matplotlib reads two names as x "
                f"and y ({args[1]!r} against {args[0]!r}); liveplot plots every metric against the step, so "
                f"call plot once for each: " + "; ".join(f"ax.plot({n!r})" for n in names)
            )
        style = {**(style or {}), **kwargs}
        _check_line_style(style)
        spec = self._plot._specs[self._index]
        assert spec["kind"] == "curve", "this panel is showing an image; use a different panel for curves"
        where = self._plot._axis_of(name)
        if where is not None and where != (self._index, self._right):
            raise ValueError(
                f"{name!r} is already drawn on {self._plot[name]!r}; a metric lives on one axis. If log() "
                f"placed it before this call, call ax.plot({name!r}) before the first log()"
            )
        key = "secondary" if self._right else "metrics"
        if name not in spec[key]:
            auto = " / ".join(spec["metrics"] + spec["secondary"])
            was_auto = spec["title"] in (auto, "")  # unnamed, or named after the metrics it had
            spec[key].append(name)
            if was_auto:  # a title set with set_title() survives a later plot() call
                spec["title"] = " / ".join(spec["metrics"] + spec["secondary"])
        if style:
            spec["styles"][name] = {**spec["styles"].get(name, {}), **style}
        self._plot._place([name])
        return [_Line(self._plot, self._index, name)]

    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        kw = _text_kw(kwargs, fontdict=fontdict, labelpad=labelpad, loc=loc)
        _try("set_ylabel", str(ylabel), **kw)
        self._set("ylabel", str(ylabel), kw)

    def set_ylim(self, bottom=None, top=None, *, ymin=None, ymax=None):
        """Like `Axes.set_ylim`: `set_ylim(0, 1)`, `set_ylim((0, 1))`, or one end, `set_ylim(bottom=0)`."""
        self._set("ylim", _lim(bottom, top, ymin, ymax))

    def set_yscale(self, value, **kwargs):
        """Like `Axes.set_yscale`: "linear", "log", "symlog", "logit", ..., with the scale's kwargs."""
        _try("set_yscale", value, **kwargs)
        self._set("yscale", value, kwargs)

    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        """
        Like `Axes.axhline`: a horizontal line at `y`, dashed grey unless kwargs say otherwise. Give it
        a `label` to put it in the legend, e.g. `ax.axhline(math.log(10), label="chance", color="r")`.
        """
        line = {**kwargs, "y": float(y), **({"xmin": xmin, "xmax": xmax} if (xmin, xmax) != (0, 1) else {})}
        _try("axhline", **line)
        self._plot._specs[self._index][self._key("axhlines")].append(line)
        self._plot._send_layout()

    def set(self, **kwargs):
        """Like `Axes.set`: `ax.set(ylabel="loss", ylim=(0, 1), yscale="log", title=...)`."""
        for key, value in kwargs.items():
            getattr(self, f"set_{key}")(value)

    # panel-level setters, so plot["acc"].set_title(...) works as it would on a matplotlib Axes
    def set_title(self, label, fontdict=None, loc=None, pad=None, **kwargs):
        self.panel.set_title(label, fontdict, loc, pad, **kwargs)

    def legend(self, **kwargs):
        self.panel.legend(**kwargs)

    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        self.panel.set_xlabel(xlabel, fontdict, labelpad, loc=loc, **kwargs)

    def set_xlim(self, left=None, right=None, *, xmin=None, xmax=None):
        self.panel.set_xlim(left, right, xmin=xmin, xmax=xmax)

    def set_xscale(self, value, **kwargs):
        self.panel.set_xscale(value, **kwargs)

    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        self.panel.axvline(x, ymin, ymax, **kwargs)

    def set_smooth(self, weight):
        self.panel.set_smooth(weight)

    def twinx(self):
        return self.panel.twinx()

    def __repr__(self):
        side = "right" if self._right else "left"
        return f"<liveplot axis: panel {self._index}, {side}: {' '.join(self.metrics) or '(empty)'}>"


class Panel:
    """
    One panel of the grid, the Axes of a matplotlib figure: `plot.axes[i]`, what `LivePlot.subplots`
    hands out, or `plot[metric].panel`. `.left` / `.right` are its y-axes (`.right` exists only if the
    panel has secondary metrics; `twinx()` makes it). y-setters here act on the left axis.
    """

    def __init__(self, plot, index: int):
        self._plot, self._index = plot, index

    @property
    def spec(self) -> dict:
        return self._plot._specs[self._index]

    @property
    def metrics(self) -> list:
        return list(self.spec["metrics"]) + list(self.spec["secondary"])

    @property
    def lines(self) -> list:
        """Like `Axes.lines`: a handle on each metric's line on this panel (both y-axes), for restyling."""
        return [_Line(self._plot, self._index, name) for name in self.metrics]

    @property
    def left(self) -> _Axis:
        return _Axis(self._plot, self._index, right=False)

    @property
    def right(self) -> _Axis:
        assert self.spec["secondary"], "this panel has no right-hand axis (no metrics after '|'; twinx() makes one)"
        return _Axis(self._plot, self._index, right=True)

    def twinx(self) -> _Axis:
        """Like `Axes.twinx`: this panel's right-hand y-axis, whether or not it holds anything yet."""
        return _Axis(self._plot, self._index, right=True)

    def plot(self, *args, **kwargs) -> list:
        """Like `Axes.plot`, on the left axis: `ax.plot("loss", "r--", label="train")`. See `_Axis.plot`."""
        return self.left.plot(*args, **kwargs)

    def imshow(self, x, *, rows=None, cols=None, griddim=None, vmin=None, vmax=None, scale_each=False,
               channels=None, cmap="gray", padding=0, pad_value=0, max_images=64, max_cols=8, **kwargs):
        """
        Like `Axes.imshow`, but for a whole batch and repeatable: show `x` on this panel, replacing
        whatever was there. A batch is tiled into a grid -- `rows` or `cols` alone infers the other,
        `griddim=(rows, cols)` fixes both (padding with blanks, or dropping the tail). With neither,
        the grid is near-square (3 -> 2x2, 5 -> 3x2, 8 -> 3x3, 64 -> 8x8), at most `max_cols` wide.
        Only the first `max_images` are shown, with a warning (`max_images=None` for all of them).
        Images sit edge to edge; `padding` puts that many pixels of `pad_value` between them. The
        title shows the step the image was drawn at. Other kwargs (`interpolation`, `aspect`, ...) go
        to matplotlib's `imshow`; the default interpolation is "nearest", so pixels stay crisp.

        Accepts (H, W), (C, H, W), (H, W, C), (B, H, W), (B, C, H, W) and (B, H, W, C); (3, H, W)
        and (4, H, W) are ambiguous and raise, telling you which `channels=` to pass.

        Values are scaled to the full range by default, over the whole batch; `vmin` / `vmax` fix
        the range instead (worth doing for a live view -- otherwise the black point moves every
        frame), and `scale_each=True` scales each image on its own, as make_grid does. Unlike
        matplotlib, `vmin` / `vmax` are honoured for colour images too.
        """
        from ._images import to_grid

        spec = self.spec
        assert not (spec["metrics"] or spec["secondary"]), \
            f"panel {self._index} is showing curves ({' '.join(spec['metrics'] + spec['secondary'])}); " \
            f"use a different panel for the image"
        arr = to_grid(x, rows=rows, cols=cols, griddim=griddim, vmin=vmin, vmax=vmax, scale_each=scale_each,
                      channels=channels, padding=padding, pad_value=pad_value, max_images=max_images,
                      max_cols=max_cols)
        if kwargs != spec["kw"].get("imshow", {}):
            taken = sorted({"X", "vmin", "vmax", "norm"} & set(kwargs))
            assert not taken, f"{taken}: liveplot scales the pixels itself (see vmin / vmax above)"
            _try("imshow", [[0, 1], [1, 0]], **kwargs)
        if spec["kind"] != "image" or spec["cmap"] != cmap or spec["kw"].get("imshow", {}) != kwargs:
            spec["kind"], spec["cmap"], spec["kw"]["imshow"] = "image", cmap, kwargs
            self._plot._send_layout()
        self._plot._send_image(self._index, arr)
        return self

    def _set(self, key, value, kw=None):
        self.spec[key] = value
        if kw is not None:
            self.spec["kw"][key] = kw
        self._plot._send_layout()

    def set_title(self, label, fontdict=None, loc=None, pad=None, **kwargs):
        """Like `Axes.set_title`, text kwargs included: `set_title("training", fontsize=10, loc="left")`."""
        kw = _text_kw(kwargs, fontdict=fontdict, loc=loc, pad=pad)
        _try("set_title", str(label), **kw)
        self._set("title", str(label), kw)

    def legend(self, *args, **kwargs):
        """
        Like `Axes.legend`, for the panel's one legend (both y-axes' curves and its labelled reference
        lines): kwargs go to matplotlib, e.g. `legend(loc="upper left", ncols=3)`, or
        `legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=3)` to put it under the panel.
        Every panel has a legend already; this only moves or styles it. What it lists comes from the
        labels: `ax.plot("loss", label="train")`, `axhline(..., label=...)`, and a label starting with
        "_" leaves an artist out, as in matplotlib.
        """
        assert not args, (
            "legend() takes keyword arguments only: the entries come from the labels given to plot(), "
            "axhline() and axvline() (a label starting with '_' leaves one out)"
        )
        _try("legend", [], [], **kwargs)
        self._set("legend", kwargs)

    def set_xlabel(self, xlabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        kw = _text_kw(kwargs, fontdict=fontdict, labelpad=labelpad, loc=loc)
        _try("set_xlabel", str(xlabel), **kw)
        self._set("xlabel", str(xlabel), kw)

    def set_xlim(self, left=None, right=None, *, xmin=None, xmax=None):
        """Like `Axes.set_xlim`: `set_xlim(0, 1000)`, `set_xlim((0, 1000))`, or one end, `set_xlim(left=100)`."""
        self._set("xlim", _lim(left, right, xmin, xmax))

    def set_xscale(self, value, **kwargs):
        """Like `Axes.set_xscale`: "linear", "log", "symlog", ..., with the scale's kwargs."""
        _try("set_xscale", value, **kwargs)
        self._set("xscale", value, kwargs)

    def set_smooth(self, weight):
        """wandb-style smoothing weight in [0, 1) for every curve on this panel; 0 turns it off."""
        self._set("smooth", _check_smooth(weight))

    def axvline(self, x=0, ymin=0, ymax=1, **kwargs):
        """
        Like `Axes.axvline`: a vertical line at `x`, dotted grey unless kwargs say otherwise, with a
        `label` for the legend. To mark the current point of a run: `ax.axvline(plot.step, label="lr drop")`.
        """
        line = {**kwargs, "x": float(x), **({"ymin": ymin, "ymax": ymax} if (ymin, ymax) != (0, 1) else {})}
        _try("axvline", **line)
        self.spec["axvlines"].append(line)
        self._plot._send_layout()

    def set(self, **kwargs):
        """Like `Axes.set`: `ax.set(title="training", xlabel="examples", ylim=(0, 1), yscale="log")`."""
        for key, value in kwargs.items():
            getattr(self, f"set_{key}")(value)

    # y-setters act on the left axis, as on a matplotlib Axes
    def set_ylabel(self, ylabel, fontdict=None, labelpad=None, *, loc=None, **kwargs):
        self.left.set_ylabel(ylabel, fontdict, labelpad, loc=loc, **kwargs)

    def set_ylim(self, bottom=None, top=None, *, ymin=None, ymax=None):
        self.left.set_ylim(bottom, top, ymin=ymin, ymax=ymax)

    def set_yscale(self, value, **kwargs):
        self.left.set_yscale(value, **kwargs)

    def axhline(self, y=0, xmin=0, xmax=1, **kwargs):
        self.left.axhline(y, xmin, xmax, **kwargs)

    def __repr__(self):
        return f"Panel({self._index}: {' '.join(self.spec['metrics'])}{' | ' + ' '.join(self.spec['secondary']) if self.spec['secondary'] else ''})"


# --------------------------------------------------------------------------- the handle


class LivePlot:
    def __init__(
        self,
        *args,
        total: int | None = None,
        initial: int | float = 0,
        unit: str = "step",
        unit_scale: int | float = 1,
        refresh_seconds: float = 1.0,
        max_cols: int | None = 3,
        rows: int | None = None,
        cols: int | None = None,
        cell_size: tuple = (5, 3.5),
        dpi: int = 100,
        progress: bool = True,
        desc: str | None = None,
        record: bool | str = False,
        smooth: float | None = None,
    ):
        """
        LivePlot([iterable,] *panels, total=None, initial=0, unit="step", unit_scale=1, ...)

        iterable: anything to loop over (a range, a DataLoader, an existing tqdm bar). Iterating
            the LivePlot yields its items, shows a tqdm bar (unless progress=False or it already is
            one) and finishes the plot when the loop ends, however it ends. For nested loops, leave
            it out and wrap the inner loop with `plot(iterable, **tqdm_kwargs)` instead.
        panels: layout strings like "loss", "return | entropy", "lossD lossG | acc", or dicts (see
            module docstring). With none, every logged metric shares one panel.
        total, initial, unit, unit_scale: tqdm's arguments, with tqdm's meaning; they define the
            x-axis (see module docstring).
        """
        iterable, specs = (args[0], args[1:]) if args and not isinstance(args[0], (str, dict)) else (None, args)
        self.smooth = _check_smooth(smooth)  # default TWEMA weight for panels that don't set their own (wandb's smoothing slider)
        self._panel_defaults: dict = {}  # set_all() kwargs, applied to panels created later
        self._specs = [self._with_defaults(_normalise_panel(p)) for p in specs]
        self._explicit_layout = bool(self._specs)
        self._placed = {n for p in self._specs for n in p["metrics"] + p["secondary"]}
        # the grid and figure options the renderer needs; the dict holds subplots()' grid options and
        # the figure-level text (suptitle / supxlabel / supylabel)
        self._layout = (max_cols, rows, cols, cell_size, dpi, unit, {})
        self._iterable, self._bar, self._desc, self._progress = iterable, None, desc, progress
        self._own_bar = None  # the bar update() opened, which finish() closes
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                pass
        self.total, self.initial, self.unit, self.unit_scale = total, initial, unit, unit_scale
        self.refresh_seconds = refresh_seconds
        self.data: dict[str, tuple[list, list]] = {}  # metric -> (x values, values); full history, this side
        self.latest: dict[str, float] = {}  # most recent value of each metric (what the tqdm postfix shows)
        self.last_png: bytes | None = None
        self.n = 0  # items consumed so far across every wrapped loop (tqdm's `n`)
        self._wrapping = False  # has any loop been wrapped? (then x comes from the counter)
        self._last_x = initial  # x of the last explicit `log(step, ...)`; used when nothing is wrapped
        self._record = bool(record)
        self._record_path = record if isinstance(record, str) else None
        self.frames: list[tuple[float, bytes]] = []  # (time, png) of every frame shown, if record=True
        self._images: dict[int, object] = {}  # panel index -> the uint8 array it is showing
        self._image_steps: dict[int, object] = {}  # panel index -> the x it was shown at
        self._fixed_grid = False  # set by subplots(): grow a panel's metrics, never the grid
        self._t0 = time.monotonic()
        self._done = False
        self._proc = self._renderer = self._inbox = self._outbox = None
        self._last_draw = self._last_postfix = 0.0
        self._n_logs = 0
        self._handle = self._make_display_handle()
        if self._handle is None and not self._record:
            self.mode = "off"  # not in a notebook: nothing to draw on, just collect metrics
            return
        try:
            self._start_process()
            self.mode = "process"
        except Exception as e:  # noqa: BLE001 - whatever stops the process, keep the plot working
            warnings.warn(
                f"LivePlot: could not start the render process ({type(e).__name__}: {e}); "
                f"rendering on the training thread instead (~0.15 s per redraw).",
                stacklevel=2,
            )
            self._renderer = _FigureRenderer(self._panels_or_placeholder(), self.x_range, self._layout, images=self._image_state())
            self.mode = "thread"

    # -- setup -------------------------------------------------------------------

    @staticmethod
    def _make_display_handle():
        try:
            from IPython import get_ipython
            from IPython.display import HTML, display
        except ImportError:
            return None
        if get_ipython() is None:
            return None
        return display(HTML("<i>live plot: waiting for the first frame…</i>"), display_id=True)

    def _with_defaults(self, panel):
        if panel["smooth"] is None:
            panel["smooth"] = self.smooth
        return panel

    def _add_panel(self, spec) -> int:
        """Append a panel, with the plot's smoothing default and whatever `set_all` has asked of every panel."""
        self._specs.append(self._with_defaults(spec))
        index = len(self._specs) - 1
        if self._panel_defaults:
            Panel(self, index).set(**self._panel_defaults)
        return index

    @classmethod
    def subplots(cls, nrows: int = 1, ncols: int = 1, *, sharex=False, sharey=False, squeeze: bool = True,
                 width_ratios=None, height_ratios=None, subplot_kw=None, gridspec_kw=None, figsize=None,
                 iterable=None, **kwargs):
        """
        Like `plt.subplots`, for a live plot: returns `(plot, axes)` with a fixed nrows x ncols grid
        of empty panels, which you then fill with `ax.plot("loss")` or `ax.imshow(tensor)`.

            plot, (ax_loss, ax_samples) = LivePlot.subplots(1, 2, total=n_steps, figsize=(11, 4))
            ax_loss.plot("lossD")
            ax_loss.plot("lossG")
            ax_samples.imshow(netG(fixed_noise), rows=2, vmin=-1, vmax=1)

        The arguments are matplotlib's. `axes` is a numpy array of panels squeezed as matplotlib does
        (one panel for 1x1, 1-d for a single row or column, 2-d otherwise; `squeeze=False` always
        2-d), so `axes[0, 1]`, `axes.flat` and `axes.flatten()` all work. `figsize` is the whole
        figure in inches (liveplot's own `cell_size` is per panel); `sharex` / `sharey`,
        `width_ratios` / `height_ratios` and `gridspec_kw` go to matplotlib's grid; `subplot_kw` is
        applied to every panel with `Panel.set`, e.g. `subplot_kw=dict(xlabel="examples")`. Every
        other keyword goes to `LivePlot` (`total=`, `unit=`, ...).
        """
        import numpy as np

        assert nrows >= 1 and ncols >= 1, f"need at least a 1x1 grid, got {nrows}x{ncols}"
        for name, share in (("sharex", sharex), ("sharey", sharey)):
            assert share in (True, False, "all", "row", "col", "none"), \
                f"{name} must be a bool or one of 'all', 'row', 'col', 'none', got {share!r}"
        kwargs.setdefault("rows", nrows)
        kwargs.setdefault("cols", ncols)
        if figsize is not None:
            kwargs["cell_size"] = (figsize[0] / ncols, figsize[1] / nrows)
        gridspec_kw = dict(gridspec_kw or {})
        for key, value, count in (("width_ratios", width_ratios, ncols), ("height_ratios", height_ratios, nrows)):
            if value is not None:
                assert key not in gridspec_kw, f"'{key}' must not be defined both as parameter and as key in 'gridspec_kw'"
                gridspec_kw[key] = list(value)
            if key in gridspec_kw:
                assert len(gridspec_kw[key]) == count, f"{key} needs one entry per {'column' if count == ncols else 'row'}"
        plot = cls(*([] if iterable is None else [iterable]), **kwargs)
        plot._layout[6].update(gridspec_kw=gridspec_kw, sharex=sharex, sharey=sharey)
        for _ in range(nrows * ncols):
            plot._add_panel(_normalise_panel({"metrics": []}, allow_empty=True))
        plot._explicit_layout = True
        plot._fixed_grid = True
        axes = np.empty((nrows, ncols), dtype=object)
        for r in range(nrows):
            for c in range(ncols):
                axes[r, c] = Panel(plot, r * ncols + c)
                if subplot_kw:
                    axes[r, c].set(**subplot_kw)
        plot._send_layout()
        if squeeze:
            axes = axes.item() if axes.size == 1 else axes.squeeze()
        return plot, axes

    # -- the figure: its panels, and figure-level text ---------------------------------

    @property
    def axes(self) -> list:
        """Like `Figure.axes`: the panels, in order. `plot.axes[0].set_title("training")`."""
        return [Panel(self, i) for i in range(len(self._specs))]

    def _axis_of(self, metric: str):
        """(panel index, right-hand axis?) of the axis `metric` is drawn on, or None if it isn't placed yet."""
        for i, spec in enumerate(self._specs):
            if metric in spec["metrics"]:
                return i, False
            if metric in spec["secondary"]:
                return i, True
        return None

    def __getitem__(self, metric: str) -> _Axis:
        """
        The y-axis that holds `metric`: `plot["acc"].set_ylim(0, 1)`. Right-hand axes are found too,
        which is what makes the left/right distinction disappear from the API. A metric that hasn't
        been logged yet gets its panel created now. (matplotlib has no lookup by series name; this is
        liveplot's, for a plot whose panels came from logging rather than from `subplots`.)
        """
        if metric not in self._placed:
            self.data.setdefault(metric, ([], []))
            self._extend_layout([metric])
        where = self._axis_of(metric)
        if where is None:
            raise KeyError(metric)
        return _Axis(self, where[0], right=where[1])

    def _figure_text(self, method: str, text, kwargs):
        from matplotlib.figure import Figure

        getattr(Figure(), method)(str(text), **kwargs)  # matplotlib's own check of the kwargs (see `_try`)
        self._layout[6][method] = (str(text), kwargs)
        self._send_layout()

    def suptitle(self, t, **kwargs):
        """Like `Figure.suptitle`: a title over the whole figure, e.g. `plot.suptitle("run 3", fontsize=12)`."""
        self._figure_text("suptitle", t, kwargs)

    def supxlabel(self, t, **kwargs):
        """Like `Figure.supxlabel`: one x-label under the whole grid."""
        self._figure_text("supxlabel", t, kwargs)

    def supylabel(self, t, **kwargs):
        """Like `Figure.supylabel`: one y-label beside the whole grid."""
        self._figure_text("supylabel", t, kwargs)

    def set_all(self, **kwargs):
        """
        `Panel.set(**kwargs)` on every panel, now and for any panel created later (a metric logged for
        the first time gets its panel with these settings too): `plot.set_all(xlabel="examples",
        smooth=0.9, yscale="log")`. matplotlib's way is a loop over `fig.axes`, which also works here,
        `for ax in plot.axes: ax.set(...)`, but a loop cannot reach panels that don't exist yet.
        """
        for key in kwargs:
            assert hasattr(Panel, f"set_{key}"), f"set_all({key}=...): a panel has no set_{key}()"
        self._panel_defaults.update(kwargs)
        for panel in self.axes:
            panel.set(**kwargs)

    def _panels_or_placeholder(self):
        return self._specs or [_normalise_panel({"title": "waiting for data…", "metrics": ["_"]})]

    def _start_process(self):
        # "spawn", never "fork": the notebook process has usually initialised CUDA,
        # and a forked child inherits a CUDA context it must not touch.
        ctx = mp.get_context("spawn")
        self._inbox, self._outbox = ctx.Queue(), ctx.Queue()
        self._proc = ctx.Process(
            target=_render_worker,
            args=(self._panels_or_placeholder(), self._inbox, self._outbox, self.refresh_seconds, self._layout,
                  self.x_range, os.getpid()),
            daemon=True,
        )
        # A spawned child re-runs the parent's __main__ *file* if there is one. In a
        # notebook there isn't; in an interactive window / `python solutions.py`-style
        # session `__main__.__file__` points at the whole notebook script, which the
        # renderer has no use for (it would import torch and build environments just to
        # draw a plot). Hide it for the duration of start() so the child stays light.
        main = sys.modules.get("__main__")
        hidden = {k: main.__dict__.pop(k) for k in ("__file__", "__cached__") if main is not None and k in main.__dict__}
        try:
            self._proc.start()
        finally:
            if main is not None:
                main.__dict__.update(hidden)
        weakref.finalize(self, _terminate, self._proc, self._inbox)  # dropped without finish() -> no leak
        # Frames are displayed by this thread as soon as the renderer produces them, so a loop that
        # logs rarely (or is busy in a long step) still sees each frame when it is ready rather than
        # on its next log() call. Everything it touches is a plain attribute write or a display update.
        self._collector_done = threading.Event()
        # The thread gets only a weak reference to the plot: a strong one (e.g. a bound method as the
        # target) would keep a dropped LivePlot alive forever and its render process with it.
        self._collector = threading.Thread(
            target=_collect_loop, args=(weakref.ref(self), self._outbox, self._proc, self._collector_done),
            name="liveplot-frames", daemon=True,
        )
        self._collector.start()

    def _make_bar(self, it, tqdm_kwargs):
        if not self._progress:
            return None
        try:
            from tqdm import tqdm as tqdm_base  # every tqdm flavour (notebook, asyncio, rich) subclasses this
            from tqdm.auto import tqdm
        except ImportError:
            return None
        if isinstance(it, tqdm_base):
            return it  # the caller's own bar: reuse it (and never close it)
        kwargs = {"unit": self.unit}
        if self.unit_scale != 1:
            kwargs["unit_scale"] = self.unit_scale  # the bar shows the same numbers as the x-axis
        try:
            kwargs["total"] = len(it)
        except TypeError:
            if self.total is not None and it is self._iterable:
                # the single-loop form over an iterable with no len(): the plot knows the total. A wrapped
                # inner loop must not borrow it -- the plot-wide total spans every epoch, not this one.
                kwargs["total"] = self.total
        if self.initial:
            # ... and it starts where the x-axis does. tqdm counts raw items and scales them for
            # display, so the offset goes in as items and `total` has to grow to match; `self.n`
            # keeps a wrapped inner loop's bar lined up with the x-axis across epochs.
            offset = self.initial / self.unit_scale + self.n
            offset = int(offset) if offset == int(offset) else offset  # keep the bar's counts whole when they are
            kwargs["initial"] = offset
            if kwargs.get("total") is not None:
                kwargs["total"] += offset
        kwargs.update(tqdm_kwargs)  # the caller's kwargs win over the ones we derived
        # No iterable is given to tqdm: we drive it with update() ourselves, so that the closing
        # line of the bar still shows the final postfix (tqdm closes a wrapped iterable before we
        # could write it).
        return tqdm(**kwargs)

    # -- use ----------------------------------------------------------------------

    @property
    def x_range(self):
        """Default x-axis range (initial, initial + total * unit_scale), or None to autoscale."""
        return (self.initial, self.initial + self.total * self.unit_scale) if self.total else None

    @property
    def step(self):
        """The x value a `log()` call gets right now."""
        return self.initial + self.n * self.unit_scale if self._wrapping else self._last_x

    def __call__(self, iterable, **tqdm_kwargs):
        """
        Wrap a loop: `for batch in plot(loader, desc="epoch 3"):`. Shows a tqdm bar (kwargs go to
        tqdm) and advances the item count that drives the x-axis. Wrap as many loops as you like
        on one plot; the count continues across them. Does not finish the plot when the loop ends.
        """
        self._wrapping = True
        bar = self._make_bar(iterable, tqdm_kwargs)
        self._bar = bar
        ours = bar is not None and bar is not iterable
        try:
            for item in (iterable if ours or bar is None else bar):
                yield item
                self.n += 1
                if ours:
                    bar.update(1)
        finally:
            if bar is not None:
                bar.set_postfix(self.latest, refresh=False)
                if ours:
                    bar.close()

    def update(self, n=1):
        """
        Like tqdm's `update`, for a loop you drive yourself: advance the count, and so the x-axis, by
        `n` items. The first call opens a bar of `total` items below the plot, as `tqdm(total=...)`
        would, and `finish()` closes it. One bar for a whole multi-epoch run:

            plot = LivePlot(total=epochs * len(loader))
            for epoch in range(epochs):
                for batch in loader:
                    plot.log(loss=train_step(batch))
                    plot.update()
            plot.finish()
        """
        self._wrapping = True
        if self._bar is None and self._own_bar is None:
            self._own_bar = self._bar = self._make_bar(None, {"desc": self._desc} if self._desc else {})
        self.n += n
        if self._own_bar is not None:
            self._own_bar.update(n)

    def set_description(self, desc=None, refresh=True):
        """Like tqdm's `set_description`: the text before the bar (e.g. f"epoch {epoch}")."""
        self._desc = desc
        if self._bar is not None:
            self._bar.set_description(desc, refresh=refresh)

    def __iter__(self):
        if self._iterable is None:
            raise TypeError("nothing to iterate: use LivePlot(iterable, ...), or plot(iterable) inside your loop")
        try:
            kwargs = {"desc": self._desc} if self._desc else {}
            yield from self(self._iterable, **kwargs)
        finally:
            self.finish()

    def log(self, *args, **metrics):
        """
        log(loss=0.3, acc=0.9)            step = the current loop step
        log(step, loss=0.3)               explicit step
        log(step, {"loss": 0.3})          dict form
        Any value convertible to float works (tensors with one element included).
        """
        step = self.step
        if args:
            if isinstance(args[0], dict):
                metrics = {**args[0], **metrics}
            else:
                step = self._last_x = args[0]  # explicit x for this call (and the default until iteration)
                if len(args) > 1:
                    metrics = {**args[1], **metrics}
        picked = {k: float(v) for k, v in metrics.items()}
        new = [k for k in picked if k not in self.data]
        for name, value in picked.items():
            self.data.setdefault(name, ([], []))
            self.data[name][0].append(step)
            self.data[name][1].append(value)
        self.latest.update(picked)
        if new:
            self._extend_layout(new)
        if self.mode == "process":
            self._inbox.put(("data", step, picked))
            self._n_logs += 1
            if self._n_logs % 100 == 0 and not self._proc.is_alive():
                self._fall_back_to_thread("the render process died")
        if self.mode == "thread":
            self._renderer.add(step, picked)
            if time.monotonic() - self._last_draw >= self.refresh_seconds:
                self._show(self._renderer.render())
        if self._bar is not None and time.monotonic() - self._last_postfix > 0.1:
            self._bar.set_postfix(self.latest, refresh=False)
            self._last_postfix = time.monotonic()

    def _extend_layout(self, new_metrics):
        """Metrics no panel mentions: all on one shared panel if no layout was given, else one panel each."""
        unplaced = [m for m in new_metrics if m not in self._placed]
        if not unplaced:
            return
        curves = [i for i, spec in enumerate(self._specs) if spec["kind"] == "curve"]
        if self._fixed_grid:
            # subplots() promised a grid of this shape, so grow a panel rather than the grid: the
            # first one still empty, else the first curve panel there is.
            assert curves, "subplots() gave this plot no curve panel to hold " + ", ".join(unplaced)
            empty = [i for i in curves if not self._specs[i]["metrics"] and not self._specs[i]["secondary"]]
            self._join_panel(empty[0] if empty else curves[0], unplaced)
        elif self._explicit_layout:
            for m in unplaced:
                self._add_panel(_normalise_panel({"metrics": [m]}))
        elif not curves:  # nothing but image panels so far (plot.imshow() before the first log)
            self._add_panel(_normalise_panel({"metrics": unplaced}))
        else:
            self._join_panel(curves[0], unplaced)
        self._placed.update(unplaced)
        self._send_layout()

    def _join_panel(self, index: int, unplaced: list):
        """Add metrics to an existing panel, keeping a title the user chose."""
        spec = self._specs[index]
        auto = " / ".join(spec["metrics"] + spec["secondary"])
        was_auto = spec["title"] in (auto, "")  # a title nobody chose, or a fresh subplots() panel
        spec["metrics"].extend(unplaced)
        if was_auto:  # a title set with set_title() survives a newly discovered metric
            spec["title"] = " / ".join(spec["metrics"] + spec["secondary"])

    def imshow(self, x, **kwargs):
        """
        Like pyplot's `plt.imshow`, which draws on the current Axes: show an image on the plot's image
        panel, creating it on first use, replacing whatever was there. Keywords go to `Panel.imshow`.
        """
        for i, spec in enumerate(self._specs):
            if spec["kind"] == "image":
                return Panel(self, i).imshow(x, **kwargs)
        assert not self._fixed_grid, \
            "this plot's grid comes from subplots(); call imshow() on one of its panels instead"
        index = self._add_panel(_normalise_panel({"kind": "image"}))
        self._send_layout()
        return Panel(self, index).imshow(x, **kwargs)

    def _place(self, metrics):
        """Record metrics as already belonging to a panel, and redraw."""
        self._placed.update(metrics)
        for name in metrics:
            self.data.setdefault(name, ([], []))
        self._send_layout()

    def _send_image(self, index: int, arr):
        x = self.step if self._wrapping or self.latest else None  # a step in the title only once there are steps
        self._images[index], self._image_steps[index] = arr, x
        if self.mode == "process":
            self._inbox.put(("image", index, arr, x))
        elif self.mode == "thread":
            self._renderer.add_image(index, arr, x)

    def _image_state(self) -> dict:
        """What a fresh renderer needs to redraw the images: panel index -> (array, x)."""
        return {i: (arr, self._image_steps.get(i)) for i, arr in self._images.items()}

    def _send_layout(self):
        if self.mode == "process":
            self._inbox.put(("layout", self._specs, self._layout))
        elif self.mode == "thread":
            self._renderer.layout = self._layout
            self._renderer.set_layout(self._specs)

    def figure(self):
        """
        A matplotlib Figure of the plot as it stands (same panels, data, reference lines and marks),
        built on the calling thread and independent of the render process: title it, tweak it,
        `fig.savefig("run.png")`, or show it in a report. Safe to call during or after training.
        """
        renderer = _FigureRenderer(self._panels_or_placeholder(), self.x_range, self._layout, images=self._image_state())
        for name, (xs, ys) in self.data.items():
            renderer.hist[name] = (list(xs), list(ys))
        renderer.render()  # sets the line data and autoscales
        return renderer.fig

    def refresh(self):
        """Force a redraw now (thread mode only; the render process paces itself)."""
        if self.mode == "thread":
            self._show(self._renderer.render())

    def finish(self):
        """Draw the final frame and shut the renderer down. Safe to call twice, and to interrupt."""
        if self._done:
            return
        self._done = True
        try:
            if self.mode == "process":
                if self._proc.is_alive():
                    self._inbox.put(None)  # renderer draws the final frame, sends it, then None
                self._collector_done.wait(timeout=5.0)  # the collector shows that frame and exits
            elif self.mode == "thread":
                self._show(self._renderer.render())
        finally:
            if self.mode == "process":
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():  # e.g. still importing matplotlib on a very busy machine
                    self._proc.terminate()
                    self._proc.join(timeout=2.0)
            if self._own_bar is not None:
                self._own_bar.set_postfix(self.latest, refresh=False)
                self._own_bar.close()
            if self._record_path and self.frames:
                self.save_gif(self._record_path)

    def save_gif(self, path, speedup: float = 1.0, max_frame_ms: int = 1000, hold_last_ms: int = 1500, colors: int = 256):
        """
        Write the recorded frames (needs `record=True`) as an animated GIF that replays at the real
        pace of the run divided by `speedup`, with no single frame shown longer than `max_frame_ms`.
        Returns the path. Uses Pillow (installed with matplotlib).
        """
        import io

        from PIL import Image

        if not self.frames:
            raise ValueError("nothing recorded: create the plot with record=True")
        images = [Image.open(io.BytesIO(png)).convert("RGB") for _, png in self.frames]
        # A GIF has ONE palette for the whole animation. Quantising each frame to its own adaptive
        # palette (the obvious thing) corrupts the colours of every frame whose palette differs from
        # the first one's. So build a single palette from a sample of frames and map every frame to it.
        # 256 colours (GIF's maximum) costs a few percent in file size over 64 for a line chart and
        # keeps anti-aliased edges and legend swatches true.
        w, h = images[0].size
        sample = sorted({0, len(images) - 1, *range(0, len(images), max(1, len(images) // 6))})[:8]
        sheet = Image.new("RGB", (w, h * len(sample)))
        for k, i in enumerate(sample):
            sheet.paste(images[i], (0, k * h))
        palette = sheet.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
        images = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in images]
        times = [t for t, _ in self.frames]
        durations = [min(int(1000 * (b - a) / speedup), max_frame_ms) for a, b in zip(times, times[1:])] + [hold_last_ms]
        durations = [max(d, 20) for d in durations]  # GIF viewers ignore very short delays
        images[0].save(path, save_all=True, append_images=images[1:], duration=durations, loop=0, optimize=False)
        return path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()

    # -- plumbing -----------------------------------------------------------------

    def _fall_back_to_thread(self, why):
        # Drawing is never worth an exception in someone's training loop: whatever killed the render
        # process will usually kill a renderer built here too (a bad panel spec, a missing backend),
        # so if this fails, say so once and carry on collecting into plot.data.
        try:
            renderer = _FigureRenderer(self._panels_or_placeholder(), self.x_range, self._layout, images=self._image_state())
            for name, (xs, ys) in self.data.items():
                renderer.hist[name] = (list(xs), list(ys))
        except Exception as e:  # noqa: BLE001
            self.mode = "off"
            warnings.warn(
                f"LivePlot: {why}, and drawing on the training thread failed too "
                f"({type(e).__name__}: {e}); collecting into plot.data only.",
                stacklevel=3,
            )
            return
        warnings.warn(f"LivePlot: {why}; rendering on the training thread from now on.", stacklevel=3)
        self._renderer = renderer
        self.mode = "thread"

    def _show(self, png: bytes):
        self.last_png = png
        self._last_draw = time.monotonic()
        if self._record:
            self.frames.append((time.monotonic() - self._t0, png))
        if self._handle is not None:
            from IPython.display import Image

            self._handle.update(Image(data=png, format="png"))
