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
numbers). `total` (in items, like tqdm's) sets the bar's length, and with
fixed_x=True the x range; the single-loop form takes it from len(iterable). An explicit `plot.log(step, ...)` uses that x for that
call only, like wandb's `step=`.

Metrics are discovered from what you log: with no layout given, every metric goes
on ONE panel with a legend. To split them up, give panel strings:

    LivePlot(range(N), "loss", "return | entropy", "lossD lossG | acc")

Each string is one panel. Names separated by spaces share the left y-axis; a `|`
puts the names after it on a right-hand y-axis. Metrics you log that no string
mentions get a panel of their own. Every panel has a legend. For labels or fixed
ranges, use a dict instead of a string:

    {"metrics": ["acc"], "ylim": (0, 1), "ylabel": "test accuracy", "xlabel": "epoch"}

(the dict form; the setters below are the nicer way). Panels and axes are addressed
and configured with matplotlib's own names, before or during the loop:

    plot["acc"].set_ylim(0, 1)                 # the y-axis holding a metric (left or right)
    plot["acc"].set_ylabel("test accuracy")
    plot["loss"].axhline(0.1, label="target", color="red")
    plot.panels[0].set_title("training")       # a panel: title, xlabel, xlim, axvline, smooth
    plot.panels[0].right.set_yscale("log")     # .left / .right are the panel's y-axes
    plot.set(xlabel="examples", smooth=0.9)    # on the plot: every panel, like Axes.set

Available on axes: set_ylabel, set_ylim, set_yscale, axhline, set(**kw); on panels:
set_title, set_xlabel, set_xlim, set_smooth, axvline, set(**kw), plus the left
axis's setters; on the plot: all of these for every panel, `axhline` (every left
axis, or `metric=` for one) and `axvline` (every panel). Extra kwargs on
`axhline` / `axvline` go to the matplotlib artist. Smoothing is wandb's
time-weighted EMA with the same 0 to 1 weight (`smooth=0.9`), raw values faded
behind.

Images beside curves: `LivePlot.subplots` mirrors `plt.subplots`, and a panel can
hold a picture instead of lines -- loss curves updating every step next to samples
from a generator, in one figure:

    plot, (ax_loss, ax_samples) = LivePlot.subplots(1, 2, total=n_steps, figsize=(11, 4))
    ax_loss.plot("lossD", "lossG")             # which metrics live on this axis
    ax_loss.twinx().plot("D(x)")               # matplotlib's spelling for a right-hand axis
    ...
    ax_samples.imshow(netG(noise), rows=2, vmin=-1, vmax=1)   # replaces the last one

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
               "axvlines", "smooth", "yscale", "yscale2", "kind", "cmap", "legend"}
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
            assert len(spec[lim]) == 2, f"{lim} must be a (low, high) pair"
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
        "yscale": spec.get("yscale", "linear"),  # "linear" or "log", left axis
        "yscale2": spec.get("yscale2", "linear"),  # ... right axis
        "kind": kind,  # "curve" (metric lines) or "image" (one picture, overwritten in place)
        "cmap": spec.get("cmap", "gray"),  # image panels only; matplotlib ignores it for RGB data
        "legend": dict(spec.get("legend") or {}),  # Axes.legend kwargs (loc, ncols, bbox_to_anchor, ...)
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


def _compact_formatter():
    """
    A tick formatter for the x-axis that writes large numbers as 100k, 1.5M, 2B, so the labels of a
    long run (e.g. 300000 examples) don't run into each other. One suffix per axis, chosen from its
    largest tick; an axis whose ticks all stay below 10,000 is labelled as matplotlib would.
    """
    from matplotlib.ticker import ScalarFormatter

    class CompactFormatter(ScalarFormatter):
        def set_locs(self, locs):
            super().set_locs(locs)
            vmin, vmax = sorted(self.axis.get_view_interval()) if self.axis is not None else (0, 0)
            big = max([abs(v) for v in locs if vmin <= v <= vmax] or [0])
            self.scale, self.suffix = next(((10.0**p, x) for p, x in ((9, "B"), (6, "M"), (3, "k")) if big >= max(10.0**p, 1e4)), (1, ""))

        def __call__(self, x, pos=None):
            if not getattr(self, "suffix", ""):
                return super().__call__(x, pos)
            return "0" if x == 0 else self.fix_minus(f"{round(x / self.scale, 6):g}{self.suffix}")

        def get_offset(self):
            return "" if getattr(self, "suffix", "") else super().get_offset()

    return CompactFormatter()


# --------------------------------------------------------------------------- the figure


class _FigureRenderer:
    """
    Owns one matplotlib figure and the metric history it draws. Built on the
    object-oriented API (no pyplot), so it has no global state and works the
    same inside the render process or on the calling thread. `set_layout`
    rebuilds the figure for a new panel list, keeping the history.
    """

    def __init__(self, panels, xlim, layout, hist=None, images=None):
        self.xlim, self.layout = xlim, layout  # xlim = default x range or None; layout = (max_cols, rows, cols, cell_size, dpi, xlabel)
        self.hist = hist if hist is not None else {}
        self.images = dict(images) if images else {}  # panel index -> (the uint8 array it is showing, its x)
        self.axvlines: list[dict] = []  # vertical reference lines (matplotlib axvline kwargs), drawn on every panel
        self.set_layout(panels)

    def set_layout(self, panels):
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        max_cols, rows_opt, cols_opt, cell_size, dpi, xlabel, *rest = self.layout
        gridspec_kw = rest[0] if rest else {}  # width_ratios / height_ratios, from subplots()
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
        axes = self.fig.subplots(rows, cols, squeeze=False, gridspec_kw=gridspec_kw).flatten()
        palette = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        self.lines = {}
        self.raw_lines = {}  # metric -> faded line of the unsmoothed values, on smoothed panels
        self.smooth = {}  # metric -> TWEMA weight in (0, 1) or None
        self.fixed = {}  # axes -> (x fixed?, y fixed?): fixed axes are never autoscaled
        self.panel_cmaps = {i: p["cmap"] for i, p in enumerate(panels)}
        self.panel_titles = {i: p["title"] for i, p in enumerate(panels)}
        self.image_axes = {}  # panel index -> its Axes, for the image panels
        self.image_artists = {}  # ... and the AxesImage drawn on it, so a redraw can set_data in place
        for i, (ax, panel) in enumerate(zip(axes, panels)):
            if panel["kind"] == "image":
                ax.set_title(panel["title"])
                ax.set_axis_off()  # pixel indices along the edge of a tiled grid are just noise
                self.image_axes[i] = ax
                continue
            names = panel["metrics"] + panel["secondary"]
            ax2 = ax.twinx() if panel["secondary"] else None
            ax.set_yscale(panel["yscale"])
            if ax2 is not None:
                ax2.set_yscale(panel["yscale2"])
            window = panel["smooth"] if panel["smooth"] else None  # TWEMA weight in (0, 1)
            assert window is None or 0 < window < 1, f"smooth must be a weight in (0, 1), got {window!r}"
            handles = []
            for j, name in enumerate(names):
                target = ax2 if name in panel["secondary"] else ax
                color = palette[j % len(palette)]
                if window:
                    (self.raw_lines[name],) = target.plot([], [], lw=0.8, color=color, alpha=0.25)
                (line,) = target.plot([], [], lw=1.2, color=color, label=name)
                self.lines[name] = line
                self.smooth[name] = window
                self.hist.setdefault(name, ([], []))
                handles.append(line)
            for target, key in ((ax, "axhlines"), (ax2, "axhlines2")):
                for kw in panel[key] if target is not None else []:
                    handles.append(target.axhline(**{**_REF_LINE_STYLE, **kw}))
                    names = names + [kw["label"]]
            ax.set_title(panel["title"])
            ax.set_xlabel(panel["xlabel"] or xlabel)
            ax.xaxis.set_major_formatter(_compact_formatter())
            xlim = panel["xlim"] or self.xlim
            if xlim:
                ax.set_xlim(*xlim)
            if panel["ylim"]:
                ax.set_ylim(*panel["ylim"])
            self.fixed[ax] = (xlim is not None, panel["ylim"] is not None)
            if names and names != ["_"]:  # (nothing logged yet, or the "waiting for data" placeholder)
                # every panel gets a legend (reference lines included); Panel.legend(**kwargs) moves or styles it
                ax.legend(handles, names, **{"loc": "best", "fontsize": 8, **panel["legend"]})
            if ax2 is not None:
                ax.set_ylabel(panel["ylabel"] or ", ".join(panel["metrics"]))
                ax2.set_ylabel(panel["ylabel2"] or ", ".join(panel["secondary"]))
                if panel["ylim2"]:
                    ax2.set_ylim(*panel["ylim2"])
                self.fixed[ax2] = (xlim is not None, panel["ylim2"] is not None)
            elif panel["ylabel"]:
                ax.set_ylabel(panel["ylabel"])
        for ax in axes[n:]:
            ax.set_visible(False)
        self.axes = list(axes[:n])
        for ax, panel in zip(self.axes, panels):
            for kw in panel["axvlines"]:
                self._draw_axvline(kw, [ax])
        for kw in self.axvlines:
            self._draw_axvline(kw)
        for index in self.images:  # a re-layout rebuilds the figure; put the pictures back
            if index in self.image_axes:
                self._draw_image(index)
        self.fig.tight_layout(pad=0.6)

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
    def add_axvline(self, kw: dict):
        self.axvlines.append(kw)
        self._draw_axvline(kw)

    def _draw_axvline(self, kw, axes=None):
        label = kw.get("label")
        style = {**_REF_LINE_STYLE, "linestyle": ":", **{k: v for k, v in kw.items() if k != "label"}}
        picture = set(self.image_axes.values())  # a vertical line across a picture means nothing
        for ax in ([a for a in self.axes if a not in picture] if axes is None else axes):
            ax.axvline(**style)
            if label:
                ax.text(kw["x"], 0.98, f" {label}", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=7, color="0.3", rotation=90)

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
            title, when = self.panel_titles[index], f"{self.layout[5]} {x:g}"
            ax.set_title(f"{title} ({when})" if title else when)
        artist = self.image_artists.get(index)
        if artist is not None and artist.get_array().shape == arr.shape:
            artist.set_data(arr)  # same size as last time: no new artist, no rescale
            return
        if artist is not None:
            artist.remove()
        cmap = self.panel_cmaps.get(index, "gray")
        self.image_artists[index] = ax.imshow(arr, cmap=cmap, interpolation="nearest")

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
        for line in self.lines.values():
            fixed_x, fixed_y = self.fixed[line.axes]
            line.axes.relim()
            line.axes.autoscale_view(scalex=not fixed_x, scaley=not fixed_y)
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
            elif item[0] == "axvline":
                renderer.add_axvline(item[1])
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


def _lim(a, b):
    """Accept matplotlib's forms: set_xlim((lo, hi)), set_xlim(lo, hi), set_xlim(lo) / set_xlim(right=hi) not supported."""
    if b is None and isinstance(a, (tuple, list)):
        a, b = a
    assert a is not None and b is not None, "give both limits, e.g. set_ylim(0, 1)"
    return (float(a), float(b))


class _Axis:
    """One y-axis of a panel: the left one, or the right-hand one.

    You do not create these. `plot["acc"]` returns the axis that holds the metric `acc`, whichever
    side it is on. `panel.left`, `panel.right` and `panel.twinx()` return a panel's axes directly.

    The y setters (`set_ylabel`, `set_ylim`, `set_yscale`, `axhline`) act on this axis only. The
    other setters (`set_title`, `set_xlabel`, `set_xlim`, `set_smooth`, `axvline`, `legend`) act on
    the panel the axis belongs to, because a panel's two y-axes share one title, x-axis and legend.

    Setters can be called before the loop or during it. The figure is laid out again on the next
    frame.

    Example:
        ```python
        plot = LivePlot(loader, "loss | acc")
        plot["acc"].set_ylim(0, 1)            # the right-hand axis
        plot["loss"].set_yscale("log")        # the left-hand axis
        plot["loss"].axhline(0.1, "target")
        ```
    """

    def __init__(self, plot, index: int, right: bool):
        self._plot, self._index, self._right = plot, index, right

    @property
    def panel(self):
        """The `Panel` this axis belongs to."""
        return Panel(self._plot, self._index)

    @property
    def metrics(self) -> list:
        """Names of the metrics drawn on this axis, as a new list."""
        return list(self._plot._specs[self._index]["secondary" if self._right else "metrics"])

    def _set(self, key, value):
        self._plot._specs[self._index][key + ("2" if self._right else "")] = value
        self._plot._send_layout()

    def plot(self, *metrics):
        """Put metrics on this axis.

        This names metrics. It takes no data: the values arrive later through `LivePlot.log`.
        A metric already on this axis is not added twice. If the panel still has its automatic
        title (the metric names joined by " / "), the title is updated to include the new names.

        Args:
            *metrics (str): One or more metric names.

        Returns:
            _Axis: This axis, so calls can be chained.

        Raises:
            AssertionError: If no name is given, a name is not a string, or the panel is showing
                an image.

        Example:
            ```python
            plot, (ax_loss, ax_img) = LivePlot.subplots(1, 2)
            ax_loss.plot("lossD", "lossG")       # two curves on the left axis
            ax_loss.twinx().plot("D(x)")         # one curve on the right axis
            ```
        """
        assert metrics, "plot() needs at least one metric name"
        assert all(isinstance(m, str) for m in metrics), \
            f"plot() takes metric names, not data: {[m for m in metrics if not isinstance(m, str)]!r}"
        panel = self._plot._specs[self._index]
        assert panel["kind"] == "curve", "this panel is showing an image; use a different panel for curves"
        key = "secondary" if self._right else "metrics"
        auto = " / ".join(panel["metrics"] + panel["secondary"])
        was_auto = panel["title"] in (auto, "")  # unnamed, or named after the metrics it had
        panel[key].extend(m for m in metrics if m not in panel[key])
        if was_auto:  # a title set with set_title() survives a later plot() call
            panel["title"] = " / ".join(panel["metrics"] + panel["secondary"])
        self._plot._place(metrics)
        return self

    def set_ylabel(self, ylabel):
        """Set this axis's y label.

        Without one, a panel with two y-axes labels each with its metric names, and a panel with
        one y-axis has no y label.

        Args:
            ylabel (str): The label text.
        """
        self._set("ylabel", str(ylabel))

    def set_ylim(self, bottom=None, top=None):
        """Fix this axis's y range. Without it the range follows the data.

        Args:
            bottom (float | tuple[float, float]): The lower limit, or a `(bottom, top)` pair.
            top (float): The upper limit. Both limits are required: `set_ylim(0)` raises.

        Example:
            ```python
            plot["acc"].set_ylim(0, 1)
            plot["acc"].set_ylim((0, 1))   # the same
            ```
        """
        self._set("ylim", _lim(bottom, top))

    def set_yscale(self, value):
        """Set this axis's y scale.

        Args:
            value (str): `"linear"` or `"log"`. Anything else raises `AssertionError`.
        """
        assert value in ("linear", "log"), f"yscale must be 'linear' or 'log', got {value!r}"
        self._set("yscale", value)

    def axhline(self, y, label=None, **kwargs):
        """Draw a horizontal reference line on this axis.

        The line is dashed grey by default and always gets an entry in the panel's legend.

        Args:
            y (float): Where to draw the line, in this axis's units.
            label (str | None): The legend text. Defaults to the value of `y`.
            **kwargs (Any): Passed to matplotlib's `Axes.axhline` (`color`, `linestyle`, `linewidth`,
                `alpha`, ...).

        Example:
            ```python
            plot["loss"].axhline(math.log(d_vocab), "uniform", color="red")
            ```
        """
        line = {**kwargs, "y": float(y), "label": str(label) if label is not None else f"{float(y):g}"}
        self._plot._specs[self._index]["axhlines2" if self._right else "axhlines"].append(line)
        self._plot._send_layout()

    def set(self, **kwargs):
        """Call several setters at once, like matplotlib's `Axes.set`.

        Each keyword `name=value` calls `set_name(value)`.

        Args:
            **kwargs (Any): Any of `ylabel`, `ylim`, `yscale` (this axis) and `title`, `xlabel`, `xlim`,
                `smooth` (the panel). Limits are `(low, high)` pairs. An unknown name raises
                `AttributeError`.

        Example:
            ```python
            plot["acc"].set(ylabel="test accuracy", ylim=(0, 1), title="evaluation")
            ```
        """
        for key, value in kwargs.items():
            getattr(self, f"set_{key}")(value)

    # panel-level setters, so plot["acc"].set_title(...) works as it would on a matplotlib Axes
    def set_title(self, label):
        """Set the title of this axis's panel. Same as `Panel.set_title`."""
        self.panel.set_title(label)

    def legend(self, **kwargs):
        """Move or restyle the legend of this axis's panel. Same as `Panel.legend`."""
        self.panel.legend(**kwargs)

    def set_xlabel(self, xlabel):
        """Set the x label of this axis's panel. Same as `Panel.set_xlabel`."""
        self.panel.set_xlabel(xlabel)

    def set_xlim(self, left=None, right=None):
        """Fix the x range of this axis's panel. Same as `Panel.set_xlim`."""
        self.panel.set_xlim(left, right)

    def axvline(self, x=None, label=None, **kwargs):
        """Draw a vertical line on this axis's panel. Same as `Panel.axvline`."""
        self.panel.axvline(x, label, **kwargs)

    def set_smooth(self, weight):
        """Set the smoothing weight of this axis's panel. Same as `Panel.set_smooth`."""
        self.panel.set_smooth(weight)


class Panel:
    """One panel of the grid: a set of curves with one or two y-axes, or one image.

    You do not create these. Get one from `plot.panels[i]`, from `plot[metric].panel`, or from
    `LivePlot.subplots`, which returns the panels of a fixed grid.

    A panel has a title, an x-axis, a legend and a smoothing weight, and up to two y-axes:
    `panel.left` and `panel.right`. The y setters on the panel itself (`set_ylabel`, `set_ylim`,
    `set_yscale`, `axhline`) act on the left axis.

    Setters can be called before the loop or during it. The figure is laid out again on the next
    frame.

    Example:
        ```python
        plot = LivePlot(loader, "loss | acc", "lr")
        plot.panels[0].set_title("training")
        plot.panels[0].right.set_ylim(0, 1)   # the axis holding acc
        plot.panels[1].set_yscale("log")      # the lr panel
        ```
    """

    def __init__(self, plot, index: int):
        self._plot, self._index = plot, index

    @property
    def spec(self) -> dict:
        """The panel's settings as a dict (title, metrics, limits, ...). Mostly for debugging."""
        return self._plot._specs[self._index]

    @property
    def metrics(self) -> list:
        """Names of the metrics on this panel: the left axis's first, then the right axis's."""
        return list(self.spec["metrics"]) + list(self.spec["secondary"])

    @property
    def left(self) -> _Axis:
        """The left y-axis."""
        return _Axis(self._plot, self._index, right=False)

    @property
    def right(self) -> _Axis:
        """The right-hand y-axis.

        Raises `AssertionError` if the panel has no metrics on a right-hand axis. Use `twinx()`
        to get the right-hand axis of a panel that has none yet.
        """
        assert self.spec["secondary"], "this panel has no right-hand axis (no metrics after '|')"
        return _Axis(self._plot, self._index, right=True)

    def twinx(self) -> _Axis:
        """Return this panel's right-hand y-axis, like matplotlib's `Axes.twinx`.

        Unlike `right`, this works on a panel with nothing on its right-hand axis yet. The axis
        appears in the figure once a metric is put on it.

        Returns:
            The right-hand axis.

        Example:
            ```python
            plot, ax = LivePlot.subplots()
            ax.plot("loss")
            ax.twinx().plot("acc")   # acc on the right-hand axis
            ```
        """
        return _Axis(self._plot, self._index, right=True)

    def plot(self, *metrics) -> _Axis:
        """Put metrics on this panel's left axis.

        This names metrics. It takes no data: the values arrive later through `LivePlot.log`.

        Args:
            *metrics (str): One or more metric names.

        Returns:
            The left axis.

        Example:
            ```python
            plot, (ax_loss, ax_acc) = LivePlot.subplots(1, 2)
            ax_loss.plot("train_loss", "val_loss")   # two curves on one axis
            ax_acc.plot("acc")
            ```
        """
        return self.left.plot(*metrics)

    def imshow(self, x, *, rows=None, cols=None, griddim=None, vmin=None, vmax=None, scale_each=False,
               channels=None, cmap="gray", padding=0, pad_value=0, max_images=64, max_cols=8):
        """Show an image, or a batch of images as a grid, on this panel.

        Each call replaces the previous picture. Once the plot has a step (a loop is wrapped or
        something has been logged), the panel's title shows the step the picture was drawn at.

        Accepted shapes: `(H, W)`, `(1, H, W)`, `(H, W, 3|4)`, `(B, H, W)`, `(B, 1|3|4, H, W)` and
        `(B, H, W, 1|3|4)`. `(3, H, W)` and `(4, H, W)` are ambiguous and raise `ValueError`;
        pass `channels`. A 3-d array whose last dimension is 3 or 4 is read as one colour image.

        Args:
            x (ArrayLike): A torch tensor, numpy array, or anything `numpy.asarray` accepts. Tensors are
                detached and moved to the CPU for you.
            rows (int | None): Number of grid rows. The number of columns is inferred.
            cols (int | None): Number of grid columns. The number of rows is inferred.
            griddim (tuple[int, int] | None): `(rows, cols)`, fixing both. Missing cells are left blank and extra images
                are dropped. Do not combine with `rows` or `cols`.
            vmin (float | None): Value at the dark end of the range. Defaults to the minimum of the batch.
            vmax (float | None): Value at the bright end of the range. Defaults to the maximum of the batch. Fixing `vmin` and
                `vmax` stops the brightness changing from frame to frame. They apply to colour
                images too. `uint8` input is shown as it is unless one of them is given.
            scale_each (bool): Scale each image to its own range instead of the batch's.
            channels (str | None): How to read a 3-d input: `"first"` for `(C, H, W)`, `"last"` for
                `(H, W, C)`, `"none"` for a batch of grayscale `(B, H, W)`.
            cmap (str): Matplotlib colormap for grayscale images.
            padding (int): Pixels of `pad_value` around every image in a grid. 0 puts them edge to edge.
            pad_value (int): The value (0 to 255) of padding pixels and blank cells.
            max_images (int | None): Show only this many images from the batch, with a warning. `None` shows
                all of them.
            max_cols (int | None): Widest grid to choose when neither `rows`, `cols` nor `griddim` is given.
                The default grid is near-square: 3 images give 2x2, 8 give 3x3, 64 give 8x8.

        Returns:
            Panel: This panel.

        Raises:
            AssertionError: If the panel already holds curves.
            ValueError: If the shape is ambiguous or not 2-, 3- or 4-dimensional.

        Example:
            ```python
            plot, (ax_loss, ax_samples) = LivePlot.subplots(1, 2)
            ax_loss.plot("lossD", "lossG")
            ax_samples.imshow(netG(fixed_noise), rows=2, vmin=-1, vmax=1)
            ```
        """
        from ._images import to_grid

        spec = self.spec
        assert not (spec["metrics"] or spec["secondary"]), \
            f"panel {self._index} is showing curves ({' '.join(spec['metrics'] + spec['secondary'])}); " \
            f"use a different panel for the image"
        arr = to_grid(x, rows=rows, cols=cols, griddim=griddim, vmin=vmin, vmax=vmax, scale_each=scale_each,
                      channels=channels, padding=padding, pad_value=pad_value, max_images=max_images,
                      max_cols=max_cols)
        if spec["kind"] != "image" or spec["cmap"] != cmap:
            spec["kind"], spec["cmap"] = "image", cmap
            self._plot._send_layout()
        self._plot._send_image(self._index, arr)
        return self

    def _set(self, key, value):
        self.spec[key] = value
        self._plot._send_layout()

    def set_title(self, label):
        """Set the panel's title.

        Without one, the title is the panel's metric names joined by " / ".

        Args:
            label (str): The title text.
        """
        self._set("title", str(label))

    def legend(self, **kwargs):
        """Move or restyle the panel's legend.

        A panel has one legend. It lists the curves of both y-axes and the horizontal reference
        lines. Each call replaces the settings of the previous call.

        Args:
            **kwargs (Any): Passed to matplotlib's `Axes.legend` (`loc`, `ncols`, `bbox_to_anchor`,
                `fontsize`, ...). The defaults are `loc="best"` and `fontsize=8`.

        Example:
            ```python
            # under the panel, in three columns
            ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncols=3)
            ```
        """
        self._set("legend", kwargs)

    def set_xlabel(self, xlabel):
        """Set the panel's x label. Without one, the label is the plot's `unit` ("step" by default).

        Args:
            xlabel (str): The label text.
        """
        self._set("xlabel", str(xlabel))

    def set_xlim(self, left=None, right=None):
        """Fix the panel's x range.

        This overrides the range that `fixed_x=True` gives the plot. Without either, the x range
        follows the data.

        Args:
            left (float | tuple[float, float]): The lower limit, or a `(left, right)` pair.
            right (float): The upper limit. Both limits are required.
        """
        self._set("xlim", _lim(left, right))

    def set_smooth(self, weight):
        """Smooth every curve on this panel.

        The curve drawn is wandb's time-weighted exponential moving average. The raw values stay
        visible as a faded line behind it. `plot.data` always holds the raw values.

        Args:
            weight (float): The smoothing weight, from 0 up to but not including 1. Larger is smoother.
                0 turns smoothing off.
        """
        self._set("smooth", _check_smooth(weight))

    def axvline(self, x=None, label=None, **kwargs):
        """Draw a vertical line on this panel only.

        The line is dotted grey by default.

        Args:
            x (float | None): Where to draw the line. Defaults to the plot's current step (`plot.step`) at the
                time of the call.
            label (str | None): Text written along the line. It does not go in the legend.
            **kwargs (Any): Passed to matplotlib's `Axes.axvline` (`color`, `linestyle`, ...).

        Example:
            ```python
            plot.panels[0].axvline(label="lr drop")         # at the current step
            plot.panels[1].axvline(2000, "checkpoint")      # at x = 2000
            ```
        """
        line = {**kwargs, "x": float(self._plot.step if x is None else x), "label": label}
        self.spec["axvlines"].append(line)
        self._plot._send_layout()

    def set(self, **kwargs):
        """Call several setters at once, like matplotlib's `Axes.set`.

        Each keyword `name=value` calls `set_name(value)`.

        Args:
            **kwargs (Any): Any of `title`, `xlabel`, `xlim`, `smooth`, `ylabel`, `ylim`, `yscale`.
                Limits are `(low, high)` pairs. The y settings go to the left axis. An unknown
                name raises `AttributeError`.

        Example:
            ```python
            plot.panels[0].set(title="training", xlabel="examples", ylim=(0, 1))
            ```
        """
        for key, value in kwargs.items():
            getattr(self, f"set_{key}")(value)

    # y-setters act on the left axis, as on a matplotlib Axes
    def set_ylabel(self, ylabel):
        """Set the left axis's y label. Same as `panel.left.set_ylabel(ylabel)`."""
        self.left.set_ylabel(ylabel)

    def set_ylim(self, bottom=None, top=None):
        """Fix the left axis's y range. Same as `panel.left.set_ylim(bottom, top)`."""
        self.left.set_ylim(bottom, top)

    def set_yscale(self, value):
        """Set the left axis's y scale, `"linear"` or `"log"`. Same as `panel.left.set_yscale(value)`."""
        self.left.set_yscale(value)

    def axhline(self, y, label=None, **kwargs):
        """Draw a horizontal reference line on the left axis. Same as `panel.left.axhline(...)`."""
        self.left.axhline(y, label, **kwargs)

    def __repr__(self):
        return f"Panel({self._index}: {' '.join(self.spec['metrics'])}{' | ' + ' '.join(self.spec['secondary']) if self.spec['secondary'] else ''})"


class _PanelGrid:
    """The 2-d grid of panels that `LivePlot.subplots` returns for two or more rows and columns.

    It is indexed like matplotlib's array of axes: `axes[0][1]` and `axes[0, 1]` are the same
    panel. Iterating it yields rows, each a list of panels, so it unpacks like a nested list.

    Example:
        ```python
        plot, axes = LivePlot.subplots(2, 2)
        axes[0, 0].plot("loss")
        (ax_a, ax_b), (ax_c, ax_d) = axes
        ```
    """

    def __init__(self, rows: list):
        self._rows = rows

    def __getitem__(self, key):
        if isinstance(key, tuple):
            row, col = key
            return self._rows[row][col]
        return self._rows[key]

    def __len__(self):
        return len(self._rows)

    def __iter__(self):
        return iter(self._rows)

    @property
    def flat(self) -> list:
        """Every panel as one list, row by row."""
        return [panel for row in self._rows for panel in row]

    @property
    def shape(self) -> tuple:
        """`(rows, columns)`."""
        return (len(self._rows), len(self._rows[0]))

    def _squeezed(self):
        """matplotlib's squeeze: 1x1 -> the panel, 1xN or Nx1 -> a flat list, else the grid."""
        flat = self.flat
        if len(flat) == 1:
            return flat[0]
        rows, cols = self.shape
        return flat if rows == 1 or cols == 1 else self

    def __repr__(self):
        return f"_PanelGrid({self.shape[0]}x{self.shape[1]})"


# --------------------------------------------------------------------------- the handle


class LivePlot:
    """A live plot of the metrics you log, with a tqdm progress bar underneath.

    Metrics are named when you log them: `plot.log(loss=0.3)`. Each name becomes a curve, and the
    figure redraws in the notebook's output cell while the loop runs.

    The x value of a point is the plot's step counter at the time of the `log` call. `log` itself
    does not advance the counter. Advance it in one of three ways:

    1. Iterate the plot: `for batch in LivePlot(loader)`. Each item advances the counter by one.
    2. Wrap an inner loop with `plot(loader)`, or call `plot.update()` once per step. Use these
       for nested epoch and batch loops.
    3. Pass the x value yourself: `plot.log(step, loss=0.3)`.

    If you do none of these, every point is logged at the same x (`initial`, 0 by default) and
    the curve collapses to a vertical line.

    Outside a notebook nothing is drawn, but `data` still collects everything.

    Example:
        ```python
        from liveplot import LivePlot

        plot = LivePlot(loader, "loss | acc", "lr")   # two panels; acc on a right-hand axis
        plot["acc"].set_ylim(0, 1)
        for batch in plot:                            # wrapping the loader advances x
            loss, acc = train_step(batch)
            plot.log(loss=loss, acc=acc, lr=get_lr())
        ```

    Attributes:
        data (dict[str, tuple[list, list]]): The full history, `{metric: (xs, values)}`. It is
            filled in whether or not anything is drawn.
        latest (dict[str, float]): The most recent value of each metric. The progress bar shows
            these.
        n (int): Items consumed so far, across every wrapped loop and `update()` call. Same
            meaning as tqdm's `n`.
        total (int | None): The `total` argument, or `len(iterable)` if that was not given.
        initial (int | float): The `initial` argument.
        unit (str): The `unit` argument.
        unit_scale (int | float): The `unit_scale` argument.
        fixed_x (bool): The `fixed_x` argument.
        refresh_seconds (float): The `refresh_seconds` argument.
        smooth (float | None): The default smoothing weight for panels.
        mode (str): `"process"` when a separate render process draws the figure, `"thread"` when
            that process could not start or died, and drawing happens inside `log`, `"off"` when nothing
            is drawn (`show=False`, or not in a notebook, and not recording).
        frames (list[tuple[float, bytes]]): With `record` set, every frame shown so far, as
            `(seconds since the plot was created, PNG bytes)`.
        last_png (bytes | None): The most recent frame, as PNG bytes.
        axvlines (list[dict]): The plot-wide vertical lines added with `axvline`.
    """

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
        show: bool = True,
        progress: bool = True,
        desc: str | None = None,
        record: bool | str = False,
        smooth: float | None = None,
        fixed_x: bool = False,
    ):
        """Create a plot: `LivePlot([iterable,] *panels, **options)`.

        Args:
            *args (Any): An optional iterable, then any number of panel layouts.

                The iterable is anything you would loop over: a `range`, a `DataLoader`, an
                existing tqdm bar. If the first argument is not a string or a dict, it is taken
                as the iterable. Iterating the plot then yields its items, advances x by one per
                item, shows a progress bar, and calls `finish()` when the loop ends. An existing
                tqdm bar is reused, not wrapped in a second one. Leave the iterable out for
                nested loops and use `plot(inner_iterable)` or `plot.update()`.

                Each layout string is one panel. Names separated by spaces share the left
                y-axis. Names after a `|` go on a right-hand y-axis: `"loss"`, `"train_loss
                val_loss"`, `"loss | acc"`. With no layout, every metric you log shares one
                panel. With a layout, a metric that no string mentions gets a panel of its own.
                A dict such as `{"metrics": ["acc"], "ylim": (0, 1)}` is also accepted in place
                of a string, but the setters are the intended way to configure a panel.
            total: Number of items in the whole run. Optional. Defaults to `len(iterable)` when
                there is an iterable with a length. It does two things. It is the length of the
                progress bar when the bar cannot get one from the iterable: the bar that
                `update()` opens, and the bar for an iterable that has no `len()`. And with
                `fixed_x=True` it fixes the x range. With the default `fixed_x=False` it does
                not change the figure. Bars made by `plot(iterable)` use `len(iterable)`.
            initial: The x value of the first item. Use `initial=1` for a 1-based x-axis.
            unit: Name of one item. It is the default x label and the unit on the progress bar.
            unit_scale: How much x advances per item. `x = initial + n * unit_scale`. For
                example `unit="examples", unit_scale=batch_size` plots against examples seen.
            refresh_seconds: Minimum time between redraws. Points that arrive in between are
                drawn together in the next frame. 0 redraws whenever new data arrives.
            max_cols: Widest grid of panels. `None` gives a near-square grid.
            rows: Fix the number of grid rows.
            cols: Fix the number of grid columns.
            cell_size: `(width, height)` of one panel, in inches.
            dpi: Resolution of the rendered image.
            show: `False` draws nothing. Metrics are still collected in `data` and shown on the
                progress bar.
            progress: `False` shows no progress bar.
            desc: Text before the progress bar, for the bar made by iterating the plot or by
                `update()`. Bars made by `plot(iterable, desc=...)` take their own.
            record: `True` keeps every rendered frame in `frames`. A path such as `"run.gif"`
                also writes an animated GIF there at `finish()`. Recording works outside a
                notebook too.
            smooth: Default smoothing weight for every panel, from 0 up to but not including 1.
                See `set_smooth`.
            fixed_x: `False` (the default) lets the x-axis grow with the data. `True`, with a
                known `total`, fixes the x-axis to `(initial, initial + total * unit_scale)` from
                the first frame, so the curve grows into an axis that already spans the run.
                It has no effect without a `total`.
        """
        iterable, specs = (args[0], args[1:]) if args and not isinstance(args[0], (str, dict)) else (None, args)
        self.smooth = _check_smooth(smooth)  # default TWEMA weight for panels that don't set their own (wandb's smoothing slider)
        self._panel_defaults: dict = {}  # plot-level set_*() values, applied to panels created later
        self._specs = [self._with_defaults(_normalise_panel(p)) for p in specs]
        self._explicit_layout = bool(self._specs)
        self._placed = {n for p in self._specs for n in p["metrics"] + p["secondary"]}
        self._layout = (max_cols, rows, cols, cell_size, dpi, unit)
        self._iterable, self._bar, self._desc, self._progress = iterable, None, desc, progress
        self._own_bar = None  # the bar update() opened, which finish() closes
        if total is None and iterable is not None:
            try:
                total = len(iterable)
            except TypeError:
                pass
        self.total, self.initial, self.unit, self.unit_scale = total, initial, unit, unit_scale
        self.fixed_x = fixed_x
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
        self.axvlines: list[dict] = []  # vertical reference lines added with axvline()
        self._images: dict[int, object] = {}  # panel index -> the uint8 array it is showing
        self._image_steps: dict[int, object] = {}  # panel index -> the x it was shown at
        self._fixed_grid = False  # set by subplots(): grow a panel's metrics, never the grid
        self._t0 = time.monotonic()
        self._done = False
        self._proc = self._renderer = self._inbox = self._outbox = None
        self._last_draw = self._last_postfix = 0.0
        self._n_logs = 0
        self._handle = self._make_display_handle() if show else None
        if self._handle is None and not self._record:
            self.mode = "off"  # show=False, or not in a notebook: nothing to draw on, just collect metrics
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
        for key, value in self._panel_defaults.items():  # plot-level set_xlabel(...) etc. made before the panel existed
            panel[key] = value
        return panel

    @classmethod
    def subplots(cls, nrows: int = 1, ncols: int = 1, *, figsize=None, squeeze: bool = True,
                 width_ratios=None, height_ratios=None, iterable=None, **kwargs):
        """Create a plot with a fixed grid of empty panels, like `plt.subplots`.

        Fill each panel with `ax.plot("loss", ...)` for curves or `ax.imshow(tensor)` for an
        image. The grid never grows: a metric you log without placing it joins the first panel
        that is still empty, or the first curve panel if none is empty.

        Args:
            nrows: Number of grid rows.
            ncols: Number of grid columns.
            figsize (tuple[float, float] | None): `(width, height)` of the whole figure in inches. Without it, each panel
                is `cell_size` inches.
            squeeze: `True` returns one `Panel` for a 1x1 grid, a flat list for a single row or
                column, and a 2-d grid otherwise. `False` always returns the 2-d grid.
            width_ratios (Sequence[float] | None): Relative column widths, one per column, e.g. `(1, 1.3)`.
            height_ratios (Sequence[float] | None): Relative row heights, one per row.
            iterable (Iterable | None): An iterable to wrap, as in `LivePlot(iterable)`.
            **kwargs (Any): Any other `LivePlot` option, such as `total` or `smooth`.

        Returns:
            tuple[LivePlot, Panel | list[Panel] | _PanelGrid]: `(plot, axes)`, the plot and its panels.

        Example:
            ```python
            plot, (ax_loss, ax_samples) = LivePlot.subplots(1, 2, figsize=(11, 4))
            ax_loss.plot("lossD", "lossG")           # two metrics on the left axis
            ax_loss.twinx().plot("D_real")           # one on the right-hand axis
            for batch in plot(loader):
                plot.log(lossD=..., lossG=..., D_real=...)
                if plot.n % 250 == 0:
                    ax_samples.imshow(netG(fixed_noise), rows=2, vmin=-1, vmax=1)
            plot.finish()
            ```
        """
        assert nrows >= 1 and ncols >= 1, f"need at least a 1x1 grid, got {nrows}x{ncols}"
        kwargs.setdefault("rows", nrows)
        kwargs.setdefault("cols", ncols)
        if figsize is not None:
            kwargs["cell_size"] = (figsize[0] / ncols, figsize[1] / nrows)
        plot = cls(*([] if iterable is None else [iterable]), **kwargs)
        gridspec_kw = {k: list(v) for k, v in (("width_ratios", width_ratios), ("height_ratios", height_ratios)) if v is not None}
        if gridspec_kw:
            assert len(gridspec_kw.get("width_ratios", [0] * ncols)) == ncols, "width_ratios needs one entry per column"
            assert len(gridspec_kw.get("height_ratios", [0] * nrows)) == nrows, "height_ratios needs one entry per row"
            plot._layout = (*plot._layout[:6], gridspec_kw)
        plot._specs.extend(
            plot._with_defaults(_normalise_panel({"metrics": []}, allow_empty=True)) for _ in range(nrows * ncols)
        )
        plot._explicit_layout = True
        plot._fixed_grid = True
        plot._send_layout()
        grid = _PanelGrid([[Panel(plot, r * ncols + c) for c in range(ncols)] for r in range(nrows)])
        return plot, (grid._squeezed() if squeeze else grid)

    # -- panels and axes, addressed like matplotlib -----------------------------------

    @property
    def panels(self) -> list:
        """The panels of the grid, in order, as a list of `Panel`.

        Example:
            ```python
            plot.panels[0].set_title("training")
            ```
        """
        return [Panel(self, i) for i in range(len(self._specs))]

    def __getitem__(self, metric: str) -> _Axis:
        """Return the y-axis that holds a metric: `plot["acc"]`.

        This finds the axis whichever side it is on, so the same spelling configures a left or a
        right-hand axis. A metric that no panel holds yet is placed now, as `log` would place it.

        Args:
            metric: A metric name.

        Returns:
            The axis object. Its setters are `set_ylim`, `set_ylabel`, `set_yscale`, `axhline`
            and `set`, plus the panel's setters.

        Example:
            ```python
            plot = LivePlot(loader, "loss | acc")
            plot["acc"].set_ylim(0, 1)
            plot["loss"].axhline(0.1, "target")
            ```
        """
        if metric not in self._placed:
            self.data.setdefault(metric, ([], []))
            self._extend_layout([metric])
        for i, spec in enumerate(self._specs):
            if metric in spec["metrics"]:
                return _Axis(self, i, right=False)
            if metric in spec["secondary"]:
                return _Axis(self, i, right=True)
        raise KeyError(metric)

    def _set_all(self, key, value):
        """A plot-level setter: every existing panel, and every panel created later."""
        self._panel_defaults[key] = value
        for spec in self._specs:
            spec[key] = value
        self._send_layout()

    def set_title(self, label):
        """Give every panel the same title, including panels created later.

        To title one panel, use `plot.panels[i].set_title(...)` or `plot[metric].set_title(...)`.

        Args:
            label (str): The title text.
        """
        self._set_all("title", str(label))

    def set_xlabel(self, xlabel):
        """Set the x label of every panel, including panels created later.

        Args:
            xlabel (str): The label text.
        """
        self._set_all("xlabel", str(xlabel))

    def set_xlim(self, left=None, right=None):
        """Fix the x range of every panel, including panels created later.

        Args:
            left (float | tuple[float, float]): The lower limit, or a `(left, right)` pair.
            right (float): The upper limit. Both limits are required.
        """
        self._set_all("xlim", _lim(left, right))

    def set_ylabel(self, ylabel):
        """Set the left y label of every panel, including panels created later.

        Args:
            ylabel (str): The label text.
        """
        self._set_all("ylabel", str(ylabel))

    def set_ylim(self, bottom=None, top=None):
        """Fix the left y range of every panel, including panels created later.

        Right-hand axes are not changed. Use `plot[metric].set_ylim(...)` for those.

        Args:
            bottom (float | tuple[float, float]): The lower limit, or a `(bottom, top)` pair.
            top (float): The upper limit. Both limits are required.
        """
        self._set_all("ylim", _lim(bottom, top))

    def set_yscale(self, value):
        """Set the left y scale of every panel, including panels created later.

        Args:
            value (str): `"linear"` or `"log"`.
        """
        assert value in ("linear", "log"), f"yscale must be 'linear' or 'log', got {value!r}"
        self._set_all("yscale", value)

    def set_smooth(self, weight):
        """Smooth the curves of every panel, including panels created later.

        The curve drawn is wandb's time-weighted exponential moving average, with the raw values
        as a faded line behind it. `data` always holds the raw values.

        Args:
            weight (float): The smoothing weight, from 0 up to but not including 1. Larger is smoother.
                0 turns smoothing off.
        """
        self.smooth = _check_smooth(weight)
        self._set_all("smooth", weight)

    def set(self, **kwargs):
        """Call several plot-level setters at once, like matplotlib's `Axes.set`.

        Each keyword `name=value` calls `set_name(value)`, so it applies to every panel.

        Args:
            **kwargs (Any): Any of `title`, `xlabel`, `xlim`, `ylabel`, `ylim`, `yscale`, `smooth`.
                Limits are `(low, high)` pairs. An unknown name raises `AttributeError`.

        Example:
            ```python
            plot.set(xlabel="examples", yscale="log", smooth=0.9)
            ```
        """
        for key, value in kwargs.items():
            getattr(self, f"set_{key}")(value)

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
        """The fixed x-axis range, or `None` when the x-axis follows the data.

        It is `(initial, initial + total * unit_scale)` when `fixed_x=True` and `total` is known,
        and `None` otherwise.
        """
        return (self.initial, self.initial + self.total * self.unit_scale) if self.total and self.fixed_x else None

    @property
    def step(self):
        """The x value that a `log()` call without an explicit step gets right now.

        Once a loop is wrapped or `update()` has been called, this is
        `initial + n * unit_scale`. Before that, it is the step of the last `log(step, ...)`
        call, or `initial` if there has been none.
        """
        return self.initial + self.n * self.unit_scale if self._wrapping else self._last_x

    def __call__(self, iterable, **tqdm_kwargs):
        """Wrap a loop so that each item advances the x-axis by one step: `plot(loader)`.

        Use this for nested loops. Create the plot once, then wrap the inner loop on every epoch.
        The step count continues from one wrapped loop to the next, and each gets its own
        progress bar of length `len(iterable)`. The count is 0 inside the body for the first
        item, like `for step in range(n)`. The plot is not finished when the loop ends, so call
        `finish()` yourself or use `with`.

        Args:
            iterable (Iterable): What to loop over. An existing tqdm bar is reused.
            **tqdm_kwargs (Any): Passed to tqdm, e.g. `desc="epoch 3"`.

        Yields:
            Any: The items of `iterable`.

        Example:
            ```python
            plot = LivePlot("loss", "acc")
            for epoch in range(epochs):
                for batch in plot(loader, desc=f"epoch {epoch}"):
                    plot.log(loss=train_step(batch))     # one point per batch
                plot.log(acc=evaluate())                 # one point per epoch
            plot.finish()
            ```
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
        """Advance the x-axis by hand, like tqdm's `update`.

        Use this when you do not want to wrap the loop. Call it once per step, after logging
        that step's metrics. The first call opens one progress bar for the whole run, of length
        `total` if you gave one, and `finish()` closes it.

        Args:
            n (int): How many items to advance by.

        Example:
            ```python
            plot = LivePlot(total=epochs * len(loader))   # total only sets the bar's length
            for epoch in range(epochs):
                plot.set_description(f"epoch {epoch}")
                for batch in loader:
                    plot.log(loss=train_step(batch))
                    plot.update()
            plot.finish()
            ```
        """
        self._wrapping = True
        if self._bar is None and self._own_bar is None:
            self._own_bar = self._bar = self._make_bar(None, {"desc": self._desc} if self._desc else {})
        self.n += n
        if self._own_bar is not None:
            self._own_bar.update(n)

    def set_description(self, desc=None, refresh=True):
        """Set the text before the progress bar, like tqdm's `set_description`.

        Args:
            desc (str | None): The text, e.g. `f"epoch {epoch}"`.
            refresh (bool): Redraw the bar straight away.
        """
        self._desc = desc
        if self._bar is not None:
            self._bar.set_description(desc, refresh=refresh)

    def __iter__(self):
        """Iterate over the iterable given to the constructor: `for batch in plot`.

        Each item advances the x-axis by one step, a progress bar is shown, and `finish()` is
        called when the loop ends, including when it ends with `break` or an exception. Raises
        `TypeError` if the plot was created without an iterable.
        """
        if self._iterable is None:
            raise TypeError("nothing to iterate: use LivePlot(iterable, ...), or plot(iterable) inside your loop")
        try:
            kwargs = {"desc": self._desc} if self._desc else {}
            yield from self(self._iterable, **kwargs)
        finally:
            self.finish()

    def log(self, *args, **metrics):
        """Record one value for each of one or more metrics.

        A metric is created the first time you log it. Each call adds one point per metric, at
        the plot's current x (`plot.step`). Metrics can be logged at different rates: each keeps
        its own list of points.

        `log` does not advance x. Wrap the loop (`for batch in plot` or `plot(loader)`), call
        `plot.update()`, or pass the step as the first argument. Otherwise every point has the
        same x.

        Args:
            *args (Any): Optionally the x value for this call, then optionally a dict of metrics:
                `log(step, loss=0.3)`, `log(step, {"loss": 0.3})` or `log({"loss": 0.3})`. In a
                wrapped loop an explicit step applies to this call only. With no wrapped loop
                it also becomes the x of later calls that give no step.
            **metrics (float): Metric values by name: `log(loss=0.3, acc=0.9)`. A value is anything
                `float()` accepts, including a one-element tensor.

        Example:
            ```python
            plot = LivePlot(range(1000), "loss | acc")
            for step in plot:
                plot.log(loss=train_step())      # every step
                if step % 50 == 0:
                    plot.log(acc=evaluate())     # every 50 steps, same x-axis
            ```
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
                self._specs.append(self._with_defaults(_normalise_panel({"metrics": [m]})))
        elif not curves:  # nothing but image panels so far (plot.imshow() before the first log)
            self._specs.append(self._with_defaults(_normalise_panel({"metrics": unplaced})))
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

    def axhline(self, y, label=None, *, metric=None, **kwargs):
        """Draw a horizontal reference line.

        With `metric`, the line goes on the axis that holds that metric, the same as
        `plot[metric].axhline(y, label)`. Without it, the line goes on the left axis of every
        panel that exists now. Panels created later do not get it. The line is dashed grey by
        default and has a legend entry.

        Args:
            y (float): Where to draw the line.
            label (str | None): The legend text. Defaults to the value of `y`.
            metric (str | None): Draw the line only on the axis holding this metric.
            **kwargs (Any): Passed to matplotlib's `Axes.axhline` (`color`, `linestyle`, `linewidth`,
                `alpha`, ...).

        Example:
            ```python
            plot.axhline(math.log(d_vocab), "uniform", metric="loss")
            ```
        """
        if metric is not None:
            return self[metric].axhline(y, label, **kwargs)
        line = {**kwargs, "y": float(y), "label": str(label) if label is not None else f"{float(y):g}"}
        for spec in self._specs:
            spec["axhlines"].append(line)
        self._send_layout()

    def axvline(self, x=None, label=None, **kwargs):
        """Draw a vertical line on every curve panel, including panels created later.

        The line is dotted grey by default. For one panel only, use `Panel.axvline`.

        Args:
            x (float | None): Where to draw the line. Defaults to the current step (`plot.step`) at the time of
                the call.
            label (str | None): Text written along the line. It does not go in the legend.
            **kwargs (Any): Passed to matplotlib's `Axes.axvline` (`color`, `linestyle`, ...).

        Example:
            ```python
            if step == warmup_steps:
                plot.axvline(label="warmup ends")
            ```
        """
        line = {**kwargs, "x": float(self.step if x is None else x), "label": label}
        self.axvlines.append(line)
        if self.mode == "process":
            self._inbox.put(("axvline", line))
        elif self.mode == "thread":
            self._renderer.add_axvline(line)

    def imshow(self, x, **kwargs):
        """Show an image, or a batch of images as a grid, replacing the previous one.

        The picture goes on the plot's first image panel. If the plot has none, one is added.
        On a plot made by `subplots`, an image panel must already exist, so call `imshow` on a
        panel there first.

        Args:
            x (ArrayLike): A torch tensor, numpy array or other array-like. See `Panel.imshow` for the
                accepted shapes.
            **kwargs (Any): Passed to `Panel.imshow` (`rows`, `cols`, `vmin`, `vmax`, ...).

        Returns:
            Panel: The panel that holds the image.

        Example:
            ```python
            plot = LivePlot(range(steps))
            for step in plot:
                plot.log(loss=train_step())
                if step % 100 == 0:
                    plot.imshow(model(holdout), vmin=0, vmax=1)
            ```
        """
        for i, spec in enumerate(self._specs):
            if spec["kind"] == "image":
                return Panel(self, i).imshow(x, **kwargs)
        assert not self._fixed_grid, \
            "this plot's grid comes from subplots(); call imshow() on one of its panels instead"
        self._specs.append(self._with_defaults(_normalise_panel({"kind": "image"})))
        self._send_layout()
        return Panel(self, len(self._specs) - 1).imshow(x, **kwargs)

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
        """Build a matplotlib `Figure` of the plot as it stands.

        The figure has the same panels, data, reference lines and images as the live plot. It is
        built from `data` on the calling thread and is separate from the live display, so
        changing it does not change the live plot. It works during or after the run, and also
        with `show=False` or outside a notebook.

        Returns:
            matplotlib.figure.Figure: A new figure.

        Example:
            ```python
            fig = plot.figure()
            fig.suptitle("run 12")
            fig.savefig("run12.png", dpi=150)
            ```
        """
        renderer = _FigureRenderer(self._panels_or_placeholder(), self.x_range, self._layout, images=self._image_state())
        for name, (xs, ys) in self.data.items():
            renderer.hist[name] = (list(xs), list(ys))
        for line in self.axvlines:
            renderer.add_axvline(line)
        renderer.render()  # sets the line data and autoscales
        return renderer.fig

    def refresh(self):
        """Redraw now, if drawing happens on the calling thread (`mode == "thread"`).

        In the usual mode a separate process draws on its own schedule and this does nothing.
        """
        if self.mode == "thread":
            self._show(self._renderer.render())

    def finish(self):
        """Draw the final frame, close the progress bar and stop the render process.

        Iterating the plot (`for batch in plot`) and the `with` block call this for you. Call it
        yourself after nested loops that use `plot(loader)` or `plot.update()`. Calling it twice
        is safe. `data` stays available afterwards. If `record` was a path, the GIF is written
        here.
        """
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
        """Write the recorded frames as an animated GIF.

        The plot must have been created with `record=True` or `record="path.gif"`. The GIF
        replays at the pace of the run.

        Args:
            path (str): Where to write the GIF.
            speedup: Replay this many times faster than the run.
            max_frame_ms: Longest time any one frame is shown, in milliseconds.
            hold_last_ms: How long the last frame is shown before the GIF loops.
            colors: Size of the GIF's colour palette, at most 256.

        Returns:
            str: The path that was written.

        Raises:
            ValueError: If no frames were recorded.
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
        """Use the plot as a context manager: `finish()` is called when the block exits.

        Example:
            ```python
            with LivePlot("loss") as plot:
                for step in range(num_steps):
                    plot.log(step, loss=train_step())
            ```
        """
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
            for line in self.axvlines:
                renderer.add_axvline(line)
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
