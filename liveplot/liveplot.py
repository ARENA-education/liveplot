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

(allowed keys: title, metrics, secondary, xlabel, ylabel, ylabel2, xlim, ylim, ylim2,
axhlines, axhlines2, smooth, yscale, yscale2; the names follow matplotlib's
`Axes.set(...)` keywords, with a `2` suffix for the right-hand axis). Reference
lines follow matplotlib too: `axhlines={"uniform": 10.8}` in a panel (or a list of
`axhline` kwargs) draws a dashed line at that level with a legend entry,
`plot.axhline(y, label, metric=...)` does the same at run time, and
`plot.axvline(label="lr drop")` draws a dotted vertical line on every panel at the
current x. Extra kwargs go to the artists. Smoothing: `smooth=0.9` on a panel (or
on LivePlot, as the default for every panel) draws each curve as wandb's
time-weighted EMA with that weight, with the raw values faded behind it. Log axes:
`yscale="log"` (`yscale2` for the right axis).

How it works: the training thread only appends numbers (~40 us per `log`). A
separate *render process* owns the matplotlib figure, redraws it at most once per
`refresh_seconds` (default 1.0; points arriving in between are batched into the
next frame; 0 means redraw on every arrival, as fast as rendering allows), and
sends back PNG bytes that get swapped into a fixed output cell. The output is a plain image, so it behaves the same in Jupyter, Colab, VS
Code and Cursor: no widgets, no CDN, no JavaScript. The progress bar is tqdm
(`tqdm.auto`), shown below the plot with the latest logged values as its postfix;
pass an existing tqdm object as the iterable to reuse yours instead.

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
import time
import warnings
import weakref

_PANEL_KEYS = {"title", "metrics", "secondary", "xlabel", "ylabel", "ylabel2", "xlim", "ylim", "ylim2", "axhlines", "axhlines2",
               "smooth", "yscale", "yscale2"}
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


def _normalise_panel(spec) -> dict:
    if isinstance(spec, str):
        spec = _parse_panel_string(spec)
    unknown = set(spec) - _PANEL_KEYS
    assert not unknown, f"unknown panel keys {sorted(unknown)}; allowed: {sorted(_PANEL_KEYS)}"
    metrics, secondary = list(spec.get("metrics", [])), list(spec.get("secondary", []))
    assert metrics or secondary, f"a panel needs at least one metric: {spec!r}"
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
        "smooth": spec.get("smooth"),  # TWEMA weight in [0, 1); None = plot-wide default; 0 = off
        "yscale": spec.get("yscale", "linear"),  # "linear" or "log", left axis
        "yscale2": spec.get("yscale2", "linear"),  # ... right axis
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

    def __init__(self, panels, xlim, layout, hist=None):
        self.xlim, self.layout = xlim, layout  # xlim = default x range or None; layout = (max_cols, rows, cols, cell_size, dpi, xlabel)
        self.hist = hist if hist is not None else {}
        self.axvlines: list[dict] = []  # vertical reference lines (matplotlib axvline kwargs), drawn on every panel
        self.set_layout(panels)

    def set_layout(self, panels):
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        max_cols, rows_opt, cols_opt, cell_size, dpi, xlabel = self.layout
        n = len(panels)
        rows, cols = _grid_shape(n, max_cols, rows_opt, cols_opt)
        self.fig = Figure(figsize=(cell_size[0] * cols, cell_size[1] * rows))
        FigureCanvasAgg(self.fig)
        self.dpi = dpi
        axes = self.fig.subplots(rows, cols, squeeze=False).flatten()
        palette = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
        self.lines = {}
        self.raw_lines = {}  # metric -> faded line of the unsmoothed values, on smoothed panels
        self.smooth = {}  # metric -> TWEMA weight in (0, 1) or None
        self.fixed = {}  # axes -> (x fixed?, y fixed?): fixed axes are never autoscaled
        for ax, panel in zip(axes, panels):
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
            xlim = panel["xlim"] or self.xlim
            if xlim:
                ax.set_xlim(*xlim)
            if panel["ylim"]:
                ax.set_ylim(*panel["ylim"])
            self.fixed[ax] = (xlim is not None, panel["ylim"] is not None)
            if names != ["_"]:  # (the "waiting for data" placeholder has no legend)
                ax.legend(handles, names, loc="best", fontsize=8)  # every panel gets a legend (reference lines included)
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
        for kw in self.axvlines:
            self._draw_axvline(kw)
        self.fig.tight_layout()

    def add_axvline(self, kw: dict):
        self.axvlines.append(kw)
        self._draw_axvline(kw)

    def _draw_axvline(self, kw):
        label = kw.get("label")
        style = {**_REF_LINE_STYLE, "linestyle": ":", **{k: v for k, v in kw.items() if k != "label"}}
        for ax in self.axes:
            ax.axvline(**style)
            if label:
                ax.text(kw["x"], 0.98, f" {label}", transform=ax.get_xaxis_transform(), va="top", ha="left", fontsize=7, color="0.3", rotation=90)

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
    metrics); ("layout", panels) to rebuild the figure; None to finish (draw one
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
                renderer.set_layout(item[1])
                dirty = True
            elif item[0] == "axvline":
                renderer.add_axvline(item[1])
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
        self.smooth = smooth  # default TWEMA weight for panels that don't set their own (wandb's smoothing slider)
        self.panels = [self._with_defaults(_normalise_panel(p)) for p in specs]
        self._explicit_layout = bool(self.panels)
        self._placed = {n for p in self.panels for n in p["metrics"] + p["secondary"]}
        self._layout = (max_cols, rows, cols, cell_size, dpi, unit)
        self._iterable, self._bar, self._desc, self._progress = iterable, None, desc, progress
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
        self.axvlines: list[dict] = []  # vertical reference lines added with axvline()
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
            self._renderer = _FigureRenderer(self._panels_or_placeholder(), self.x_range, self._layout)
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

    def _panels_or_placeholder(self):
        return self.panels or [_normalise_panel({"title": "waiting for data…", "metrics": ["_"]})]

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
        if "total" not in tqdm_kwargs:
            try:
                kwargs["total"] = len(it)
            except TypeError:
                pass
        # No iterable is given to tqdm: we drive it with update() ourselves, so that the closing
        # line of the bar still shows the final postfix (tqdm closes a wrapped iterable before we
        # could write it).
        return tqdm(**kwargs, **tqdm_kwargs)

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
            else:
                self._collect(block=False)
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
        if self._explicit_layout:
            for m in unplaced:
                self.panels.append(self._with_defaults(_normalise_panel({"metrics": [m]})))
        elif not self.panels:
            self.panels.append(self._with_defaults(_normalise_panel({"metrics": unplaced})))
        else:
            self.panels[0]["metrics"].extend(unplaced)
            self.panels[0]["title"] = " / ".join(self.panels[0]["metrics"])
        self._placed.update(unplaced)
        self._send_layout()

    def axhline(self, y, label=None, *, metric=None, **kwargs):
        """
        Like matplotlib's `Axes.axhline`: a horizontal reference line at `y` with `label` in the legend,
        dashed grey by default; any other kwargs (color, linestyle, linewidth, alpha, ...) go to the
        artist. It goes on the axis of `metric`'s panel, or on the left axis of every panel if `metric`
        is None. E.g. the loss a model must beat: `plot.axhline(math.log(d_vocab), "uniform", metric="loss")`.
        """
        line = {**kwargs, "y": float(y), "label": str(label) if label is not None else f"{float(y):g}"}
        if metric is not None and metric not in self._placed:
            self.data.setdefault(metric, ([], []))  # create the metric's panel now so the line has somewhere to go
            self._extend_layout([metric])
        for panel in self.panels:
            if metric is None or metric in panel["metrics"]:
                panel["axhlines"].append(line)
            elif metric in panel["secondary"]:
                panel["axhlines2"].append(line)
        self._send_layout()

    def axvline(self, x=None, label=None, **kwargs):
        """
        Like matplotlib's `Axes.axvline`, on every panel: a vertical line at `x` (default: the current
        step), dotted grey by default, with `label` written along it; other kwargs go to the artist.
        E.g. `plot.axvline(label="lr drop")` where the learning rate changes.
        """
        line = {**kwargs, "x": float(self.step if x is None else x), "label": label}
        self.axvlines.append(line)
        if self.mode == "process":
            self._inbox.put(("axvline", line))
        elif self.mode == "thread":
            self._renderer.add_axvline(line)

    def _send_layout(self):
        if self.mode == "process":
            self._inbox.put(("layout", self.panels))
        elif self.mode == "thread":
            self._renderer.set_layout(self.panels)

    def figure(self):
        """
        A matplotlib Figure of the plot as it stands (same panels, data, reference lines and marks),
        built on the calling thread and independent of the render process: title it, tweak it,
        `fig.savefig("run.png")`, or show it in a report. Safe to call during or after training.
        """
        renderer = _FigureRenderer(self._panels_or_placeholder(), self.x_range, self._layout)
        for name, (xs, ys) in self.data.items():
            renderer.hist[name] = (list(xs), list(ys))
        for line in self.axvlines:
            renderer.add_axvline(line)
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
                    self._inbox.put(None)
                    self._collect(block=True, timeout=5.0)
            elif self.mode == "thread":
                self._show(self._renderer.render())
        finally:
            if self.mode == "process":
                self._proc.join(timeout=2.0)
                if self._proc.is_alive():  # e.g. still importing matplotlib on a very busy machine
                    self._proc.terminate()
                    self._proc.join(timeout=2.0)
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
        warnings.warn(f"LivePlot: {why}; rendering on the training thread from now on.", stacklevel=3)
        self._renderer = _FigureRenderer(self._panels_or_placeholder(), self.x_range, self._layout)
        for name, (xs, ys) in self.data.items():
            self._renderer.hist[name] = (list(xs), list(ys))
        for line in self.axvlines:
            self._renderer.add_axvline(line)
        self.mode = "thread"

    def _collect(self, block: bool, timeout: float = 15.0):
        """Take the newest frame off the outbox (if any) and display it."""
        png, deadline = None, time.monotonic() + timeout
        while True:
            try:
                item = self._outbox.get(timeout=max(0.0, deadline - time.monotonic())) if block else self._outbox.get_nowait()
            except queue.Empty:
                break
            if item is None:  # renderer has sent its final frame
                break
            png = item
            if not block:
                break
            if not self._proc.is_alive() and self._outbox.empty():
                break
        if png is not None:
            self._show(png)

    def _show(self, png: bytes):
        self.last_png = png
        self._last_draw = time.monotonic()
        if self._record:
            self.frames.append((time.monotonic() - self._t0, png))
        if self._handle is not None:
            from IPython.display import Image

            self._handle.update(Image(data=png, format="png"))
