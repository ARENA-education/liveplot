import math
import os
import signal
import subprocess
import sys
import time

import pytest

from liveplot import LivePlot
from liveplot.liveplot import _FigureRenderer, _grid_shape, _normalise_panel, _parse_panel_string

LAYOUT = (3, None, None, (4, 3), 50, "step")  # max_cols, rows, cols, cell_size, dpi, default xlabel
PNG = b"\x89PNG\r\n\x1a\n"


def test_panel_strings():
    assert _parse_panel_string("loss") == {"metrics": ["loss"], "secondary": []}
    assert _parse_panel_string("return | entropy") == {"metrics": ["return"], "secondary": ["entropy"]}
    assert _parse_panel_string("lossD lossG | acc") == {"metrics": ["lossD", "lossG"], "secondary": ["acc"]}
    p = _normalise_panel("lossD lossG | acc")
    assert p["title"] == "lossD / lossG / acc" and p["xlabel"] is None  # None -> plot-wide unit
    with pytest.raises(AssertionError):
        _parse_panel_string("a | b | c")
    with pytest.raises(AssertionError):
        _normalise_panel({"metrics": ["a"], "colour": "red"})
    with pytest.raises(AssertionError):
        _normalise_panel({"metrics": ["a"], "ylim": (0,)})


def test_grid_shape():
    assert _grid_shape(5, max_cols=3, rows=None, cols=None) == (2, 3)
    assert _grid_shape(5, max_cols=None, rows=None, cols=None) == (3, 2)  # live_plotter's near-square rule
    assert _grid_shape(5, max_cols=3, rows=1, cols=None) == (1, 5)
    with pytest.raises(AssertionError):
        _grid_shape(5, max_cols=3, rows=2, cols=2)


def test_renderer_labels_limits_and_relayout():
    panels = [_normalise_panel({"metrics": ["loss"], "secondary": ["acc"], "ylabel": "L", "ylabel2": "A", "ylim2": (0, 1)})]
    r = _FigureRenderer(panels, xlim=None, layout=LAYOUT)
    for step in range(20):
        r.add(step, {"loss": 100.0 * step, "acc": 5.0 * step})
    assert r.render()[:8] == PNG
    ax_loss, ax_acc = r.lines["loss"].axes, r.lines["acc"].axes
    assert ax_loss.get_ylabel() == "L" and ax_acc.get_ylabel() == "A" and ax_loss.get_xlabel() == "step"
    assert ax_acc.get_ylim() == (0, 1), "fixed secondary y-range must survive autoscale"
    assert ax_loss.get_ylim()[1] >= 1900 and ax_loss.get_legend() is not None
    # re-layout keeps history and draws the new metric
    r.set_layout(panels + [_normalise_panel("lr")])
    r.add(20, {"lr": 0.1})
    assert r.render()[:8] == PNG and len(r.hist["loss"][0]) == 20 and "lr" in r.lines


def test_off_mode_discovery_and_log_forms():
    p = LivePlot()  # pytest is not a notebook: no display handle
    assert p.mode == "off" and p._proc is None
    p.log(0, loss=1.0)
    p.log(1, {"loss": 0.5}, acc=0.1)
    p.log(acc=0.2)  # nothing wrapped: x stays at the last explicit step, 1
    assert p.data == {"loss": ([0, 1], [1.0, 0.5]), "acc": ([1, 1], [0.1, 0.2])}
    assert [pn["metrics"] for pn in p._specs] == [["loss", "acc"]], "no layout given: everything on one panel"
    p.finish(); p.finish()  # idempotent

    q = LivePlot("loss", "a | b")
    q.log(0, loss=1, a=2, b=3, extra=4)
    assert [(pn["metrics"], pn["secondary"]) for pn in q._specs] == [(["loss"], []), (["a"], ["b"]), (["extra"], [])]


def test_x_axis_follows_tqdm_counting():
    p = LivePlot(range(5, 10), progress=False)  # values are yielded but never used for x
    seen = [(item, p.step) for item in p]
    assert seen == [(5, 0), (6, 1), (7, 2), (8, 3), (9, 4)] and p._done
    assert p.total == 5 and p.n == 5 and p.x_range == (0, 5)

    q = LivePlot(range(5, 10), initial=5, progress=False)  # tqdm's `initial` shifts the start
    assert [q.step for _ in q] == [5, 6, 7, 8, 9] and q.x_range == (5, 10)

    r = LivePlot(range(3), unit="examples", unit_scale=128, progress=False)  # x in examples seen
    xs = []
    for _ in r:
        r.log(loss=1.0)
        xs.append(r.step)
    assert xs == [0, 128, 256] and r.data["loss"][0] == xs and r.x_range == (0, 384)
    assert r._layout[-1] == "examples", "default x label is the unit"

    s = LivePlot(range(3), progress=False)  # explicit step is per call, like wandb's step=
    for i in s:
        s.log(loss=1.0)
        s.log(1000 + i, acc=0.5)
    assert s.data["loss"][0] == [0, 1, 2] and s.data["acc"][0] == [1000, 1001, 1002]


def test_nested_loops_continue_the_count():
    loader = ["batch"] * 4
    p = LivePlot("loss", "acc", total=3 * len(loader), progress=False)
    for epoch in range(3):
        for _ in p(loader):
            p.log(loss=1.0)
        p.log(acc=float(epoch))
    p.finish()
    assert p.data["loss"][0] == list(range(12)), "x continues across epochs"
    assert p.data["acc"][0] == [4, 8, 12], "logged after the inner loop: x is items consumed so far"
    assert p.x_range == (0, 12)


def test_iterator_form_with_tqdm_postfix():
    tqdm = pytest.importorskip("tqdm")
    p = LivePlot(range(30), desc="train", unit="ex", unit_scale=4)
    for step in p:
        p.log(loss=1.0 / (step + 1), acc=step / 30)
        time.sleep(0.005)
    assert p._bar is not None and p._bar.n == 30 and p._bar.unit == "ex" and p._bar.unit_scale == 4
    assert p._bar.postfix and "loss" in p._bar.postfix and "acc" in p._bar.postfix
    bar = tqdm.tqdm(range(3), disable=True)
    q = LivePlot(bar)  # caller's own tqdm bar is reused, not wrapped
    assert list(q) == [0, 1, 2] and q._bar is bar
    r = LivePlot(progress=False)
    bars = []
    for epoch in range(2):
        for _ in r(range(3), desc=f"epoch {epoch}"):
            pass
        bars.append(r._bar)
    assert bars == [None, None], "progress=False: no bars even when wrapping"


class _FakeHandle:
    def __init__(self):
        self.frames = []

    def update(self, img):
        self.frames.append(img.data)


@pytest.fixture
def fake_notebook(monkeypatch):
    h = _FakeHandle()
    monkeypatch.setattr(LivePlot, "_make_display_handle", staticmethod(lambda: h))
    return h


def wait_for_first_frame(p, h, timeout=90):
    """The render child imports matplotlib at startup (~1 s idle, much more on a loaded CI box)."""
    t0 = time.monotonic()
    while not h.frames and time.monotonic() - t0 < timeout:
        p.log(0, loss=1.0)
        time.sleep(0.05)
    assert h.frames, "render child produced no frame"


def test_process_mode_end_to_end(fake_notebook):
    with LivePlot("loss", {"metrics": ["acc"], "ylim": (0, 1)}, total=10_000, refresh_seconds=0.2) as p:
        assert p.mode == "process"
        wait_for_first_frame(p, fake_notebook)
        n_warmup, n_frames0 = len(p.data["loss"][0]), len(fake_notebook.frames)
        costs, step, t_end = [], 0, time.monotonic() + 120  # log until three more frames have been shown
        while len(fake_notebook.frames) < n_frames0 + 3 and time.monotonic() < t_end:  # (a frame can take
            t0 = time.perf_counter()                                                      # seconds on a loaded box)
            p.log(step, loss=math.exp(-step / 100), acc=min(step / 300, 1.0), lr=1e-3)  # lr is discovered
            costs.append(time.perf_counter() - t0)
            step += 1
            time.sleep(0.002)
    costs.sort()
    assert costs[len(costs) // 2] < 0.001, "median log() must stay well under a millisecond"
    assert len(fake_notebook.frames) >= n_frames0 + 3 and all(f[:8] == PNG for f in fake_notebook.frames)
    assert not p._proc.is_alive()
    assert len(p.data["loss"][0]) == n_warmup + step and p.last_png == fake_notebook.frames[-1]
    assert [pn["metrics"] for pn in p._specs] == [["loss"], ["acc"], ["lr"]]


def test_interrupt_inside_with_block_is_clean(fake_notebook):
    with pytest.raises(KeyboardInterrupt):
        with LivePlot(refresh_seconds=0.2) as p:
            wait_for_first_frame(p, fake_notebook)
            n_warmup = len(p.data["loss"][0])
            for step in range(50):
                p.log(step, loss=1.0 / (step + 1))
                if step == 30:
                    raise KeyboardInterrupt
    assert not p._proc.is_alive()
    assert len(p.data["loss"][0]) == n_warmup + 31 and fake_notebook.frames, "final frame drawn, data kept"


@pytest.mark.skipif(sys.platform == "win32", reason="os.kill(pid, SIGINT) terminates the process on Windows")
def test_render_child_ignores_sigint(fake_notebook):
    """Jupyter's interrupt goes to the whole process group; the renderer must shrug it off."""
    p = LivePlot(refresh_seconds=0.1)
    wait_for_first_frame(p, fake_notebook)
    os.kill(p._proc.pid, signal.SIGINT)
    time.sleep(0.5)
    assert p._proc.is_alive(), "render process must survive SIGINT"
    for step in range(1, 20):
        p.log(step, loss=1.0 / step)
        time.sleep(0.02)
    p.finish()
    assert fake_notebook.frames and not p._proc.is_alive()


def test_dropped_plot_does_not_leak_process(fake_notebook):
    p = LivePlot()
    wait_for_first_frame(p, fake_notebook)
    proc = p._proc
    assert proc.is_alive()
    del p
    import gc; gc.collect()
    proc.join(timeout=3)
    assert not proc.is_alive(), "finalizer must shut the render process down"


def test_render_child_exits_when_parent_dies():
    """Start a LivePlot in a throwaway interpreter that dies without finish(); the child must follow."""
    code = f"""
import os, sys, time; sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
from liveplot import LivePlot
class H:
    def update(self, img): pass
LivePlot._make_display_handle = staticmethod(lambda: H())
p = LivePlot(refresh_seconds=0.1); p.log(0, loss=1.0)
t0 = time.time()
while p.last_png is None and time.time() - t0 < 90: p.log(0, loss=1.0); time.sleep(0.05)
print(p._proc.pid, flush=True)
os._exit(0)   # no finish(), no finalizers, no atexit: the parent just vanishes
"""
    child_pid = int(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30).stdout.strip())
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        try:
            os.kill(child_pid, 0)
        except ProcessLookupError:
            return  # gone: good
        time.sleep(0.1)
    pytest.fail("render process outlived its parent")


def test_record_and_save_gif(tmp_path):
    """Recording works with no notebook at all: frames are rendered anyway and stitched into an animated GIF."""
    gif = tmp_path / "run.gif"
    p = LivePlot(range(60), record=str(gif), refresh_seconds=0.1, progress=False, cell_size=(3, 2), dpi=40)
    assert p.mode == "process", "record=True starts the renderer even without a display handle"
    t0 = time.monotonic()
    while p.last_png is None and time.monotonic() - t0 < 90:  # child startup
        p.log(0, loss=1.0)
        time.sleep(0.05)
    for step in p:
        p.log(loss=1.0 / (step + 1))
        time.sleep(0.03)
    assert len(p.frames) >= 3 and all(png[:8] == PNG for _, png in p.frames)
    assert gif.exists()
    from PIL import Image
    im = Image.open(gif)
    # Pillow merges identical consecutive frames (the warm-up ones are), accumulating their durations
    assert im.is_animated and 3 <= im.n_frames <= len(p.frames)
    # colour fidelity: every GIF frame must be a faithful copy of one of the recorded PNGs (a per-frame
    # palette bug used to scramble the colours of frames whose palette differed from the first one's)
    import io
    from PIL import ImageChops, ImageStat
    sources = [Image.open(io.BytesIO(png)).convert("RGB") for _, png in p.frames]
    for k in range(im.n_frames):
        im.seek(k)
        err = min(max(ImageStat.Stat(ImageChops.difference(im.convert("RGB"), src)).mean) for src in sources)
        assert err < 8, f"GIF frame {k} differs from every recorded frame (mean channel error {err:.1f})"
    q = LivePlot(progress=False)
    with pytest.raises(ValueError):
        q.save_gif(tmp_path / "empty.gif")


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="reads /proc")
def test_refresh_zero_does_not_spin_when_idle(fake_notebook):
    """refresh_seconds=0 means 'redraw on every arrival', not 'poll the queue in a loop'."""
    def cpu_seconds(pid):
        with open(f"/proc/{pid}/stat") as f:
            fields = f.read().split(")")[-1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")

    p = LivePlot(refresh_seconds=0)
    wait_for_first_frame(p, fake_notebook)
    time.sleep(0.3)
    before = cpu_seconds(p._proc.pid)
    time.sleep(2.0)  # idle: nothing logged
    idle = cpu_seconds(p._proc.pid) - before
    n_before = len(fake_notebook.frames)
    for step in range(1, 60):  # busy: frames should now come as fast as rendering allows
        p.log(step, loss=1.0 / step)
        time.sleep(0.01)
    p.finish()
    assert idle < 0.2, f"render child used {idle:.2f}s CPU while idle"
    assert len(fake_notebook.frames) - n_before >= 2
