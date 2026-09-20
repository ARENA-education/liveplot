"""Failure modes that must not reach the training loop, and tqdm arguments that must reach the bar."""

import warnings

import pytest

from liveplot import LivePlot
from liveplot.liveplot import _normalise_panel


def test_bad_smooth_weight_is_caught_where_it_is_given():
    """A window (20) or wandb's excluded 1.0 used to sail past the constructor and kill the render child."""
    for bad in (20, 1.0, -0.1):
        with pytest.raises(AssertionError, match="weight in"):
            LivePlot("loss", smooth=bad, progress=False)
        with pytest.raises(AssertionError, match="weight in"):
            _normalise_panel({"metrics": ["loss"], "smooth": bad})
    p = LivePlot("loss", {"metrics": ["acc"], "smooth": 0}, smooth=0.9, progress=False)  # the legal range still passes
    assert [pn["smooth"] for pn in p._specs] == [0.9, 0]


def test_a_renderer_that_cannot_be_built_does_not_break_the_loop(monkeypatch):
    """The fallback renderer usually fails for the same reason the child did; log() must survive it."""
    p = LivePlot("loss", progress=False)
    assert p.mode == "off"
    p.mode = "thread"  # pretend we were drawing here
    monkeypatch.setattr("liveplot.liveplot._FigureRenderer", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no backend")))
    with pytest.warns(UserWarning, match="collecting into plot.data only"):
        p._fall_back_to_thread("the render process died")
    assert p.mode == "off"
    for step in range(5):
        p.log(step, loss=1.0)  # must not raise
    assert p.data["loss"][0] == [0, 1, 2, 3, 4], "data is still collected"


def test_set_title_survives_a_newly_discovered_metric():
    p = LivePlot(progress=False)
    p.set_title("my run")
    p.log(0, loss=1.0)
    p.log(1, acc=0.5)  # joins the shared panel; used to reset the title to "loss / acc"
    assert p.panels[0].spec["title"] == "my run"
    p.panels[0].set_title("chosen later")
    p.log(2, lr=1e-3)
    assert p.panels[0].spec["title"] == "chosen later"
    q = LivePlot(progress=False)  # a title nobody chose still tracks the metrics
    q.log(0, loss=1.0)
    assert q.panels[0].spec["title"] == "loss"
    q.log(1, acc=0.5)
    assert q.panels[0].spec["title"] == "loss / acc"


def test_tqdm_kwargs_reach_the_bar_without_colliding():
    pytest.importorskip("tqdm")
    p = LivePlot(progress=True)
    for _ in p(range(3), unit="batch", desc="epoch 0"):  # used to raise TypeError: multiple values for 'unit'
        pass
    assert p._bar.unit == "batch" and p._bar.desc == "epoch 0" and p._bar.total == 3


def test_initial_and_total_reach_the_bar():
    pytest.importorskip("tqdm")
    p = LivePlot(range(5), initial=100, progress=True)
    for _ in p:
        pass
    assert p.step == 105 and p.x_range == (100, 105)
    assert p._bar.n == 105 and p._bar.total == 105, "the bar counts in the x-axis's numbers"

    q = LivePlot(range(3), initial=256, unit="examples", unit_scale=128, progress=True)
    for _ in q:
        pass
    assert q.x_range == (256, 640)
    assert q._bar.n * q._bar.unit_scale == 640 and q._bar.total * q._bar.unit_scale == 640

    def gen():
        yield from range(4)

    r = LivePlot(gen(), total=4, progress=True)  # no len(): the bar takes the plot's total
    for _ in r:
        pass
    assert r._bar.total == 4

    s = LivePlot("loss", total=12, progress=True)  # ... but a wrapped inner loop must not borrow it
    for _ in s(gen()):
        pass
    assert s._bar.total is None, "the plot-wide total spans every epoch, not this one"


def test_default_bars_are_unchanged():
    """initial=0 is the default: none of the above may alter the ordinary bar."""
    pytest.importorskip("tqdm")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        p = LivePlot(range(7), progress=True)
        for _ in p:
            pass
    assert p._bar.n == 7 and p._bar.total == 7 and p._bar.unit == "step"
