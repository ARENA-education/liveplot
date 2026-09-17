import math
import time

import pytest

from liveplot import LivePlot
from liveplot.liveplot import _FigureRenderer, _normalise_axhlines, _normalise_panel

LAYOUT = (3, None, None, (4, 3), 50, "step")
PNG = b"\x89PNG\r\n\x1a\n"


def test_normalise_axhlines():
    assert _normalise_axhlines(None) == []
    assert _normalise_axhlines({"uniform": 10.8, "entropy": 7}) == [{"y": 10.8, "label": "uniform"}, {"y": 7.0, "label": "entropy"}]
    assert _normalise_axhlines([0.5, 1]) == [{"y": 0.5, "label": "0.5"}, {"y": 1.0, "label": "1"}]
    assert _normalise_axhlines([dict(y=2, label="two", color="red", linestyle="-")]) == [{"y": 2.0, "label": "two", "color": "red", "linestyle": "-"}]
    p = _normalise_panel({"metrics": ["loss"], "secondary": ["acc"], "axhlines": {"target": 0.1}, "axhlines2": [0.9]})
    assert p["axhlines"] == [{"y": 0.1, "label": "target"}] and p["axhlines2"] == [{"y": 0.9, "label": "0.9"}]
    with pytest.raises(AssertionError):
        _normalise_panel({"metrics": ["loss"], "hlines": {"a": 1}})  # the old / misspelt key is caught
    with pytest.raises(AssertionError):
        _normalise_axhlines([dict(label="no y")])


def test_renderer_draws_reference_lines_with_matplotlib_kwargs():
    panels = [_normalise_panel({"metrics": ["loss"], "secondary": ["acc"],
                                "axhlines": [dict(y=2.3, label="uniform", color="red", linestyle="-")],
                                "axhlines2": {"solved": 0.95}})]
    r = _FigureRenderer(panels, xlim=None, layout=LAYOUT)
    ax, ax2 = r.lines["loss"].axes, r.lines["acc"].axes
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["loss", "acc", "uniform", "solved"]
    ref = ax.get_lines()[1]
    assert ref.get_color() == "red" and ref.get_linestyle() == "-", "kwargs reach the matplotlib artist"
    assert ax2.get_lines()[1].get_linestyle() == "--", "default style when no kwargs given"
    r.add_axvline({"x": 5, "label": "lr drop", "color": "blue"})
    assert len(ax.get_lines()) == 3 and ax.get_lines()[2].get_color() == "blue" and ax.get_lines()[2].get_linestyle() == ":"
    assert any(t.get_text().strip() == "lr drop" for t in ax.texts)
    r.set_layout(panels + [_normalise_panel("lr")])  # re-layout keeps the vertical line, on the new panel too
    assert all(any(line.get_linestyle() == ":" for line in a.get_lines()) for a in r.axes)
    for step in range(10):
        r.add(step, {"loss": 3.0 - step / 5, "acc": step / 10})
    assert r.render()[:8] == PNG
    assert ax.get_ylim()[0] <= 2.3 <= ax.get_ylim()[1], "autoscale keeps the reference line in view"


def test_axhline_and_axvline_off_mode():
    p = LivePlot("loss | acc")
    p.log(0, loss=1.0, acc=0.5)
    p.axhline(0.2, "target", metric="loss", color="green")
    p.axhline(0.9, metric="acc")  # label defaults to the value
    p.axhline(0.0)  # no metric: left axis of every panel
    assert p._specs[0]["axhlines"] == [{"color": "green", "y": 0.2, "label": "target"}, {"y": 0.0, "label": "0"}]
    assert p._specs[0]["axhlines2"] == [{"y": 0.9, "label": "0.9"}]
    p.axhline(1e-3, "final lr", metric="lr")  # metric not logged yet: creates its panel
    assert p._specs[1]["metrics"] == ["lr"] and p._specs[1]["axhlines"] == [{"y": 0.001, "label": "final lr"}]
    p.log(7, loss=0.5)
    p.axvline(label="lr drop")  # x defaults to the current step
    p.axvline(9, "later", linewidth=2)
    assert p.axvlines == [{"x": 7.0, "label": "lr drop"}, {"linewidth": 2, "x": 9.0, "label": "later"}]


class _FakeHandle:
    def __init__(self):
        self.frames = []

    def update(self, img):
        self.frames.append(img.data)


def test_process_mode_reference_lines(monkeypatch):
    h = _FakeHandle()
    monkeypatch.setattr(LivePlot, "_make_display_handle", staticmethod(lambda: h))
    with LivePlot({"metrics": ["loss"], "axhlines": {"uniform": math.log(50257)}}, refresh_seconds=0.1) as p:
        t0 = time.monotonic()
        while not h.frames and time.monotonic() - t0 < 90:  # child startup
            p.log(0, loss=11.0)
            time.sleep(0.05)
        for step in range(40):
            p.log(step, loss=11 - step / 8)
            if step == 20:
                p.axvline(label="halfway")
                p.axhline(7.35, "unigram", metric="loss")
            time.sleep(0.03)
    assert h.frames and all(f[:8] == PNG for f in h.frames) and not p._proc.is_alive()
