import math
import time

import pytest

from liveplot import LivePlot
from liveplot.liveplot import _FigureRenderer, _normalise_hlines, _normalise_panel

LAYOUT = (3, None, None, (4, 3), 50, "step")
PNG = b"\x89PNG\r\n\x1a\n"


def test_normalise_hlines():
    assert _normalise_hlines(None) == []
    assert _normalise_hlines({"uniform": 10.8, "entropy": 7}) == [("uniform", 10.8), ("entropy", 7.0)]
    assert _normalise_hlines([0.5, 1]) == [("0.5", 0.5), ("1", 1.0)]
    p = _normalise_panel({"metrics": ["loss"], "secondary": ["acc"], "hlines": {"target": 0.1}, "hlines2": [0.9]})
    assert p["hlines"] == [("target", 0.1)] and p["hlines2"] == [("0.9", 0.9)]
    with pytest.raises(AssertionError):
        _normalise_panel({"metrics": ["loss"], "hline": 1})  # misspelt key is caught


def test_renderer_draws_reference_lines_and_marks():
    panels = [_normalise_panel({"metrics": ["loss"], "secondary": ["acc"], "hlines": {"uniform": 2.3}, "hlines2": {"solved": 0.95}})]
    r = _FigureRenderer(panels, xlim=None, layout=LAYOUT)
    ax, ax2 = r.lines["loss"].axes, r.lines["acc"].axes
    legend_labels = [t.get_text() for t in ax.get_legend().get_texts()]
    assert legend_labels == ["loss", "acc", "uniform", "solved"], "reference lines get legend entries"
    assert len(ax.get_lines()) == 2 and len(ax2.get_lines()) == 2  # curve + hline on each axis
    r.add_mark(5, "lr drop")
    assert len(ax.get_lines()) == 3 and any(t.get_text().strip() == "lr drop" for t in ax.texts)
    r.set_layout(panels + [_normalise_panel("lr")])  # re-layout keeps marks and draws them on the new panel too
    assert all(any(line.get_linestyle() == ":" for line in a.get_lines()) for a in r.axes)
    for step in range(10):
        r.add(step, {"loss": 3.0 - step / 5, "acc": step / 10})
    assert r.render()[:8] == PNG
    assert ax.get_ylim()[0] <= 2.3 <= ax.get_ylim()[1], "autoscale keeps the reference line in view"


def test_hline_and_mark_off_mode():
    p = LivePlot("loss | acc")
    p.log(0, loss=1.0, acc=0.5)
    p.hline(0.2, "target", metric="loss")
    p.hline(0.9, metric="acc")  # label defaults to the value
    p.hline(0.0)  # no metric: left axis of every panel
    assert p.panels[0]["hlines"] == [("target", 0.2), ("0", 0.0)] and p.panels[0]["hlines2"] == [("0.9", 0.9)]
    p.hline(1e-3, "final lr", metric="lr")  # metric not logged yet: creates its panel
    assert p.panels[1]["metrics"] == ["lr"] and p.panels[1]["hlines"] == [("final lr", 0.001)]
    p.log(7, loss=0.5)
    p.mark("lr drop")
    p.mark("later", x=9)
    assert p.marks == [(7, "lr drop"), (9, "later")]


class _FakeHandle:
    def __init__(self):
        self.frames = []

    def update(self, img):
        self.frames.append(img.data)


def test_process_mode_reference_lines(monkeypatch, tmp_path):
    h = _FakeHandle()
    monkeypatch.setattr(LivePlot, "_make_display_handle", staticmethod(lambda: h))
    with LivePlot({"metrics": ["loss"], "hlines": {"uniform": math.log(50257)}}, refresh_seconds=0.1) as p:
        t0 = time.monotonic()
        while not h.frames and time.monotonic() - t0 < 90:  # child startup
            p.log(0, loss=11.0)
            time.sleep(0.05)
        for step in range(40):
            p.log(step, loss=11 - step / 8)
            if step == 20:
                p.mark("halfway")
                p.hline(7.35, "unigram", metric="loss")
            time.sleep(0.03)
    assert h.frames and all(f[:8] == PNG for f in h.frames) and not p._proc.is_alive()
    (tmp_path / "frame.png").write_bytes(h.frames[-1])
    print("final frame:", tmp_path / "frame.png")
