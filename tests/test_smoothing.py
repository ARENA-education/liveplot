import math
import random

import pytest

from liveplot import LivePlot
from liveplot.liveplot import _FigureRenderer, _normalise_panel, _rolling_mean

LAYOUT = (3, None, None, (4, 3), 50, "step")
PNG = b"\x89PNG\r\n\x1a\n"


def test_rolling_mean():
    assert list(_rolling_mean([1, 2, 3, 4, 5], 2)) == [1.0, 1.5, 2.5, 3.5, 4.5]
    assert list(_rolling_mean([1, 2, 3, 4, 5], 100)) == [1.0, 1.5, 2.0, 2.5, 3.0]  # expanding window at the start
    assert list(_rolling_mean([7.0], 5)) == [7.0]


def test_panel_keys():
    p = _normalise_panel({"metrics": ["loss"], "smooth": 20, "yscale": "log"})
    assert p["smooth"] == 20 and p["yscale"] == "log" and p["yscale2"] == "linear"
    assert _normalise_panel("loss")["smooth"] is None
    with pytest.raises(AssertionError):
        _normalise_panel({"metrics": ["loss"], "smoothing": 20})


def test_renderer_smooths_and_fades_raw():
    panels = [_normalise_panel({"metrics": ["loss"], "smooth": 10}), _normalise_panel({"metrics": ["acc"], "yscale": "log"})]
    r = _FigureRenderer(panels, xlim=None, layout=LAYOUT)
    rng = random.Random(0)
    for step in range(200):
        r.add(step, {"loss": math.exp(-step / 50) + 0.3 * (rng.random() - 0.5), "acc": 10 ** (-step / 100)})
    assert r.render()[:8] == PNG
    ax = r.lines["loss"].axes
    assert len(ax.get_lines()) == 2, "raw (faded) + smoothed line"
    assert r.raw_lines["loss"].get_alpha() == 0.25 and r.lines["loss"].get_alpha() is None
    ys = r.lines["loss"].get_ydata()
    raw = r.raw_lines["loss"].get_ydata()
    assert len(ys) == 200 and max(abs(a - b) for a, b in zip(ys[1:], ys[:-1])) < max(abs(a - b) for a, b in zip(raw[1:], raw[:-1]))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["loss"], "the raw line has no legend entry"
    assert r.lines["acc"].axes.get_yscale() == "log"
    assert "acc" not in r.raw_lines and len(r.lines["acc"].axes.get_lines()) == 1


def test_plot_wide_default_and_override():
    p = LivePlot("loss", {"metrics": ["acc"], "smooth": 0}, smooth=25)
    p.log(0, loss=1.0, acc=0.5, lr=1e-3)  # lr is discovered -> gets the default too
    assert [pn["smooth"] for pn in p.panels] == [25, 0, 25]
    q = LivePlot()
    q.log(0, loss=1.0)
    assert q.panels[0]["smooth"] is None
