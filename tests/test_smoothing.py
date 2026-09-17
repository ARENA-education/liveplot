import math
import random

import numpy as np
import pytest

from liveplot import LivePlot
from liveplot.liveplot import _FigureRenderer, _normalise_panel, _twema

LAYOUT = (3, None, None, (4, 3), 50, "step")
PNG = b"\x89PNG\r\n\x1a\n"


def wandb_twema(xs, ys, param, range_of_x, viewport_scale=1000):
    """Line-for-line port of the JavaScript on docs.wandb.ai (line-plot/smoothing), used as the oracle."""
    weight = min(math.sqrt(param or 0), 0.999)
    last_y, debias, out = 0.0, 0.0, []
    for i, y in enumerate(ys):
        prev = i - 1 if i > 0 else 0
        change = (xs[i] - xs[prev]) / range_of_x * viewport_scale
        adj = weight ** change
        last_y = last_y * adj + y
        debias = debias * adj + 1
        out.append(last_y / debias)
    return out


def test_twema_matches_wandb_reference():
    rng = random.Random(0)
    xs = sorted(rng.sample(range(0, 5000), 300))  # irregular spacing: the "time-weighted" part matters
    ys = [math.exp(-x / 1500) + 0.3 * (rng.random() - 0.5) for x in xs]
    for param in (0.0, 0.5, 0.9, 0.99, 1.0):
        ours = _twema(xs, ys, param, x_range=5000)
        assert np.allclose(ours, wandb_twema(xs, ys, param, 5000)), f"param {param}"
    assert np.allclose(_twema(xs, ys, 0.0, 5000), ys), "weight 0: no smoothing"
    assert _twema([0], [3.0], 0.9, 1)[0] == 3.0, "first point is not biased towards zero"


def test_twema_is_smoother_and_time_weighted():
    rng = random.Random(1)
    xs = list(range(1000))
    ys = [0.5 + 0.4 * (rng.random() - 0.5) for _ in xs]
    sm = _twema(xs, ys, 0.9, 1000)
    assert np.std(np.diff(sm)) < 0.2 * np.std(np.diff(ys))
    # the same points spread over a 10x larger x-range decay 10x faster per unit x => less smoothing
    wide = _twema([x * 10 for x in xs], ys, 0.9, 1000)
    assert np.std(np.diff(wide)) > np.std(np.diff(sm))


def test_panel_keys():
    p = _normalise_panel({"metrics": ["loss"], "smooth": 0.9, "yscale": "log"})
    assert p["smooth"] == 0.9 and p["yscale"] == "log" and p["yscale2"] == "linear"
    assert _normalise_panel("loss")["smooth"] is None
    with pytest.raises(AssertionError):
        _normalise_panel({"metrics": ["loss"], "smoothing": 0.9})
    with pytest.raises(AssertionError):
        _FigureRenderer([_normalise_panel({"metrics": ["loss"], "smooth": 20})], xlim=None, layout=LAYOUT)  # a window, not a weight


def test_renderer_smooths_and_fades_raw():
    panels = [_normalise_panel({"metrics": ["loss"], "smooth": 0.9}), _normalise_panel({"metrics": ["acc"], "yscale": "log"})]
    r = _FigureRenderer(panels, xlim=(0, 200), layout=LAYOUT)
    rng = random.Random(0)
    for step in range(200):
        r.add(step, {"loss": math.exp(-step / 50) + 0.3 * (rng.random() - 0.5), "acc": 10 ** (-step / 100)})
    assert r.render()[:8] == PNG
    ax = r.lines["loss"].axes
    assert len(ax.get_lines()) == 2, "raw (faded) + smoothed line"
    assert r.raw_lines["loss"].get_alpha() == 0.25 and r.lines["loss"].get_alpha() is None
    ys, raw = r.lines["loss"].get_ydata(), r.raw_lines["loss"].get_ydata()
    assert len(ys) == 200 and np.std(np.diff(ys)) < np.std(np.diff(raw))
    assert [t.get_text() for t in ax.get_legend().get_texts()] == ["loss"], "the raw line has no legend entry"
    assert r.lines["acc"].axes.get_yscale() == "log"
    assert "acc" not in r.raw_lines and len(r.lines["acc"].axes.get_lines()) == 1


def test_plot_wide_default_and_override():
    p = LivePlot("loss", {"metrics": ["acc"], "smooth": 0}, smooth=0.9)
    p.log(0, loss=1.0, acc=0.5, lr=1e-3)  # lr is discovered -> gets the default too
    assert [pn["smooth"] for pn in p.panels] == [0.9, 0, 0.9]
    q = LivePlot()
    q.log(0, loss=1.0)
    assert q.panels[0]["smooth"] is None
