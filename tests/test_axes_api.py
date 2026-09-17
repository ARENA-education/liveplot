import pytest

from liveplot import LivePlot
from liveplot.liveplot import Panel, _Axis


def test_addressing_by_metric_and_by_panel():
    p = LivePlot("loss | acc", "lr")
    p.log(0, loss=1.0, acc=0.5, lr=1e-3)
    assert [type(x) for x in p.panels] == [Panel, Panel] and p.panels[0].metrics == ["loss", "acc"]
    left, right, lr = p["loss"], p["acc"], p["lr"]
    assert isinstance(left, _Axis) and not left._right and right._right and left.panel.spec is p.panels[0].spec
    assert right.metrics == ["acc"] and lr.panel.spec is p.panels[1].spec
    assert repr(p.panels[0]) == "Panel(0: loss | acc)"
    with pytest.raises(AssertionError):
        p.panels[1].right  # no metrics after '|'


def test_matplotlib_named_setters_go_to_the_right_axis():
    p = LivePlot("loss | acc")
    p.log(0, loss=1.0, acc=0.5)
    p["loss"].set_ylabel("cross-entropy")
    p["acc"].set_ylabel("test accuracy")
    p["acc"].set_ylim(0, 1)  # matplotlib's two-argument form
    p["loss"].set_ylim((0.0, 2.0))  # ... and the tuple form
    p["acc"].set_yscale("log")
    p.panels[0].set_title("training")
    p.panels[0].set_xlabel("examples")
    p.panels[0].set_xlim(0, 500)
    spec = p.panels[0].spec
    assert spec["ylabel"] == "cross-entropy" and spec["ylabel2"] == "test accuracy"
    assert spec["ylim"] == (0.0, 2.0) and spec["ylim2"] == (0.0, 1.0) and spec["yscale2"] == "log" and spec["yscale"] == "linear"
    assert spec["title"] == "training" and spec["xlabel"] == "examples" and spec["xlim"] == (0.0, 500.0)
    p["acc"].set_title("via the axis, like a matplotlib Axes")  # panel-level setter reachable from an axis
    assert spec["title"] == "via the axis, like a matplotlib Axes"
    p.panels[0].set_ylabel("left again")  # y-setters on a panel act on its left axis
    assert spec["ylabel"] == "left again"
    with pytest.raises(AssertionError):
        p["acc"].set_yscale("sqrt")
    with pytest.raises(AssertionError):
        p["acc"].set_ylim(0)


def test_set_kwargs_and_reference_lines():
    p = LivePlot("loss | acc")
    p.log(0, loss=1.0, acc=0.5)
    p["acc"].set(ylabel="acc", ylim=(0, 1), yscale="log")
    p.panels[0].set(title="t", xlabel="x", smooth=0.5)
    spec = p.panels[0].spec
    assert (spec["ylabel2"], spec["ylim2"], spec["yscale2"], spec["title"], spec["xlabel"], spec["smooth"]) == ("acc", (0.0, 1.0), "log", "t", "x", 0.5)
    p["loss"].axhline(0.1, "target", color="red")
    p["acc"].axhline(0.9)
    p.panels[0].axvline(3, "here", linewidth=2)
    p.log(7, loss=0.5)
    p.panels[0].axvline(label="now")  # x defaults to the current step
    assert spec["axhlines"] == [{"color": "red", "y": 0.1, "label": "target"}] and spec["axhlines2"] == [{"y": 0.9, "label": "0.9"}]
    assert spec["axvlines"] == [{"linewidth": 2, "x": 3.0, "label": "here"}, {"x": 7.0, "label": "now"}]
    assert p.axvlines == [], "panel-level axvline is not a plot-wide one"
    p.axhline(0.0, "zero", metric="loss")  # the plot-level form with metric= delegates to the axis
    assert spec["axhlines"][-1] == {"y": 0.0, "label": "zero"}


def test_plot_level_setters_apply_to_every_panel_and_later_ones():
    p = LivePlot("loss", "acc")
    p.set(xlabel="examples", yscale="log")
    p.set_smooth(0.8)
    p.log(0, loss=1.0, acc=0.5, lr=1e-3)  # lr's panel is created after the setters were called
    for panel in p.panels:
        assert panel.spec["xlabel"] == "examples" and panel.spec["yscale"] == "log" and panel.spec["smooth"] == 0.8
    p.set_ylim(0, 10)
    assert all(panel.spec["ylim"] == (0.0, 10.0) for panel in p.panels)
    with pytest.raises(AssertionError):
        p.set_smooth(1.5)


def test_zero_config_then_configure():
    p = LivePlot()
    p.set_title("my run")  # before any panel exists
    p.log(0, loss=1.0)
    assert p.panels[0].spec["title"] == "my run"
    p["val_loss"].set_ylabel("held-out")  # addressing a metric not logged yet creates its panel
    assert p.panels[0].metrics == ["loss", "val_loss"], "no layout given: the new metric joins the shared panel"
    assert p.panels[0].spec["ylabel"] == "held-out"


def test_figure_reflects_setters():
    p = LivePlot("loss | acc", progress=False)
    for step in range(5):
        p.log(step, loss=1.0 / (step + 1), acc=step / 5)
    p["acc"].set_ylim(0, 1)
    p["acc"].set_ylabel("accuracy")
    p.panels[0].set_title("hello")
    p.panels[0].axvline(2, "two")
    fig = p.figure()
    ax, ax2 = fig.axes[0], fig.axes[1]
    assert ax.get_title() == "hello" and ax2.get_ylabel() == "accuracy" and ax2.get_ylim() == (0.0, 1.0)
    assert any(t.get_text().strip() == "two" for t in ax.texts)
