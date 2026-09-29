import numpy as np
import pytest

from liveplot import LivePlot
from liveplot.liveplot import Panel, _Axis, _Line, _parse_fmt


def test_addressing_by_metric_and_by_panel():
    p = LivePlot("loss | acc", "lr")
    p.log(0, loss=1.0, acc=0.5, lr=1e-3)
    assert [type(x) for x in p.axes] == [Panel, Panel] and p.axes[0].metrics == ["loss", "acc"]
    left, right, lr = p["loss"], p["acc"], p["lr"]
    assert isinstance(left, _Axis) and not left._right and right._right and left.panel.spec is p.axes[0].spec
    assert right.metrics == ["acc"] and lr.panel.spec is p.axes[1].spec
    assert repr(p.axes[0]) == "Panel(0: loss | acc)"
    with pytest.raises(AssertionError):
        p.axes[1].right  # no metrics after '|'


def test_matplotlib_named_setters_go_to_the_right_axis():
    p = LivePlot("loss | acc")
    p.log(0, loss=1.0, acc=0.5)
    p["loss"].set_ylabel("cross-entropy")
    p["acc"].set_ylabel("test accuracy")
    p["acc"].set_ylim(0, 1)  # matplotlib's two-argument form
    p["loss"].set_ylim((0.0, 2.0))  # ... and the tuple form
    p["acc"].set_yscale("log")
    p.axes[0].set_title("training")
    p.axes[0].set_xlabel("examples")
    p.axes[0].set_xlim(0, 500)
    spec = p.axes[0].spec
    assert spec["ylabel"] == "cross-entropy" and spec["ylabel2"] == "test accuracy"
    assert spec["ylim"] == (0.0, 2.0) and spec["ylim2"] == (0.0, 1.0) and spec["yscale2"] == "log" and spec["yscale"] == "linear"
    assert spec["title"] == "training" and spec["xlabel"] == "examples" and spec["xlim"] == (0.0, 500.0)
    p["acc"].set_title("via the axis, like a matplotlib Axes")  # panel-level setter reachable from an axis
    assert spec["title"] == "via the axis, like a matplotlib Axes"
    p.axes[0].set_ylabel("left again")  # y-setters on a panel act on its left axis
    assert spec["ylabel"] == "left again"


def test_limits_take_matplotlibs_forms_including_one_sided():
    p = LivePlot("loss | acc")
    p.log(0, loss=1.0, acc=0.5)
    p["loss"].set_ylim(bottom=0)
    p["acc"].set_ylim(top=1)
    p.axes[0].set_xlim(left=10)
    spec = p.axes[0].spec
    assert spec["ylim"] == (0.0, None) and spec["ylim2"] == (None, 1.0) and spec["xlim"] == (10.0, None)
    p["loss"].set_ylim(ymin=0.5, ymax=2)  # matplotlib's aliases
    assert spec["ylim"] == (0.5, 2.0)
    p["loss"].set_ylim()  # nothing given: back to autoscaling
    assert spec["ylim"] is None
    with pytest.raises(AssertionError, match="once"):
        p["loss"].set_ylim(0, ymin=0)


def test_one_sided_limit_pins_one_end_and_autoscales_the_other():
    p = LivePlot("loss | acc", progress=False)
    for step in range(10):
        p.log(step, loss=5.0 - step / 5, acc=0.2 + step / 50)
    p["loss"].set_ylim(bottom=0)
    p["acc"].set_ylim(top=1)
    fig = p.figure()
    ax, ax2 = fig.axes[0], fig.axes[1]
    assert ax.get_ylim()[0] == 0 and 5.0 <= ax.get_ylim()[1] < 6, "bottom pinned, top follows the data"
    assert ax2.get_ylim()[1] == 1 and 0.1 < ax2.get_ylim()[0] <= 0.2, "top pinned, bottom follows the data"


def test_setters_pass_matplotlib_kwargs_through_and_check_them_early():
    p = LivePlot("loss | acc", progress=False)
    for step in (1, 10, 100):  # positive x, for the log x-axis below
        p.log(step, loss=1.0 / step, acc=0.5)
    p.axes[0].set_title("training", loc="left", fontsize=9)
    p["acc"].set_ylabel("accuracy", color="tab:orange")
    p["loss"].set_yscale("symlog", linthresh=0.1)
    p.axes[0].set_xscale("log")
    fig = p.figure()
    ax, ax2 = fig.axes[0], fig.axes[1]
    assert ax.get_title(loc="left") == "training" and ax.title.get_text() == ""
    assert ax2.yaxis.label.get_color() == "tab:orange"
    assert ax.get_yscale() == "symlog" and ax.get_xscale() == "log"
    # matplotlib's own errors, raised here rather than in the render process
    with pytest.raises(ValueError, match="not a valid value for scale"):
        p["acc"].set_yscale("sqrt")
    with pytest.raises(AttributeError, match="fontsizee"):
        p.axes[0].set_title("t", fontsizee=3)
    with pytest.raises(TypeError):
        p.axes[0].legend(locc="upper left")


def test_set_kwargs_and_reference_lines():
    p = LivePlot("loss | acc")
    p.log(0, loss=1.0, acc=0.5)
    p["acc"].set(ylabel="acc", ylim=(0, 1), yscale="log")
    p.axes[0].set(title="t", xlabel="x", smooth=0.5)
    spec = p.axes[0].spec
    assert (spec["ylabel2"], spec["ylim2"], spec["yscale2"], spec["title"], spec["xlabel"], spec["smooth"]) == ("acc", (0.0, 1.0), "log", "t", "x", 0.5)
    p["loss"].axhline(0.1, color="red", label="target")
    p["acc"].axhline(0.9)
    p["acc"].axhline(0.5, 0.25, 0.75)  # matplotlib's xmin / xmax
    p.axes[0].axvline(3, label="here", linewidth=2)
    p.log(7, loss=0.5)
    p.axes[0].axvline(p.step, label="now")  # the current point of the run, spelt out
    assert spec["axhlines"] == [{"color": "red", "label": "target", "y": 0.1}]
    assert spec["axhlines2"] == [{"y": 0.9}, {"y": 0.5, "xmin": 0.25, "xmax": 0.75}]
    assert spec["axvlines"] == [{"label": "here", "linewidth": 2, "x": 3.0}, {"label": "now", "x": 7.0}]
    with pytest.raises(ValueError, match="xmin"):
        p["loss"].axhline(0.1, "target")  # matplotlib's second argument is xmin, not a label


def test_set_all_applies_to_every_panel_and_later_ones():
    p = LivePlot("loss", "acc")
    p.set_all(xlabel="examples", yscale="log", smooth=0.8)
    p.log(0, loss=1.0, acc=0.5, lr=1e-3)  # lr's panel is created after set_all
    for panel in p.axes:
        assert panel.spec["xlabel"] == "examples" and panel.spec["yscale"] == "log" and panel.spec["smooth"] == 0.8
    p.set_all(ylim=(0, 10))
    assert all(panel.spec["ylim"] == (0.0, 10.0) for panel in p.axes)
    with pytest.raises(AssertionError):
        p.set_all(smooth=1.5)
    with pytest.raises(AssertionError, match="no set_colour"):
        p.set_all(colour="red")


def test_the_plot_is_the_figure():
    p = LivePlot("loss", "acc", progress=False)
    p.log(0, loss=1.0, acc=0.5)
    p.suptitle("run 3", fontsize=14)
    p.supxlabel("examples")
    fig = p.figure()
    assert fig._suptitle.get_text() == "run 3" and fig._suptitle.get_fontsize() == 14
    assert fig._supxlabel.get_text() == "examples"
    assert [ax.get_title() for ax in fig.axes if ax.get_visible()] == ["loss", "acc"], "panel titles untouched"
    for name in ("set_title", "set_xlabel", "set_ylim", "axhline", "axvline", "panels"):
        assert not hasattr(p, name), f"{name} would act on every panel; Figure has no such method"


def test_zero_config_then_configure():
    p = LivePlot()
    p.set_all(title="my run")  # before any panel exists
    p.log(0, loss=1.0)
    assert p.axes[0].spec["title"] == "my run"
    p["val_loss"].set_ylabel("held-out")  # addressing a metric not logged yet creates its panel
    assert p.axes[0].metrics == ["loss", "val_loss"], "no layout given: the new metric joins the shared panel"
    assert p.axes[0].spec["ylabel"] == "held-out"


def test_plot_takes_fmt_strings_and_line2d_kwargs():
    assert _parse_fmt("r--") == {"color": "r", "linestyle": "--"}
    assert _parse_fmt("o") == {"marker": "o", "linestyle": "None"}, "a marker alone means no line, as in matplotlib"
    assert _parse_fmt("C2-.") == {"color": "C2", "linestyle": "-."}
    assert _parse_fmt("tab:orange") == {"color": "tab:orange"} and _parse_fmt("#1f77b4") == {"color": "#1f77b4"}
    assert _parse_fmt("lossG") is None
    plot, ax = LivePlot.subplots(progress=False)
    lines = ax.plot("loss", "r--", label="train loss", alpha=0.5)
    assert isinstance(lines, list) and len(lines) == 1 and isinstance(lines[0], _Line)
    (line,) = lines
    assert line.get_label() == "train loss"
    line.set_linewidth(3)
    line.set(marker="o")
    assert ax.spec["styles"]["loss"] == {"color": "r", "linestyle": "--", "label": "train loss", "alpha": 0.5,
                                         "linewidth": 3, "marker": "o"}
    ax.lines[0].set_color("k")  # Axes.lines
    for step in range(3):
        plot.log(step, loss=1.0 / (step + 1))
    drawn = plot.figure().axes[0].get_lines()[0]
    assert (drawn.get_color(), drawn.get_linestyle(), drawn.get_linewidth(), drawn.get_alpha(), drawn.get_marker()) == ("k", "--", 3, 0.5, "o")
    assert [t.get_text() for t in plot.figure().axes[0].get_legend().get_texts()] == ["train loss"]
    with pytest.raises(AttributeError, match="colr"):
        ax.plot("loss", colr="r")
    with pytest.raises(AssertionError, match="log()"):
        ax.plot("loss", xdata=[1, 2])


def test_two_names_are_matplotlibs_x_and_y_so_they_raise():
    plot, ax = LivePlot.subplots(progress=False)
    with pytest.raises(TypeError, match=r"one metric per call.*'lossG' against 'lossD'.*ax.plot\('lossD'\); ax.plot\('lossG'\)"):
        ax.plot("lossD", "lossG")
    with pytest.raises(TypeError, match="one metric per call"):
        ax.plot("lossD", "lossG", "r--")
    with pytest.raises(AssertionError, match="not data"):
        ax.plot(np.arange(3))


def test_underscore_labels_stay_out_of_the_legend():
    p = LivePlot("loss | acc", progress=False)
    p.log(0, loss=1.0, acc=0.5)
    p["acc"].panel.right.plot("acc", label="_hidden")
    p["loss"].axhline(0.5)  # unlabelled: no entry, as in matplotlib
    p["loss"].axhline(0.2, label="target")
    fig = p.figure()
    assert [t.get_text() for t in fig.axes[0].get_legend().get_texts()] == ["loss", "target"]


def test_figure_reflects_setters():
    p = LivePlot("loss | acc", progress=False)
    for step in range(5):
        p.log(step, loss=1.0 / (step + 1), acc=step / 5)
    p["acc"].set_ylim(0, 1)
    p["acc"].set_ylabel("accuracy")
    p.axes[0].set_title("hello")
    p.axes[0].axvline(2, label="two")
    fig = p.figure()
    ax, ax2 = fig.axes[0], fig.axes[1]
    assert ax.get_title() == "hello" and ax2.get_ylabel() == "accuracy" and ax2.get_ylim() == (0.0, 1.0)
    assert "two" in [t.get_text() for t in ax.get_legend().get_texts()], "a labelled axvline is in the legend"
