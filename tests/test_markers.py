"""set_marker puts a marker on every point of one axis's curves, legend included, and nowhere else."""

from liveplot import LivePlot


def test_set_marker_on_one_axis():
    p = LivePlot("loss | acc", show=False, progress=False)
    p["acc"].set_marker("o")
    p.log(0, acc=0.1)
    for step in range(10):
        p.log(step, loss=1.0)
    ax_loss, ax_acc = [ax for ax in p.figure().axes if ax.get_visible()]
    assert ax_acc.get_lines()[0].get_marker() == "o"
    assert ax_loss.get_lines()[0].get_marker() == "None"
    assert [h.get_marker() for h in ax_loss.get_legend().legend_handles] == ["None", "o"]


def test_set_marker_none_turns_it_off():
    p = LivePlot("loss | acc", show=False, progress=False)
    p["acc"].set(marker="o")
    p["acc"].set_marker(None)
    p.log(0, acc=0.1)
    assert [ax for ax in p.figure().axes if ax.get_visible()][1].get_lines()[0].get_marker() == "None"
