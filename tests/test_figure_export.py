from liveplot import LivePlot


def test_figure_reflects_current_state(tmp_path):
    p = LivePlot("loss | acc", {"metrics": ["lr"], "smooth": 5}, total=100, unit="examples", unit_scale=10, progress=False)
    for step in range(20):
        p.log(step * 10, loss=1.0 / (step + 1), acc=step / 20, lr=1e-3)
    p.hline(0.2, "target", metric="loss")
    p.mark("halfway", x=100)
    fig = p.figure()
    axes = [ax for ax in fig.axes if ax.get_visible()]
    assert len(axes) == 3  # panel 1 left + right axis, panel 2
    ax_loss = axes[0]
    labels = [t.get_text() for t in ax_loss.get_legend().get_texts()]
    assert labels == ["loss", "acc", "target"]
    assert ax_loss.get_xlim() == (0, 1000) and ax_loss.get_xlabel() == "examples"
    assert list(ax_loss.get_lines()[0].get_xdata()) == [step * 10 for step in range(20)]
    assert any(t.get_text().strip() == "halfway" for t in ax_loss.texts)
    out = tmp_path / "run.png"
    fig.savefig(out)
    assert out.stat().st_size > 1000
    p.log(200, loss=0.01)  # a later figure() sees the new point; the old figure is untouched
    fig2 = p.figure()
    assert len([ax for ax in fig2.axes if ax.get_visible()][0].get_lines()[0].get_xdata()) == 21
    assert len(ax_loss.get_lines()[0].get_xdata()) == 20


def test_figure_before_any_data():
    p = LivePlot(progress=False)
    fig = p.figure()
    assert fig.axes and fig.axes[0].get_title() == "waiting for data…"
