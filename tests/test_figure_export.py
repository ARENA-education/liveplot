from liveplot import LivePlot


def test_figure_reflects_current_state(tmp_path):
    p = LivePlot("loss | acc", {"metrics": ["lr"], "smooth": 0.9}, total=100, unit="examples", unit_scale=10, progress=False, fixed_x=True)
    for step in range(20):
        p.log(step * 10, loss=1.0 / (step + 1), acc=step / 20, lr=1e-3)
    p.axhline(0.2, "target", metric="loss")
    p.axvline(100, "halfway")
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


def _xticks(p):
    fig = p.figure()
    fig.canvas.draw()
    ax = fig.axes[0]
    lo, hi = sorted(ax.get_xlim())
    return [t.get_text() for t, x in zip(ax.get_xticklabels(), ax.get_xticks()) if lo <= x <= hi]


def _run(n, **kwargs):
    p = LivePlot(progress=False, show=False, **kwargs)
    for i in range(11):
        p.log(n * i / 10, loss=1.0)
    return p


def test_x_ticks_default_to_matplotlib():
    assert _xticks(_run(300_000)) == ["0", "50000", "100000", "150000", "200000", "250000", "300000"]


def test_si_xticks():
    for n, expected in [(300_000, ["0", "50k", "100k", "150k", "200k", "250k", "300k"]),
                        (2_500_000, ["0", "0.5M", "1M", "1.5M", "2M", "2.5M"]),
                        (3e9, ["0", "0.5G", "1G", "1.5G", "2G", "2.5G", "3G"]),
                        (3e12, ["0", "0.5T", "1T", "1.5T", "2T", "2.5T", "3T"]),
                        (5000, ["0", "1000", "2000", "3000", "4000", "5000"])]:  # below 10,000: unchanged
        assert _xticks(_run(n, si_xticks=True)) == expected


def test_si_xticks_past_trillions_are_scientific():
    p = _run(3e15, si_xticks=True)
    assert _xticks(p) == ["0.0", "0.5", "1.0", "1.5", "2.0", "2.5", "3.0"]
    assert p.figure().axes[0].xaxis.get_major_formatter().get_offset() == "1e15"


def test_si_xticks_with_subplots():
    p, (ax_a, ax_b) = LivePlot.subplots(1, 2, progress=False, show=False, si_xticks=True)
    ax_a.plot("a")
    ax_b.plot("b")
    for i in range(11):
        p.log(30_000 * i, a=1.0, b=2.0)
    fig = p.figure()
    fig.canvas.draw()
    assert "300k" in [t.get_text() for t in fig.axes[1].get_xticklabels()]
