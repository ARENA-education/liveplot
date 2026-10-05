"""Running outside a notebook: terminal IPython, save= files, and finishing at exit."""

import subprocess
import sys
import textwrap
import time
from pathlib import Path

import numpy as np

import liveplot.liveplot as lp
from liveplot import LivePlot

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


class _TerminalShell:  # what terminal IPython's get_ipython() gives: a shell, but no kernel
    pass


class _KernelShell:  # a notebook kernel's shell (Jupyter, Colab, VS Code)
    kernel = object()


def test_terminal_ipython_is_not_a_notebook(monkeypatch):
    """Terminal IPython would print every frame into the terminal, so it must count as headless."""
    import IPython
    import IPython.display

    handle = object()
    monkeypatch.setattr(IPython.display, "display", lambda *a, **k: handle)  # no real shell to display on
    monkeypatch.setattr(IPython, "get_ipython", lambda: _TerminalShell())
    assert LivePlot._make_display_handle() is None
    monkeypatch.setattr(IPython, "get_ipython", lambda: _KernelShell())
    assert LivePlot._make_display_handle() is handle


def test_save_png_is_rewritten_while_training(tmp_path, monkeypatch):
    monkeypatch.setattr(lp, "_SAVE_EVERY_SECONDS", 0.0)
    path = tmp_path / "run.png"
    p = LivePlot("loss", save=path, refresh_seconds=0)
    t0 = time.monotonic()
    i = 0
    while not path.exists() and time.monotonic() - t0 < 90:  # the render child starts in ~1 s
        p.log(i, loss=1 / (i + 1))
        i += 1
        time.sleep(0.02)
    assert path.exists(), "save= must write the file before finish(), so a killed run leaves a picture"
    assert path.read_bytes().startswith(PNG_MAGIC)
    p.finish()
    assert path.read_bytes().startswith(PNG_MAGIC)
    assert list(tmp_path.iterdir()) == [path], "no temporary file may be left behind"


def test_save_other_formats_at_finish(tmp_path):
    path = tmp_path / "run.pdf"
    p = LivePlot("loss", save=path)
    for i in range(5):
        p.log(i, loss=1 / (i + 1))
    p.finish()
    assert path.read_bytes().startswith(b"%PDF")
    assert list(tmp_path.iterdir()) == [path]


def test_save_is_written_at_exit_if_never_finished(tmp_path):
    """A script that crashes (or forgets finish()) still gets its file, from the atexit hook."""
    path = tmp_path / "run.png"
    script = textwrap.dedent(f"""
        from liveplot import LivePlot
        p = LivePlot("loss", save={str(path)!r})
        for i in range(20):
            p.log(i, loss=1 / (i + 1))
        raise RuntimeError("training crashed")
    """)
    r = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=120)
    assert "training crashed" in r.stderr
    assert path.read_bytes().startswith(PNG_MAGIC)


def _image_title(plot) -> str:
    fig = plot.snapshot()
    return fig.axes[1].get_title()


def test_imshow_title_shows_the_step_unless_asked_not_to():
    plot, (ax_loss, ax) = LivePlot.subplots(1, 2)
    ax_loss.plot("loss")
    ax.set_title("samples")
    for i in range(3):
        plot.log(i, loss=1.0)
    ax.imshow(np.zeros((4, 4)))
    assert _image_title(plot) == "samples (step 2)"
    ax.imshow(np.zeros((4, 4)), show_step=False)
    assert _image_title(plot) == "samples"
    plot.finish()
