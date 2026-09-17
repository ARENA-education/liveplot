import os
import subprocess
import sys
import time

import pytest

import liveplot
import liveplot.liveplot as L
from liveplot import LivePlot, warm


class _FakeHandle:
    def __init__(self):
        self.frames = []

    def update(self, img):
        self.frames.append((time.monotonic(), img.data))


@pytest.fixture(autouse=True)
def _reset_warm():
    yield
    if L._spare is not None and L._spare[0].is_alive():
        L._spare[1].put(None)
        L._spare[0].join(5)
    L._spare, L._keep_warm = None, False


@pytest.fixture
def fake_notebook(monkeypatch):
    h = _FakeHandle()
    monkeypatch.setattr(LivePlot, "_make_display_handle", staticmethod(lambda: h))
    return h


def _wait_warm(timeout=90):
    """A warm spare is ready once it has drawn its throwaway frame; we can't see that, so give it time."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        # the child blocks in inbox.get once warm; a crude but load-tolerant readiness signal is CPU going idle
        with open(f"/proc/{L._spare[0].pid}/stat") as f:
            fields = f.read().split(")")[-1].split()
        state = fields[0]
        if state == "S":
            time.sleep(0.3)
            with open(f"/proc/{L._spare[0].pid}/stat") as f:
                if f.read().split(")")[-1].split()[0] == "S":
                    return
        time.sleep(0.1)


def test_exported():
    assert liveplot.warm is warm


def test_plot_adopts_the_warm_process(fake_notebook):
    warm()
    spare_pid = L._spare[0].pid
    warm()  # idempotent while a spare is waiting
    assert L._spare[0].pid == spare_pid
    p = LivePlot("loss", refresh_seconds=0.1, progress=False)
    assert p._proc.pid == spare_pid, "the plot uses the spare"
    assert L._spare is not None and L._spare[0].pid != spare_pid, "and a replacement spare is started"
    p.log(0, loss=1.0)
    p.finish()
    assert fake_notebook.frames and not p._proc.is_alive()


@pytest.mark.skipif(not os.path.exists("/proc/self/stat"), reason="reads /proc")
def test_warm_first_frame_is_fast(fake_notebook):
    warm()
    _wait_warm()
    t0 = time.monotonic()
    p = LivePlot("loss", refresh_seconds=0.1, progress=False)
    p.log(0, loss=1.0)
    while not fake_notebook.frames and time.monotonic() - t0 < 30:
        time.sleep(0.005)
    first = fake_notebook.frames[0][0] - t0
    p.finish()
    assert first < 0.4, f"first frame took {first:.2f} s from a warm process"


def test_dead_spare_is_replaced(fake_notebook):
    warm()
    L._spare[0].terminate()
    L._spare[0].join(5)
    p = LivePlot("loss", refresh_seconds=0.1, progress=False)  # must not try to use the dead one
    p.log(0, loss=1.0)
    p.finish()
    assert fake_notebook.frames


def test_spare_exits_when_parent_dies():
    code = f"""
import os, sys, time; sys.path.insert(0, {os.path.dirname(os.path.dirname(os.path.abspath(__file__)))!r})
import liveplot.liveplot as L
L.warm(); time.sleep(1.0)
print(L._spare[0].pid, flush=True)
os._exit(0)
"""
    pid = int(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60).stdout.strip())
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(0.2)
    pytest.fail("unused spare outlived its parent")


def test_without_warm_no_spare_is_kept(fake_notebook):
    p = LivePlot("loss", refresh_seconds=0.1, progress=False)
    p.log(0, loss=1.0)
    p.finish()
    assert L._spare is None, "warm() was never called: plots don't leave processes behind"
