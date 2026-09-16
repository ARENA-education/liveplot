import time

import pytest

from liveplot import LivePlot

PNG = b"\x89PNG\r\n\x1a\n"


class _FakeHandle:
    def __init__(self):
        self.frames = []

    def update(self, img):
        self.frames.append((time.monotonic(), img.data))


@pytest.fixture
def fake_notebook(monkeypatch):
    h = _FakeHandle()
    monkeypatch.setattr(LivePlot, "_make_display_handle", staticmethod(lambda: h))
    return h


def test_frames_arrive_without_log_calls(fake_notebook):
    """A sparse logger: one log() call, then silence. The frame for it must still appear."""
    p = LivePlot("loss", refresh_seconds=0.1)
    p.log(0, loss=1.0)
    t0 = time.monotonic()
    while not fake_notebook.frames and time.monotonic() - t0 < 90:
        time.sleep(0.05)  # no log() calls at all
    assert fake_notebook.frames, "the collector thread must display the frame on its own"
    n = len(fake_notebook.frames)
    p.log(1, loss=0.5)
    t1 = time.monotonic()
    while len(fake_notebook.frames) == n and time.monotonic() - t1 < 90:
        time.sleep(0.05)
    assert len(fake_notebook.frames) > n and fake_notebook.frames[-1][0] > t1
    p.finish()
    assert p._collector_done.is_set() and not p._collector.is_alive() and not p._proc.is_alive()
    assert all(f[:8] == PNG for _, f in fake_notebook.frames)


def test_finish_shows_final_frame_and_stops_thread(fake_notebook):
    with LivePlot("loss", refresh_seconds=1.0) as p:
        for step in range(5):
            p.log(step, loss=1.0 / (step + 1))
    assert fake_notebook.frames and p.last_png == fake_notebook.frames[-1][1]
    assert not p._collector.is_alive()
