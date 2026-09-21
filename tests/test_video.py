"""Video panels, per-panel tiles and the notebook display (with a stand-in for IPython's display)."""

import base64
import io
import time

import numpy as np
import pytest
from PIL import Image

from liveplot import LivePlot, _video
from liveplot.liveplot import _FigureRenderer, _compose, _png

PNG = b"\x89PNG\r\n\x1a\n"


def clip(t=6, h=31, w=40, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (t, h, w, 3), dtype=np.uint8)


def test_frames_are_normalised():
    frames = _video.as_frames(clip())
    assert frames.shape == (6, 32, 40, 3) and frames.dtype == np.uint8, "odd height padded to even, for H.264"
    assert np.array_equal(frames[:, :31], clip()), "values untouched, the padding repeats the edge"
    assert _video.as_frames(np.zeros((4, 3, 8, 10))).shape == (4, 8, 10, 3), "(T, C, H, W), torch's order"
    assert _video.as_frames(np.zeros((4, 8, 10))).shape == (4, 8, 10, 3), "(T, H, W) grayscale -> RGB"
    ramp = _video.as_frames(np.linspace(-1, 1, 4 * 2 * 2).reshape(4, 2, 2), vmin=-1, vmax=1)
    assert ramp.min() == 0 and ramp.max() == 255, "float scaled from [vmin, vmax]"


def test_encoded_video_passes_through():
    fake_mp4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 20
    assert _video.load(fake_mp4) == (fake_mp4, "video/mp4")
    assert _video.sniff(b"\x1a\x45\xdf\xa3" + b"\x00" * 20) == "video/webm"
    assert _video.load(np.zeros((2, 4, 4))) is None, "frames are not an encoded video"
    with pytest.raises(AssertionError, match="MP4 or WebM"):
        _video.load(b"not a video at all")


def test_encode_h264():
    pytest.importorskip("imageio_ffmpeg")
    data = _video.encode(_video.as_frames(clip(t=10)), fps=10)
    assert _video.sniff(data) == "video/mp4" and len(data) > 500
    first = _video.first_frame(data)
    assert first.shape == (32, 40, 3), "the still read back from the encoded video"


def _renderer_with_data():
    plot, (a, b) = LivePlot.subplots(1, 2, figsize=(10, 4), progress=False, total=50)
    a.plot("loss", "acc")
    a.axhline(0.5, "target", color="red")
    for s in range(50):
        plot.log(s, loss=1 / (s + 1), acc=s / 50)
    b.imshow(np.random.default_rng(0).random((9, 3, 8, 8)))
    r = _FigureRenderer(plot._specs, plot.x_range, plot._layout, images=plot._image_state())
    for name, (xs, ys) in plot.data.items():
        r.hist[name] = (list(xs), list(ys))
    return r


def test_tiles_cover_the_figure_and_compose_back():
    r = _renderer_with_data()
    frame = r.tiles(64)
    w, h = frame["size"]
    (ra, pa), (rb, pb) = frame["tiles"][0], frame["tiles"][1]
    assert ra[0] == 0 and ra[0] + ra[2] == rb[0] and rb[0] + rb[2] == w, "side by side, no gap, no overlap"
    assert ra[3] == rb[3] == h and pa[:8] == pb[:8] == PNG
    whole = np.asarray(Image.open(io.BytesIO(r.render())).convert("RGB")).astype(int)
    composed = np.asarray(Image.open(io.BytesIO(_compose(frame))).convert("RGB")).astype(int)
    assert composed.shape == whole.shape
    right = slice(rb[0], None)  # the picture tile is full colour: identical to the whole figure there
    assert np.array_equal(composed[:, right], whole[:, right])
    assert np.abs(composed - whole).mean() < 2, "the curves tile only loses a little to its palette"


def test_palette_keeps_every_series_colour_exact():
    r = _renderer_with_data()
    rect, png = r.tiles(64)["tiles"][0]
    img = Image.open(io.BytesIO(png))
    assert img.mode == "P" and len(img.getcolors()) <= 64
    colours = {tuple(c) for _, c in img.convert("RGB").getcolors(4096)}
    for c in [(31, 119, 180), (255, 127, 14), (255, 0, 0), (255, 255, 255)]:  # C0, C1, the red target line, white
        assert c in colours, f"{c} lost to the palette"
    assert len(_png(np.zeros((4, 4, 3), np.uint8), None)) > 0, "colors=None: full colour"


class _Handle:
    def __init__(self, log, key):
        self.log, self.key = log, key

    def update(self, obj):
        self.log.append((self.key, obj.data))


@pytest.fixture
def notebook(monkeypatch):
    """IPython's display, replaced: each display() returns a handle that logs its updates."""
    import IPython.display

    log, count = [], [0]

    def fake_display(obj, display_id=None):
        count[0] += 1
        return _Handle(log, count[0])

    monkeypatch.setattr(IPython.display, "display", fake_display)
    return log


def _frame(layout_id=1, tile0=b"a", tile1=b"b", videos=None):
    return {"layout_id": layout_id, "size": (100, 50), "tiles": {0: ((0, 0, 50, 50), tile0), 1: ((50, 0, 50, 50), tile1)},
            "videos": videos or {}}


def test_display_sends_only_what_changed(notebook):
    from liveplot._display import HtmlDisplay

    d = HtmlDisplay(max_mbps=None)
    d.ensure_boxes(2, [1])
    host, box0, box1, vbox = 1, 2, 3, 4
    d.show(_frame())
    assert [k for k, _ in notebook] == [host], "the first frame builds the plot's output"
    notebook.clear()
    d.show(_frame(tile0=b"a2"))
    assert [k for k, _ in notebook] == [box0], "only the tile that changed goes out"
    assert "tile-0" in notebook[0][1] and "Date.now()" in notebook[0][1], "with the send time, for the lag note"
    notebook.clear()
    d.show(_frame(tile0=b"a2"))
    assert notebook == [], "nothing changed, nothing sent"
    d.show(_frame(layout_id=2, tile0=b"a2", videos={1: (60, 5, 30, 30)}))
    assert [k for k, _ in notebook] == [host] and "<video" in notebook[0][1], "a new layout rebuilds the output"
    notebook.clear()
    d.video(1, b"\x00\x00\x00\x18ftyp" + b"v" * 30, "video/mp4")
    assert [k for k, _ in notebook] == [vbox, vbox] and "downloading" in notebook[0][1] and "video/mp4;base64" in notebook[1][1]
    notebook.clear()
    d.finish()
    assert notebook[0][0] == host and "<script" not in notebook[0][1] and "video/mp4;base64" in notebook[0][1], \
        "finish: the final state as plain HTML, video included"
    assert {k for k, text in notebook[1:] if text == ""} == {box0, box1, vbox}, "and the mailboxes emptied"


def test_display_budget_skips_then_catches_up(notebook, monkeypatch):
    import liveplot._display as disp

    d = disp.HtmlDisplay(max_mbps=0.01)  # 2.5 kB per 2 s
    d.ensure_boxes(2)
    d.show(_frame(tile0=b"x" * 1500))  # the host: 2 kB+ already in the window
    notebook.clear()
    d.show(_frame(tile0=b"y" * 1500))
    assert notebook == [] and d.pending == {0}, "over budget: held back"
    d.show(_frame(tile0=b"z" * 1500))
    assert notebook == [] and d.pending == {0}, "still held, and now it is the newest one that waits"
    clock = [time.monotonic() + 5]
    monkeypatch.setattr(disp.time, "monotonic", lambda: clock[0])
    d.flush()
    assert len(notebook) == 1 and base64.b64encode(b"z" * 1500).decode()[:40] in notebook[0][1], \
        "2 s later it goes out: the newest tile, not the stale one (even though it alone exceeds the budget)"
    assert d.pending == set()


def test_video_api_in_a_script(tmp_path):
    """No notebook: the panel shows the first frame (no encoding), and recording captures it."""
    plot, (ax_c, ax_v) = LivePlot.subplots(1, 2, progress=False, record=True, refresh_seconds=0.05)
    ax_c.plot("loss")
    plot.log(0, loss=1.0)
    ax_v.video(clip(t=4, h=16, w=16), fps=5)
    assert plot._specs[1]["kind"] == "video" and plot._videos == {}, "nowhere to play it: nothing encoded"
    fig = plot.figure()
    img = [ax for ax in fig.axes if ax.get_visible()][1].images[0]
    assert img.get_array().shape == (16, 16, 3), "the first frame, as the panel's picture"
    plot.finish()
    with pytest.raises(AssertionError, match="showing curves"):
        ax_c.video(clip())
    q = LivePlot(progress=False)
    q.video(clip(t=2))
    assert [s["kind"] for s in q._specs] == ["video"], "plot.video makes a video panel, like plot.imshow"


def test_missing_encoder_fails_softly(notebook, monkeypatch):
    """No imageio-ffmpeg: one warning naming the fix, a message on the video panel, and no exception."""
    from liveplot._display import HtmlDisplay

    def no_ffmpeg(frames, fps):
        raise ImportError("No module named 'imageio_ffmpeg'")

    monkeypatch.setattr(_video, "encode", no_ffmpeg)
    plot = LivePlot(progress=False)
    plot._html = HtmlDisplay(max_mbps=None)
    plot._html.ensure_boxes(1, [0])
    plot._html.show(_frame(videos={0: (0, 0, 50, 50)}))
    notebook.clear()
    with pytest.warns(UserWarning, match="pip install imageio-ffmpeg"):
        plot._encode_video(0, 0, _video.as_frames(clip(t=2)), None, fps=10)
    assert len(notebook) == 1 and "imageio-ffmpeg is missing" in notebook[0][1], "said on the panel itself"
