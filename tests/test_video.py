"""`ax.video`: a looping clip in a panel, sent as an animated PNG of the whole figure."""

import io
import time

import numpy as np
import pytest

from liveplot import LivePlot
from liveplot import _video
from liveplot._images import to_grid, to_video
from liveplot.liveplot import _FigureRenderer

PNG = b"\x89PNG\r\n\x1a\n"


def clips(b=4, t=6, c=3, h=8, w=8, seed=0):
    """(B, T, C, H, W) uint8, wandb.Video's batched layout; every frame of every clip different."""
    return np.random.default_rng(seed).integers(0, 256, (b, t, c, h, w), dtype=np.uint8)


def renderer_with_video(frames, fps=10.0):
    """A curve panel and a video panel, as LivePlot.subplots(1, 2) would lay them out."""
    plot, (ax_loss, ax_video) = LivePlot.subplots(1, 2, progress=False, figsize=(6, 2.5), dpi=50)
    ax_loss.plot("loss")
    plot.log(0, loss=1.0)
    plot.log(1, loss=0.5)
    plot._specs[1]["kind"] = "image"  # what ax_video.video() does to the panel
    r = _FigureRenderer(plot._specs, None, plot._layout)
    for name, (xs, ys) in plot.data.items():
        r.hist[name] = (list(xs), list(ys))
    r.add_video(1, frames, 1, fps)
    return r


def test_video_layouts_are_wandbs():
    assert to_video(np.zeros((5, 3, 8, 8))).shape == (5, 8, 8, 3), "(T, C, H, W): one clip"
    assert to_video(np.zeros((4, 5, 3, 8, 8))).shape == (5, 16, 16, 3), "(B, T, C, H, W): a 2x2 grid per frame"
    assert to_video(np.zeros((4, 5, 8, 8, 3))).shape == (5, 16, 16, 3), "channels last"
    assert to_video(np.zeros((5, 8, 8))).shape == (5, 8, 8), "(T, H, W): grayscale"
    assert to_video(np.zeros((16, 5, 8, 8, 3)), rows=2).shape == (5, 16, 64, 3), "imshow's grid keywords"
    with pytest.raises(ValueError, match="3-, 4- or 5-d"):
        to_video(np.zeros((8, 8)))


def test_each_frame_is_the_grid_imshow_would_draw():
    x = clips()
    video = to_video(x)
    for t in range(x.shape[1]):
        assert np.array_equal(video[t], to_grid(x[:, t])), f"frame {t}"


def test_the_range_is_the_whole_clip_not_each_frame():
    """Per-frame scaling would make a dim frame as bright as a light one: the video would flicker."""
    x = np.stack([np.full((4, 4), 0.2), np.full((4, 4), 0.6), np.full((4, 4), 1.0)]).astype(np.float32)
    video = to_video(x, vmin=0, vmax=1)
    assert [int(f[0, 0]) for f in video] == [51, 153, 255]
    video = to_video(x)
    assert [int(f[0, 0]) for f in video] == [0, 128, 255], "min-max over the clip"


def test_the_renderer_sends_the_figure_as_an_endless_apng():
    video = to_video(clips(t=6))
    r = renderer_with_video(video, fps=10)
    png = r.render()
    assert png[:8] == PNG, "still a PNG: the display path doesn't change"
    assert _video.is_animated(png)
    frames = _video.frames(png)
    assert len(frames) == 6 and frames[0][1] == 100, "one image per video frame, 1/fps apart"
    from PIL import Image
    img = Image.open(io.BytesIO(png))
    assert img.info.get("loop") == 0, "loops forever"
    buf = io.BytesIO()
    r.fig.savefig(buf, format="png", dpi=r.dpi)
    assert img.size == Image.open(buf).size, "the same size as the still figure"
    # later frames repaint the video panel's pixel box, and nothing outside it
    x, y, w, h = r.videos[1]["box"]
    bb = r.image_artists[1].get_window_extent()
    assert (w, h) == (round(bb.x1) - round(bb.x0), round(bb.y1) - round(bb.y0)), "the box is the AxesImage's extent"
    first = np.asarray(frames[0][0])
    for image, _ in frames[1:]:
        diff = np.any(np.asarray(image) != first, axis=-1)
        rows, cols = np.nonzero(diff)
        assert rows.size and rows.min() >= y and rows.max() < y + h and cols.min() >= x and cols.max() < x + w


def test_every_redraw_starts_the_video_where_it_has_got_to():
    """A browser starts a new image at its first frame, so each redraw rotates the frames to 'now'."""
    r = renderer_with_video(to_video(clips(t=6)), fps=1)  # a second a frame: slow enough not to race the render
    r.videos[1]["t0"] = time.monotonic() - 0.5  # half way through frame 0
    before = [np.asarray(f) for f, _ in _video.frames(r.render())]
    r.videos[1]["t0"] = time.monotonic() - 2.5  # half way through frame 2
    after = [np.asarray(f) for f, _ in _video.frames(r.render())]
    for k in range(6):
        assert np.array_equal(after[k], before[(k + 2) % 6]), f"frame {k}"


def test_a_still_replaces_a_video_and_a_one_frame_video_is_a_still():
    plot, (ax_loss, ax_img) = LivePlot.subplots(1, 2, progress=False)
    ax_loss.plot("loss")
    ax_img.video(clips(), fps=20)
    assert plot._video_fps == {1: 20}
    ax_img.imshow(clips()[:, 0])
    assert plot._video_fps == {} and plot._images[1].shape == (16, 16, 3)
    r = renderer_with_video(to_video(clips(t=1)))
    assert not _video.is_animated(r.render())
    r = renderer_with_video(to_video(clips()))
    r.add_image(1, to_grid(clips()[:, 0]))
    assert not r.videos and not _video.is_animated(r.render())


def test_video_takes_imshows_panel_rules():
    plot, (ax_loss, ax_a, ax_b) = LivePlot.subplots(1, 3, progress=False)
    ax_loss.plot("loss")
    with pytest.raises(AssertionError, match="showing curves"):
        ax_loss.video(clips())
    ax_a.video(clips())
    with pytest.raises(AssertionError, match="one video per figure"):
        ax_b.video(clips())
    ax_a.video(clips(seed=1))  # the same panel: a new video replaces the old one
    with pytest.raises(AssertionError, match="fps"):
        ax_a.video(clips(), fps=0)
    solo = LivePlot(progress=False)
    solo.video(clips(), fps=8)  # plot-level, like plot.imshow: makes the panel
    assert [s["kind"] for s in solo._specs] == ["image"] and solo._video_fps == {0: 8}


def test_figure_shows_the_first_frame():
    plot, (ax_loss, ax_video) = LivePlot.subplots(1, 2, progress=False)
    ax_loss.plot("loss")
    plot.log(3, loss=1.0)
    ax_video.video(clips(t=5), fps=10)
    fig = plot.figure()
    (img,) = [im for ax in fig.axes for im in ax.images]
    assert np.array_equal(img.get_array(), to_grid(clips(t=5)[:, 0]))
    assert img.axes.get_title() == "step 3"


def test_save_gif_replays_the_video(tmp_path):
    """A recorded APNG stands for the video playing until the next frame: the GIF plays it."""
    still = renderer_with_video(to_video(clips(t=1))).render()
    moving = renderer_with_video(to_video(clips(t=10)), fps=25).render()  # 40 ms a frame
    plot = LivePlot(progress=False)
    plot.frames = [(0.0, still), (0.5, moving), (1.0, moving)]
    gif = plot.save_gif(tmp_path / "v.gif", hold_last_ms=0)
    from PIL import Image
    im = Image.open(gif)
    durations = []
    for k in range(im.n_frames):
        im.seek(k)
        durations.append(im.info["duration"])
    # 500 ms of still, 500 ms of video (12-13 frames), and one full loop at the end (10 frames); Pillow
    # merges identical neighbours, so count time rather than frames
    assert sum(durations) == pytest.approx(500 + 500 + 400, abs=40)
    assert im.n_frames >= 1 + 10 + 10 - 2


def test_save_gif_caps_the_frame_rate(tmp_path):
    moving = renderer_with_video(to_video(clips(t=20)), fps=100).render()  # 10 ms a frame
    plot = LivePlot(progress=False)
    plot.frames = [(0.0, moving)]
    from PIL import Image
    im = Image.open(plot.save_gif(tmp_path / "fast.gif", hold_last_ms=0))
    assert im.n_frames == 5, "one 200 ms loop at 25 fps, not 20 frames GIF viewers would slow down"


class _FakeHandle:
    def __init__(self):
        self.frames = []

    def update(self, img):
        self.frames.append(img.data)


@pytest.fixture
def fake_notebook(monkeypatch):
    h = _FakeHandle()
    monkeypatch.setattr(LivePlot, "_make_display_handle", staticmethod(lambda: h))
    return h


def test_process_mode_loops_the_video_until_the_end(fake_notebook):
    """The RL shape: curves every update, a rollout video now and then, and the last output keeps looping."""
    plot, (ax_curves, ax_video) = LivePlot.subplots(1, 2, total=100, refresh_seconds=0.1, figsize=(6, 2.5), dpi=40)
    ax_curves.plot("return")
    assert plot.mode == "process"
    t0 = time.monotonic()
    while not fake_notebook.frames and time.monotonic() - t0 < 90:  # child startup
        plot.log(0, **{"return": 0.0})
        time.sleep(0.05)
    assert fake_notebook.frames, "render child produced no frame"
    ax_video.video(clips(b=16, t=12), fps=30)
    t0 = time.monotonic()
    step = 1
    while not any(_video.is_animated(f) for f in fake_notebook.frames) and time.monotonic() - t0 < 60:
        plot.log(step, **{"return": float(step)})  # the video arrives with the next redraw
        step += 1
        time.sleep(0.05)
    plot.finish()
    assert any(_video.is_animated(f) for f in fake_notebook.frames), "no animated frame arrived"
    assert all(f[:8] == PNG for f in fake_notebook.frames)
    last = fake_notebook.frames[-1]
    assert _video.is_animated(last) and len(_video.frames(last)) == 12, "the final output is still looping"
    assert not plot._proc.is_alive()
