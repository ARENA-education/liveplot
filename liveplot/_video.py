"""
Videos for `Panel.video`: frames (or an already-encoded file) -> the bytes of a video a browser plays.

Encoding is H.264 in an MP4 with x264's defaults, through the ffmpeg that `imageio-ffmpeg` bundles, so
there is nothing to install beyond `pip install imageio-ffmpeg`. H.264 because it is the codec every
frontend liveplot targets can play: Colab / Chrome / Firefox / Safari, and VS Code, whose webview ships
with H.264 and VP8 but not VP9 (microsoft/vscode#156558). The videos are silent, which matters in VS
Code: it has no AAC decoder.

Like `_images.py`, nothing here imports torch, and numpy / imageio-ffmpeg only inside the functions.
"""

from __future__ import annotations

import os
import tempfile

_CHANNELS = (1, 3, 4)


def sniff(data: bytes) -> str | None:
    """The MIME type of an encoded video, from its first bytes, or None if it isn't one we know."""
    if len(data) > 12 and data[4:8] == b"ftyp":
        return "video/mp4"
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "video/webm"
    return None


def as_frames(x, vmin=None, vmax=None, channels=None):
    """
    A clip -> (T, H, W, 3) uint8 with even H and W (H.264's 4:2:0 needs them even; an odd edge is
    repeated). Accepts (T, H, W), (T, H, W, C) and (T, C, H, W) with C = 1, 3 or 4, numpy or torch.
    `channels="first"` / `"last"` settles a clip whose shape could be read either way. Values are scaled
    like imshow's: uint8 as is, anything else from [vmin, vmax] (default: the clip's min..max) to 0..255.
    """
    import numpy as np

    from ._images import _as_array, _to_uint8

    a = _as_array(x)
    assert a.ndim in (3, 4), f"video frames must be (T, H, W), (T, H, W, C) or (T, C, H, W), got shape {a.shape}"
    if a.ndim == 3:
        a = a[..., None]
    elif channels == "first" or (channels is None and a.shape[1] in _CHANNELS and a.shape[-1] not in _CHANNELS):
        a = np.moveaxis(a, 1, -1)
    assert a.shape[-1] in _CHANNELS, f"can't find a channel axis of size 1, 3 or 4 in video frames of shape {a.shape}"
    a = _to_uint8(a, vmin, vmax, scale_each=False)
    if a.shape[-1] == 1:
        a = np.repeat(a, 3, axis=-1)
    elif a.shape[-1] == 4:
        a = a[..., :3]
    t, h, w, _ = a.shape
    if h % 2 or w % 2:
        a = np.pad(a, ((0, 0), (0, h % 2), (0, w % 2), (0, 0)), mode="edge")
    return np.ascontiguousarray(a)


def encode(frames, fps: float) -> bytes:
    """(T, H, W, 3) uint8 with even H, W -> MP4 bytes, H.264 with x264's default settings."""
    import imageio_ffmpeg

    t, h, w, _ = frames.shape
    fd, path = tempfile.mkstemp(suffix=".mp4")
    os.close(fd)
    try:
        writer = imageio_ffmpeg.write_frames(
            path, (w, h), fps=fps, codec="libx264", pix_fmt_out="yuv420p", macro_block_size=2,
            output_params=["-movflags", "+faststart"],  # the index first, so playback can start at once
            ffmpeg_log_level="error",
        )
        writer.send(None)
        for frame in frames:
            writer.send(frame)
        writer.close()
        with open(path, "rb") as f:
            return f.read()
    finally:
        os.remove(path)


def load(x):
    """An encoded video given as bytes or a path -> (bytes, MIME type), or None if `x` is frames."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        data = bytes(x)
    elif isinstance(x, (str, os.PathLike)):
        with open(x, "rb") as f:
            data = f.read()
    else:
        return None
    mime = sniff(data)
    assert mime is not None, "video(): bytes / files must be an MP4 or WebM video (or pass the frames instead)"
    return data, mime


def first_frame(data: bytes):
    """The first frame of an encoded video as (H, W, 3) uint8, for the panel's still; None if it can't be read."""
    try:
        import numpy as np
        import imageio_ffmpeg

        fd, path = tempfile.mkstemp(suffix=".mp4" if sniff(data) == "video/mp4" else ".webm")
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        try:
            reader = imageio_ffmpeg.read_frames(path)
            meta = next(reader)
            w, h = meta["size"]
            frame = np.frombuffer(next(reader), np.uint8).reshape(h, w, 3).copy()
            reader.close()
            return frame
        finally:
            os.remove(path)
    except Exception:  # noqa: BLE001 - a still is a nicety; the video plays without it
        return None
