"""
A looping video inside the figure's PNG: the figure is drawn once, and the frames of the video are
written as an animated PNG (APNG) whose later frames repaint only the video panel's rectangle.

Why APNG: an APNG *is* a PNG (same signature, same `image/png` MIME type; a viewer that doesn't
animate shows the first frame), so it goes through liveplot's existing display path untouched --
`IPython.display.Image(data=png)` into one `display_id` output -- and every browser engine animates it
in an <img> (Chrome/Edge/Electron since 2017, Firefox since 2008, Safari since 2014). No widgets, no
JavaScript, nothing for the frontend to support beyond showing a PNG.

Why hand-written muxing rather than Pillow's APNG writer: the curves change every redraw, the video
does not. Each video frame's rectangle is compressed once, when the video arrives; a redraw then
encodes only the full first frame and splices the cached rectangles behind it (renumbering chunks
and recomputing CRCs, which is cheap). Pillow would re-diff and re-compress every frame every time.

Phase: every redraw sends a new image, and a browser starts a new image from its first frame. So
the frames are rotated to start where the video should be now, `(now - arrival) * fps`, and the
video plays on across redraws instead of restarting at every one.
"""

from __future__ import annotations

import io
import struct
import zlib

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _chunks(png: bytes):
    """(type, data) for each chunk of a PNG."""
    pos = len(PNG_SIGNATURE)
    while pos < len(png):
        (length,) = struct.unpack(">I", png[pos: pos + 4])
        kind = png[pos + 4: pos + 8]
        yield kind, png[pos + 8: pos + 8 + length]
        pos += 12 + length


def _chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)


def encode(arr, compress_level: int = 6) -> tuple[bytes, bytes]:
    """An (H, W, 3) uint8 array -> (IHDR data, the image data of its IDAT chunks, joined)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG", compress_level=compress_level)
    ihdr, idat = b"", []
    for kind, data in _chunks(buf.getvalue()):
        if kind == b"IHDR":
            ihdr = data
        elif kind == b"IDAT":
            idat.append(data)
    return ihdr, b"".join(idat)


def fit(frame, width: int, height: int):
    """One video frame -> (height, width, 3) uint8, resized the way matplotlib's imshow would show it:
    smoothed when shrinking, blocky (nearest) when enlarging a small picture."""
    import numpy as np
    from PIL import Image

    img = Image.fromarray(frame).convert("RGB")  # (H, W) grayscale, (H, W, 3) or (H, W, 4)
    if img.size != (width, height):
        shrink = width < img.size[0] or height < img.size[1]
        img = img.resize((width, height), Image.BOX if shrink else Image.NEAREST)
    return np.asarray(img)


def apng(first, box, regions: list[bytes], fps: float) -> bytes:
    """
    An endlessly looping APNG: `first` is the whole figure, (H, W, 3) uint8, showing the first video
    frame; `regions` are the compressed rectangles (from `encode`) of the frames that follow, drawn
    at `box` = (x, y, width, height) in pixels from the top left, each replacing the last.
    """
    ihdr, idat = encode(first)
    width, height = struct.unpack(">II", ihdr[:8])
    delay = (max(1, round(1000 / fps)), 1000)  # numerator / denominator, in seconds
    n = 1 + len(regions)
    seq = 0

    def fctl(w, h, x, y):
        nonlocal seq
        data = struct.pack(">IIIIIHHBB", seq, w, h, x, y, *delay, 0, 0)  # dispose NONE, blend SOURCE
        seq += 1
        return _chunk(b"fcTL", data)

    out = [PNG_SIGNATURE, _chunk(b"IHDR", ihdr), _chunk(b"acTL", struct.pack(">II", n, 0)),  # 0 plays = forever
           fctl(width, height, 0, 0), _chunk(b"IDAT", idat)]
    x, y, w, h = box
    for data in regions:
        out.append(fctl(w, h, x, y))
        out.append(_chunk(b"fdAT", struct.pack(">I", seq) + data))
        seq += 1
    out.append(_chunk(b"IEND", b""))
    return b"".join(out)


def is_animated(png: bytes) -> bool:
    """Does this PNG carry an animation? (The acTL chunk has to come before the first IDAT.)"""
    head = png[: png.find(b"IDAT")] if b"IDAT" in png else png
    return b"acTL" in head


def frames(png: bytes):
    """The frames of an (A)PNG as full RGB images, with each one's duration in ms (None for a still)."""
    from PIL import Image

    img = Image.open(io.BytesIO(png))
    out = []
    for i in range(getattr(img, "n_frames", 1)):
        img.seek(i)
        out.append((img.convert("RGB"), img.info.get("duration") if is_animated(png) else None))
    return out
