"""
Turning a tensor into something matplotlib will draw: reading its layout, tiling a batch into a
grid, and scaling the values to uint8.

Nothing here imports torch, and numpy only inside the functions that need it, so `import liveplot`
stays free (see the note at the top of liveplot.py). A caller may hand us a torch tensor, a numpy
array, or anything else with `__array__`.

Why we scale the values ourselves instead of letting `imshow` do it: matplotlib normalises *scalar*
data with vmin/vmax, but for RGB(A) data it ignores vmin/vmax completely and simply clips floats to
[0, 1] -- checked on matplotlib 3.9 and 3.11, where `imshow(rgb, vmin=-1, vmax=1)` renders pixel for
pixel the same as `imshow(clip(rgb, 0, 1))`. A DCGAN generator ends in tanh, so its output lives in
[-1, 1] and matplotlib would clip roughly 45% of it to black with no way to say otherwise. So we do
the scaling and hand `imshow` uint8, which it passes through untouched.
"""

from __future__ import annotations

import math

_CHANNELS = (1, 3, 4)  # channel counts we recognise: grayscale, RGB, RGBA


def _as_array(x):
    """Anything array-like -> a numpy array, without naming torch."""
    import numpy as np

    if hasattr(x, "detach"):
        x = x.detach()  # a tensor that wants grad
    if hasattr(x, "cpu"):
        x = x.cpu()  # ... and may be on a GPU, where numpy can't see it
    return np.asarray(x)


def _ambiguous(shape, n):
    kind = "RGBA" if n == 4 else "RGB"
    return ValueError(
        f"ambiguous shape {shape}: one {kind} image, or {n} grayscale images?\n"
        f'  channels="first" -> one {kind} image      (or pass shape {(1,) + shape})\n'
        f'  channels="none"  -> {n} grayscale images  (or pass shape {(n, 1) + shape[1:]})'
    )


def _as_bhwc(a, channels: str | None):
    """
    Any accepted layout -> (batch, height, width, channels).

    `channels` says how to read a 3-d input: "first" for (C, H, W), "last" for (H, W, C), "none"
    for a batch of grayscale (B, H, W). None works it out:

        (H, W)                      one grayscale image
        (1, H, W)                   one grayscale image -- the two readings draw the same pixels
        (3, H, W) / (4, H, W)       ambiguous, raises (see _ambiguous)
        (H, W, 3) / (H, W, 4)       one image, channels last
        (B, H, W)                   B grayscale images
        (B, 1|3|4, H, W)            B images, channels first
        (B, H, W, 1|3|4)            B images, channels last

    matplotlib itself sidesteps all of this by refusing channels-first outright ("Invalid shape
    (3, H, W) for image data"); it only ever takes (H, W), (H, W, 1), (H, W, 3) or (H, W, 4). We
    have to accept channels-first because that is how torch holds images, so the ambiguity it
    designed away is ours to resolve.
    """
    import numpy as np

    assert channels in (None, "first", "last", "none"), \
        f'channels must be "first", "last", "none" or None, got {channels!r}'
    shape = tuple(a.shape)

    if a.ndim == 2:
        return a[None, :, :, None]

    if a.ndim == 3:
        if channels == "first":
            assert shape[0] in _CHANNELS, f'channels="first" needs {_CHANNELS} channels, got shape {shape}'
            return np.moveaxis(a, 0, -1)[None]
        if channels == "last":
            assert shape[-1] in _CHANNELS, f'channels="last" needs {_CHANNELS} channels, got shape {shape}'
            return a[None]
        if channels == "none":
            return a[..., None]
        if shape[0] == 1:  # (1, H, W): one image either way
            return np.moveaxis(a, 0, -1)[None]
        if shape[0] in (3, 4):
            raise _ambiguous(shape, shape[0])
        if shape[-1] in (3, 4):  # a batch of grayscale 3 or 4 pixels wide is not a thing
            return a[None]
        return a[..., None]

    if a.ndim == 4:
        assert channels != "none", 'channels="none" describes a 3-d (B, H, W) input, not a 4-d one'
        if channels == "last" or (channels is None and shape[1] not in _CHANNELS):
            assert shape[-1] in _CHANNELS, (
                f"shape {shape} is not a batch of images: expected {_CHANNELS} channels at dim 1 "
                f"(B, C, H, W) or dim 3 (B, H, W, C)"
            )
            return a
        return np.moveaxis(a, 1, -1)  # (B, C, H, W), torch's layout, preferred when both could fit

    raise ValueError(f"imshow needs a 2-, 3- or 4-d array, got shape {shape}")


def _to_uint8(a, vmin, vmax, scale_each: bool):
    """Scale to 0..255. uint8 passes straight through unless vmin/vmax ask for a rescale."""
    import numpy as np

    if a.dtype == np.uint8 and vmin is None and vmax is None:
        return a
    a = a.astype(np.float32, copy=False)
    if scale_each:  # per image, like torchvision's make_grid(scale_each=True)
        lo = a.min(axis=(1, 2, 3), keepdims=True) if vmin is None else np.float32(vmin)
        hi = a.max(axis=(1, 2, 3), keepdims=True) if vmax is None else np.float32(vmax)
    else:
        lo = np.float32(a.min() if vmin is None else vmin)
        hi = np.float32(a.max() if vmax is None else vmax)
    a = (a - lo) / np.maximum(hi - lo, 1e-12)  # a constant image comes out at 0, as in make_grid
    return (np.clip(a, 0.0, 1.0) * 255).round().astype(np.uint8)


def grid_shape(n: int, rows: int | None, cols: int | None, grid_size=None) -> tuple[int, int]:
    """
    Rows x cols for `n` images. `grid_size=(rows, cols)` fixes both; one of `rows` / `cols` infers
    the other; with neither, the near-square rule liveplot already uses for its panel grid. Unlike
    the panel grid, the result need not fit: the caller pads short and drops long.
    """
    if grid_size is not None:
        assert rows is None and cols is None, "give grid_size, or rows/cols, not both"
        rows, cols = grid_size
    if rows is None and cols is None:
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
    elif rows is None:
        rows = math.ceil(n / cols)
    elif cols is None:
        cols = math.ceil(n / rows)
    rows, cols = int(rows), int(cols)
    assert rows >= 1 and cols >= 1, f"grid must be at least 1x1, got {rows}x{cols}"
    return rows, cols


def _tile(a, rows: int, cols: int, pad_value: int):
    """(B, H, W, C) -> (rows*H, cols*W, C), padding with blanks or dropping the tail as needed."""
    import numpy as np

    b, h, w, c = a.shape
    cells = rows * cols
    if b > cells:
        a = a[:cells]  # more images than cells: the rest fall off the end
    elif b < cells:
        a = np.concatenate([a, np.full((cells - b, h, w, c), pad_value, dtype=a.dtype)])
    return a.reshape(rows, cols, h, w, c).transpose(0, 2, 1, 3, 4).reshape(rows * h, cols * w, c)


def to_grid(x, *, rows=None, cols=None, grid_size=None, vmin=None, vmax=None,
            scale_each=False, channels=None, pad_value=0):
    """
    A tensor -> one uint8 image, (H, W) for grayscale or (H, W, 3|4), ready for `Axes.imshow`.
    Runs on the calling thread: tiling 10 64x64 RGB samples costs ~0.2 ms and quarters what the
    render process has to unpickle, and a CUDA tensor has to come back to the host here anyway.
    """
    a = _as_bhwc(_as_array(x), channels)
    rows, cols = grid_shape(a.shape[0], rows, cols, grid_size)
    a = a[: rows * cols]  # drop before scaling: an image nobody can see must not set the range
    a = _to_uint8(a, vmin, vmax, scale_each)
    grid = _tile(a, rows, cols, pad_value)
    return grid[..., 0] if grid.shape[-1] == 1 else grid
