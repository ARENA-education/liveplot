"""Shape reading, grid tiling and value scaling for `imshow` -- all of `_images.py`, no figure."""

import numpy as np
import pytest

from liveplot._images import grid_shape, to_grid


def test_accepted_layouts():
    """Every layout, read into one picture. (H, W) and (H, W, 3) come out with no batch dimension."""
    assert to_grid(np.zeros((28, 28))).shape == (28, 28)
    assert to_grid(np.zeros((1, 28, 28))).shape == (28, 28), "(1, H, W): both readings draw this"
    assert to_grid(np.zeros((64, 64, 3))).shape == (64, 64, 3), "channels last"
    assert to_grid(np.zeros((64, 64, 4))).shape == (64, 64, 4)
    assert to_grid(np.zeros((3, 64, 64)), channels="first").shape == (64, 64, 3)
    assert to_grid(np.zeros((5, 8, 8))).shape == (32, 22), "5 grayscale -> near-square 3x2, one blank, 2px gaps"
    assert to_grid(np.zeros((4, 1, 8, 8))).shape == (22, 22), "(B, 1, H, W) -> 2x2"
    assert to_grid(np.zeros((4, 3, 8, 8))).shape == (22, 22, 3), "(B, C, H, W), torch's layout"
    assert to_grid(np.zeros((4, 8, 8, 3))).shape == (22, 22, 3), "(B, H, W, C), channels last"


def test_the_ambiguous_shapes_raise_and_say_how_to_fix_it():
    """matplotlib dodges this by refusing channels-first entirely; we have to accept it, so we ask."""
    for n, kind in ((3, "RGB"), (4, "RGBA")):
        with pytest.raises(ValueError, match="ambiguous shape") as e:
            to_grid(np.zeros((n, 8, 8)))
        assert kind in str(e.value) and 'channels="first"' in str(e.value) and 'channels="none"' in str(e.value)
    # ... and both escape hatches work, giving genuinely different pictures
    assert to_grid(np.zeros((3, 8, 8)), channels="first").shape == (8, 8, 3)
    assert to_grid(np.zeros((3, 8, 8)), channels="none", padding=0).shape == (16, 16), "3 grayscale -> 2x2, one blank"
    with pytest.raises(ValueError, match="2-, 3- or 4-d"):
        to_grid(np.zeros((2, 3, 4, 5, 6)))
    with pytest.raises(AssertionError, match="not a batch of images"):
        to_grid(np.zeros((4, 7, 8, 9)))  # no dimension that could be channels


def test_grid_shape_inference():
    assert grid_shape(10, rows=2, cols=None) == (2, 5)
    assert grid_shape(10, rows=None, cols=5) == (2, 5)
    assert grid_shape(10, rows=None, cols=None) == (4, 3), "near-square, as for the panel grid"
    assert grid_shape(10, rows=None, cols=None, griddim=(2, 5)) == (2, 5)
    assert grid_shape(7, rows=3, cols=None) == (3, 3), "rounds up, leaving blanks"
    with pytest.raises(AssertionError, match="not both"):
        grid_shape(4, rows=2, cols=2, griddim=(2, 2))


def test_grid_pads_short_and_drops_long():
    imgs = np.arange(3 * 4 * 4, dtype=np.float32).reshape(3, 4, 4)  # 3 distinct grayscale images
    padded = to_grid(imgs, griddim=(2, 2), channels="none", padding=0)
    assert padded.shape == (8, 8)
    assert (padded[4:, 4:] == 0).all(), "the 4th cell is a blank, not a repeat"
    dropped = to_grid(imgs, griddim=(1, 2), channels="none", padding=0)
    assert dropped.shape == (4, 8)
    assert np.array_equal(dropped, to_grid(imgs[:2], griddim=(1, 2), channels="none", padding=0)), \
        "the tail is dropped before scaling, so an image nobody sees cannot set the range"


def test_tiling_puts_images_in_row_major_order():
    imgs = np.stack([np.full((2, 2), v, np.float32) for v in (0.0, 1.0, 2.0, 3.0)])
    g = to_grid(imgs, rows=2, cols=2, channels="none", vmin=0, vmax=3, padding=0)
    assert [g[0, 0], g[0, 3], g[3, 0], g[3, 3]] == [0, 85, 170, 255], "0 1 / 2 3, reading across"


def test_value_scaling():
    x = np.array([[-1.0, 0.0, 1.0]], dtype=np.float32)
    assert to_grid(x).tolist() == [[0, 128, 255]], "no range given: min-max over the batch"
    assert to_grid(x, vmin=-1, vmax=1).tolist() == [[0, 128, 255]], "the same range, fixed"
    assert to_grid(x, vmin=0, vmax=1).tolist() == [[0, 0, 255]], "out-of-range values clip"
    flat = np.full((1, 4, 4), 0.7, np.float32)
    assert (to_grid(flat) == 0).all(), "a constant image comes out at 0, as in make_grid"

    u8 = np.array([[0, 7, 255]], dtype=np.uint8)
    assert to_grid(u8).tolist() == [[0, 7, 255]], "uint8 passes through untouched"
    assert to_grid(u8, vmin=0, vmax=7).tolist() == [[0, 255, 255]], "... unless a range is given"


def test_scale_each():
    """Two images with very different ranges: together the dim one vanishes, apart both fill 0..255."""
    imgs = np.stack([np.array([[0.0, 0.1]]), np.array([[0.0, 1.0]])]).astype(np.float32)
    together = to_grid(imgs, rows=1, cols=2, channels="none", padding=0)
    apart = to_grid(imgs, rows=1, cols=2, channels="none", scale_each=True, padding=0)
    assert together.tolist() == [[0, 26, 0, 255]], "one global range"
    assert apart.tolist() == [[0, 255, 0, 255]], "each image on its own range"
    assert to_grid(imgs, rows=1, cols=2, channels="none", scale_each=True, vmin=0, vmax=1, padding=0).tolist() \
        == [[0, 26, 0, 255]], "an explicit range wins over scale_each"


def test_torch_tensors_without_importing_torch():
    """`to_grid` duck-types .detach()/.cpu(); nothing in liveplot names torch."""
    class FakeTensor:
        def __init__(self, a): self.a, self.detached, self.moved = a, False, False
        def detach(self): self.detached = True; return self
        def cpu(self): self.moved = True; return self
        def __array__(self, dtype=None, copy=None): return self.a

    x = FakeTensor(np.zeros((4, 1, 8, 8), np.float32))
    assert to_grid(x, padding=0).shape == (16, 16)
    assert x.detached and x.moved, "a grad-tracking GPU tensor must be brought back first"
    import liveplot.liveplot as lp
    assert "torch" not in str(lp.__dict__.keys())


def test_pad_value():
    imgs = np.ones((1, 4, 4), np.float32)
    assert (to_grid(imgs, griddim=(1, 2), channels="none", pad_value=255, padding=0)[:, 4:] == 255).all()


def test_padding_between_images():
    """make_grid's layout: `padding` pixels around every image; one image alone gets no border."""
    g = to_grid(np.ones((2, 1, 3, 3), np.float32), griddim=(1, 2), vmin=0, vmax=1, pad_value=7)
    assert g.shape == (3 + 4, 2 * 3 + 6)
    assert (g[2:5, 2:5] == 255).all() and (g[2:5, 7:10] == 255).all(), "the images"
    assert (g[:2] == 7).all() and (g[:, 5:7] == 7).all(), "the border and the gap between"
    assert to_grid(np.ones((3, 3))).shape == (3, 3)


def test_bfloat16_tensors():
    """What autocast hands back; numpy has no bfloat16, so it must be widened before np.asarray."""
    torch = pytest.importorskip("torch")
    x = torch.rand(4, 3, 5, 5, dtype=torch.bfloat16, requires_grad=True)
    assert np.array_equal(to_grid(x), to_grid(x.detach().float().numpy()))
