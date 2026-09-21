# `imshow` / `subplots` for liveplot — handoff report

Branch: `live-imshow-dev`, off `main` at `05ce94e`. Written for whoever picks this up next.

## Why

ARENA chapter 0.5 (`master_0_5.py`, VAE + DCGAN) shows generated samples with `LiveImage`, defined
in `chapter0_fundamentals/exercises/part5_vaes_and_gans/utils.py:52`. It is an `ipywidgets.Image`
whose `.value` is overwritten with PNG bytes from `samples_to_png()`. Three call sites:
`master_0_5.py:801` (autoencoder), `:1331` (VAE), `:3066` (DCGAN).

Two problems. It is a **widget**, which is the one thing liveplot exists to avoid, and it is a
**separate output area** from the loss curves, so you cannot watch samples sharpen next to the
losses. `samples_to_png` also hardcodes `(b1 b2) c h w`, so it cannot take an unbatched image, and
it upscales by hand with `Image.NEAREST` because MNIST at 28x28 is unreadable — matplotlib's
`imshow` scales to the axes box, so that branch is unnecessary here.

## What was built

Nothing in ARENA has been changed yet. This branch is the liveplot side only.

### New file: `liveplot/_images.py`

`to_grid(x, rows=, cols=, grid_size=, vmin=, vmax=, scale_each=, channels=, pad_value=)` → one
`uint8` array, `(H, W)` or `(H, W, 3|4)`. Pure numpy, no liveplot state, no torch. Everything about
shape reading, tiling and scaling lives here and is unit-tested in isolation
(`tests/test_images.py`).

Runs on the **calling** thread: ~0.19 ms for ten 64x64 RGB samples, and it quarters what the render
process unpickles (480 KiB float32 → 120 KiB uint8). It is also forced — a CUDA tensor cannot be
pickled to another process, so the `.cpu()` hop has to happen caller-side regardless. Torch is
duck-typed (`detach`, `cpu`, then `np.asarray`), never imported.

### `liveplot/liveplot.py`

- Panel specs gained `kind` (`"curve"` / `"image"`) and `cmap`. `_normalise_panel` takes
  `allow_empty=` for the blank panels `subplots()` hands out.
- `_FigureRenderer` gained `self.images` (panel index → array), which survives a re-layout exactly
  as `hist` does, plus `add_image` / `_draw_image`. `_draw_image` reuses the `AxesImage` via
  `set_data` when the shape is unchanged. Image panels get `set_axis_off()` and are skipped by
  `_draw_axvline` (a vertical line across a picture means nothing) and, naturally, by the autoscale
  loop, which iterates `self.lines`.
- New worker message `("image", panel_index, arr)`.
- `LivePlot.subplots(nrows, ncols, figsize=, squeeze=, iterable=, **kwargs)` classmethod returning
  `(plot, axes)`; `_PanelGrid` supports `axes[0][1]`, `axes[0, 1]`, `.flat`, `.shape`.
- `_Axis.plot(*names)`, `Panel.plot(*names)`, `Panel.twinx()`, `Panel.imshow(...)`,
  `LivePlot.imshow(...)`.
- `figure()` and `_fall_back_to_thread` carry the images, as they already carried `hist`.

`record=` needed no work and captures a mixed grid, since recording keeps the rendered PNG: a 1x2
curves-and-samples run wrote a 12-frame animated GIF. That would make a better `docs/demo.gif` than
the present curves-only one — worth doing, via `examples/make_gif.py`.

### Design decisions, and why

Three were settled with David; the reasoning is worth keeping because two of them came out of
checking what matplotlib actually does rather than what it is assumed to do.

1. **`ax.plot("lossD", "lossG")` names metrics.** This looked like a pun on `ax.plot(x, y)` but is
   not: matplotlib genuinely accepts strings as series names via `ax.plot("step", "lossD", data=d)`.
   liveplot is that form with the source implicit. The first sketch had a `secondary="acc"` kwarg;
   that was dropped for `ax.twinx()`, matplotlib's own spelling, which `Panel.right` already backed.

2. **`(3, H, W)` and `(4, H, W)` raise.** matplotlib's own answer is to refuse channels-first
   entirely (`TypeError: Invalid shape (3, 8, 8) for image data`) — it only ever takes `(M, N)`,
   `(M, N, 1)`, `(M, N, 3)`, `(M, N, 4)`. We must accept channels-first because torch uses it, so
   the ambiguity it designed away is ours. Note `(1, H, W)` is **not** ambiguous: both readings draw
   the same pixels (verified), so only two shapes raise, and the message names both escape hatches.
   The cost: `(3, 64, 64)` is "one CelebA image", so anyone eyeballing a single sample hits it. If
   that grates, defaulting to channels-first is a one-line change in `_as_bhwc`.

3. **liveplot scales values itself.** Per-call min-max by default, matching what `samples_to_png`
   does today, with `vmin`/`vmax` to fix the range and `scale_each` for per-image (the name is
   `make_grid`'s). It is delegated to nobody because **matplotlib's `imshow` ignores `vmin`/`vmax`
   for RGB(A) data** and just clips floats to `[0, 1]` — verified identical on 3.9.0 and 3.11.2,
   where `imshow(rgb, vmin=-1, vmax=1)` renders pixel for pixel the same as
   `imshow(clip(rgb, 0, 1))`. The DCGAN generator ends in `tanh`, so 44.8% of its output would go to
   black. Doing it ourselves also makes `vmin`/`vmax` behave uniformly for colour and grayscale,
   which is what people assume and better than matplotlib's asymmetry (scalar data *is* autoscaled).

## A matplotlib 3.9.0 bug you will hit locally

`tests/test_subplots.py::test_process_mode_curves_beside_images` is **skipped on matplotlib
3.9.0–3.9.3**, and David's conda env (`streamlit`) has exactly 3.9.0, so you will see the skip.

On 3.9.0, any RGB(A) image drawn from the **spawned render process** dies with
`ValueError: arrays must be of dtype byte, short, float32 or float64` out of matplotlib's C
extension. What the investigation established:

- grayscale is unaffected; only the multi-channel resample path fails
- the same array in the same figure draws fine in the **parent** process
- `float32` fails too, so it is not our `uint8`
- pure matplotlib in a spawned child is fine — it needs our renderer's state
- `matplotlib._image.resample` called directly with a `uint8` RGBA array is fine in both processes
- it flips to passing when perturbed in ways that cannot matter: calling `get_window_extent` first,
  or wrapping `matplotlib.image._resample` in a transparent Python function
- reproducible with numpy 1.26.4 *and* 2.4.6, so it is not the numpy pairing
- **fixed in matplotlib 3.9.4**; 3.10.x and 3.11.2 are fine

Behaviour that flips on transparent perturbation is memory corruption, not logic, so this was read
as a matplotlib bug and not worked around. `interpolation_stage="rgba"` was tried and does **not**
help (and would need matplotlib ≥3.8 while we declare ≥3.6), so it was reverted. The declared floor
was left at `matplotlib>=3.6`: raising it for the whole library over a bug in three patch releases
seemed worse than a documented skip, especially as ARENA does not pin matplotlib and so gets a good
one. Colour images stay covered in-process by `test_figure_includes_the_image`.

If you want this to stop being a caveat: bump to `matplotlib>=3.9.4` and delete
`_colour_images_are_safe_here` with its skip.

## Tests

`58 passed` on matplotlib 3.11.2 (clean venv); `57 passed, 1 skipped` on 3.9.0 (the conda env).
19 new: `tests/test_images.py` (9) and `tests/test_subplots.py` (10). No existing test changed.

Worth knowing: four of the `test_images.py` expectations were wrong on the first pass and the code
was right — `grid_shape` rounding (`(5, 8, 8)` → a 3x2 grid, not 3x3) and `np.round`'s banker's
rounding (127.5 → 128). But one genuine bug surfaced that way: dropping the tail of an oversized
batch happened **after** scaling, so an image nobody could see still set the range. `to_grid` now
truncates before scaling.

## Left undone

- **The ARENA migration.** Three `LiveImage` call sites, and `ipywidgets` then leaves the chapter's
  dependencies. The DCGAN one wants `LivePlot.subplots(1, 2)` with `lossD`/`lossG` on the left panel
  and `vmin=-1, vmax=1` on the samples; the two VAE ones currently pass `nrows=2` to stack input
  over reconstruction, which maps to `rows=2`.
- **`tight_layout` and mixed panels.** A fixed-aspect image panel beside a curve panel leaves
  whitespace. Width ratios would fix it; not attempted.
- **No inter-image padding.** `make_grid` defaults to `padding=2`, which makes grids notably easier
  to read. Deliberately out of scope for a first pass — it is about four lines in `_tile`.
- **Unpushed on `main`:** nothing. `05ce94e` is on the remote. This branch does not touch
  `.github/workflows/`, deliberately — the OAuth token here lacks the `workflow` scope, and a branch
  push carrying a workflow change is rejected outright.

## Try it

`examples/demo.py` cells 9 and 10 (curves beside a fake generator's samples, and a standalone image
overwritten in place). Both were run headlessly; CI turns the file into the Colab notebook.
