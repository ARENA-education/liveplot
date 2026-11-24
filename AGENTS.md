# Notes for coding agents

liveplot draws live training curves in Jupyter, Colab and the VS Code / Cursor interactive window, with a tqdm bar underneath. Read [`README.md`](README.md) for the user-facing overview and the module docstring at the top of [`liveplot/liveplot.py`](liveplot/liveplot.py) for the design.

## Layout

- `liveplot/liveplot.py`: the whole library. `LivePlot` is the only public name (`liveplot/__init__.py`). `Panel` and `_Axis` are what `plot.panels[i]` and `plot["metric"]` return. `_FigureRenderer` draws, and `_render_worker` runs it in the render process.
- `liveplot/_images.py`: turning arrays and tensors into image grids for `imshow`.
- `tests/`: pytest, one file per feature.
- `docs/`: the MkDocs site. `docs/index.md` includes parts of `README.md` through the `--8<-- [start:…]` / `[end:…]` markers, so edit that text in the README. `docs/api.md` is generated from the docstrings by mkdocstrings.
- `examples/demo.py`: a `# %%` cell-by-cell tour, turned into a notebook for Colab by CI. `examples/make_gif.py` records `docs/demo.gif` from cell 2 of the demo.

## Setup and checks

```
pip install -e ".[test,docs]"
python -m pytest -q tests          # what CI runs (tests.yml)
mkdocs build --strict              # what CI runs (docs.yml); a broken link or docstring fails it
```

If you change `examples/*.py`, also check it still converts: `pip install jupytext && jupytext --to ipynb examples/demo.py` (demo-notebook.yml). Delete the `.ipynb` afterwards: it is not committed.

Run all of these before pushing.

## How the library is built

Keep these when changing code:

- **Never break the training loop.** `log`, iteration and `finish` must not raise because of a drawing problem. If the render process dies or cannot start, warn and fall back (to drawing in a thread, then to collecting only). `tests/test_robustness.py` holds these cases. Bad arguments are the exception: catch them with an assertion where the user passes them, not later in the render process.
- **`log` is cheap.** The training thread only appends numbers and puts messages on a queue. All drawing happens in the render process.
- **`import liveplot` imports nothing third-party.** matplotlib, numpy and pillow are imported inside the functions that need them. IPython and tqdm are optional. The render process imports the module afresh, so this keeps its startup fast. Never import torch: accept tensors by duck typing.
- **Borrow names from the libraries people know.** The loop and x-axis follow tqdm (`total`, `initial`, `unit`, `unit_scale`, `update`, `set_description`). Axis and panel setters use matplotlib's names and signatures (`set_ylim`, `axhline`, `twinx`, `subplots`, `imshow`). `log` follows wandb. Smoothing is wandb's time-weighted EMA with the same 0 to 1 weight. Don't invent a new name when one of these has one.
- **The output is a plain PNG** swapped into one output cell: no widgets and no JavaScript, so every notebook front end behaves the same.

## Documentation

A user-visible change updates, in the same PR:

- the docstring (Google style; it becomes the API reference),
- the guide page in `docs/guide/` for that feature, and `docs/guide/index.md` if you add a page (also add it to `nav` in `mkdocs.yml`),
- `README.md` only if it changes the install, the quickstart or the overview.

Write the docs in short, plain sentences. Each new GIF in `docs/` should be recorded by the library itself (`record=True` and `save_gif`), like `examples/make_gif.py`.

## Deployment

All of these run in GitHub Actions on every push to `main`. Don't push to their output branches by hand:

- `docs.yml` builds the site and force-pushes it to `gh-pages`, which GitHub Pages serves at <https://arena-education.github.io/liveplot/>.
- `demo-notebook.yml` converts `examples/*.py` to notebooks on the orphan `demo` branch, which the Colab links point at.

## Pull requests

Branch off the latest `main`, make one self-contained change per PR, and open it on `ARENA-education/liveplot`. Say in the description what you ran to check it.
