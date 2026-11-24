# Notes for coding agents

Live training curves in notebooks, with a tqdm bar. User overview: [`README.md`](README.md). Design: the module docstring of [`liveplot/liveplot.py`](liveplot/liveplot.py).

## Layout

- `liveplot/liveplot.py`: the library. `LivePlot` is the only public name. `Panel` / `_Axis` are what `plot.panels[i]` / `plot["metric"]` return. `_FigureRenderer` draws; `_render_worker` runs it in the render process.
- `liveplot/_images.py`: arrays and tensors to image grids, for `imshow`.
- `tests/`: pytest, one file per feature.
- `docs/`: the MkDocs site. `docs/index.md` pulls README text between `--8<-- [start:…]` / `[end:…]` markers; edit it in the README. `docs/api.md` comes from the docstrings.
- `examples/demo.py`: `# %%` tour, built into a Colab notebook by CI. `examples/make_gif.py` records `docs/demo.gif`.

## Checks (run before pushing)

```
pip install -e ".[test,docs]"
python -m pytest -q tests          # tests.yml
mkdocs build --strict              # docs.yml: broken links or docstrings fail
```

If `examples/*.py` changed: `jupytext --to ipynb examples/demo.py` must work (demo-notebook.yml). Don't commit the `.ipynb`.

## Rules the code keeps

- **Never break the training loop.** Drawing problems in `log`, iteration or `finish` warn and fall back (thread, then collect-only); they never raise. Cases live in `tests/test_robustness.py`. Bad arguments assert where the user passes them, not later in the render process.
- **`log` is cheap.** It appends and queues; drawing happens in the render process.
- **`import liveplot` imports nothing third-party.** matplotlib, numpy, pillow load inside the functions that need them; IPython and tqdm are optional. Never import torch; duck-type tensors.
- **Borrow names.** Loop and x-axis: tqdm (`total`, `initial`, `unit`, `unit_scale`, `update`, `set_description`). Setters: matplotlib (`set_ylim`, `axhline`, `twinx`, `subplots`, `imshow`). `log`: wandb. Smoothing: wandb's time-weighted EMA. Don't invent a name one of these already has.
- **Output is a plain PNG** in one output cell. No widgets, no JavaScript.

## Docs

A user-visible change updates, in the same PR:

- the docstring (Google style; becomes the API reference),
- the feature's page in `docs/guide/` (new page: also `docs/guide/index.md` and `nav` in `mkdocs.yml`),
- `README.md` only for install, quickstart or overview changes.

Keep docs terse: short sentences, bullets and tables over paragraphs, one example per point, no restating. Keep the "written by Claude" notice (home page, `copyright` in `mkdocs.yml`, README). New GIFs are recorded by the library itself (`record=True`, `save_gif`), like `examples/make_gif.py`.

## Deployment

CI, on every push to `main`. Don't push to these branches by hand.

- `docs.yml` → `gh-pages`, served by GitHub Pages at <https://arena-education.github.io/liveplot/>.
- `demo-notebook.yml` → the orphan `demo` branch, which the Colab links open.

## Pull requests

Branch off latest `main`, one self-contained change per PR, open it on `ARENA-education/liveplot`, and say what you ran to check it.
