# SVJ lesson view

This view presents `notebooks/svj_var_pipeline.py` as a six-stage lesson about
the shared inference workflow. All analytical prose, figures, tables, and the
optional live-run control come from existing notebook cells. Keep the notebook
as the source of computations and scientific claims. Preserve the synthetic
illustration's limitations, historical evidence, and percent log-loss estimand.

Use the installed `marimo-studio` skill when editing this view. `index.html`
contains browser-native HTML and inline CSS; it needs no JavaScript framework,
CDN stylesheet, fonts, or icons. Keep the light surface, readable single-column
lesson, six-stage navigation, and blue accent consistent with the notebook's
plots. At narrow widths, let tables and equations scroll within their output
region without shrinking the surrounding text.

All `marimo-cell` hosts belong inside `#app-shell`. They use semantic aliases
bound in the project's `[tool.marimo-studio.cells]` configuration. Bind changed
or replacement cells with Studio's notebook binding API; do not invent runtime
cell IDs or copy values into this view. Keep native output subtrees opaque so
Studio and Lens retain the producing-cell context. Navigation and presentation
labels may be literal HTML.

Build and validate the saved view, then check the actual browser output and the
existing live control's dependent results. A successful build does not execute
the notebook, and a retained prior build does not prove source freshness.
Keep generated `.artifacts/` untracked and preserve authored files.
