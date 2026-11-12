# Tutorial webpage

The public page for the ICAIF '26 tutorial, published with GitHub Pages at
**https://fmuia.github.io/sbi4finance/**.

It is plain HTML, CSS and JavaScript, with no build step:

| File | Contents |
| --- | --- |
| `index.html` | All the text, one block per section (search for `SECTION:`) |
| `style.css` | Colours (variables at the top), layout and typography |
| `site.js` | Operating-system setup tabs, including keyboard navigation |
| `theme.js` | Light/dark toggle, saved preference and system-theme detection |
| `hero.js` | Animated background of simulated paths, volatility smiles and a volatility surface |
| `smile.js` | Smoothly interpolates the overview plot's illustrative posterior draws |
| `assets/` | Icons, BayesFlow logo, illustrations, presenter photos and fonts |

The hero animation is decorative, not a result or live market data. Its drawing
code is in `hero.js`; the page layout and text contrast are in `style.css`.
It respects reduced-motion preferences and stops when the hero is off screen
or the browser tab is hidden. The overview plot (`assets/smile-fan.svg`) is also
illustrative. Its curves animate with JavaScript while the labels and quote
markers stay fixed; its caption is part of the SVG. It remains a static image
without JavaScript. Both animations stop for reduced-motion preferences.

All text and headings use Inter, served from `assets/fonts/` alongside its
SIL Open Font License.

The top-right theme toggle remembers a selection in browser storage. Without
a saved selection, the site follows the system theme. Page, plot and hero
colors transition together; reduced-motion preferences disable transitions.

Setup instructions follow `tutorial/pyproject.toml` and `tutorial/README.md`.
The tabs cover Apple Silicon macOS, x86-64 Linux and x86-64 Windows; all three
instruction panels remain visible when JavaScript is disabled.

## Edit

Edit `index.html` directly, on GitHub (the pencil icon) or locally.
Placeholders have `class="todo"` and show as dashed yellow labels; search for
`todo` to find what is left.

To preview locally, open `index.html` in a browser, or run this from the
repository root and open <http://localhost:8000>:

```bash
python -m http.server --directory site 8000
```

## Publish

Every push to `main` that changes `site/` runs the
[Pages workflow](../.github/workflows/pages.yml), which publishes this folder
as it is, usually within a minute or two. To redeploy by hand, open Actions →
"Deploy site to GitHub Pages" → Run workflow.

One-time setup, by the repository owner: Settings → Pages → Build and
deployment → Source: **GitHub Actions**.

**The site and repository are public.** Put here only
what may be public: no unreleased results, private links or credentials.

## Before announcing the page

- Fill in or remove every `todo` placeholder.
- Delete the `<meta name="robots" content="noindex">` line, so that search
  engines can find the page.
