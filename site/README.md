# Tutorial webpage

The public page for the ICAIF '26 tutorial, published with GitHub Pages at
**https://fmuia.github.io/sbi4finance/**.

It is plain HTML and CSS, with no build step:

| File | Contents |
| --- | --- |
| `index.html` | All the text, one block per section (search for `SECTION:`) |
| `style.css` | Colours (variables at the top), layout and typography |
| `assets/` | The favicon and the hero illustration |

The hero illustration (`assets/smile-fan.svg`) is decorative, not a result.
Its colours are in the `<style>` block at the top of the file.

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
deployment → Source: **GitHub Actions**. Pages for a private repository needs
a paid plan (GitHub Pro for a personal account).

**The site is public even though the repository is private.** Put here only
what may be public: no unreleased results, private links or credentials.

## Before announcing the page

- Fill in or remove every `todo` placeholder.
- Delete the draft banner at the top of `index.html`.
- Delete the `<meta name="robots" content="noindex">` line, so that search
  engines can find the page.
