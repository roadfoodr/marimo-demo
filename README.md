# Introducing marimo: A Next-Generation Reactive Notebook for Python

Materials for a presentation at the [Charlottesville Data Science Meetup](https://www.meetup.com/charlottesville-data-science-meetup/), introducing [marimo](https://marimo.io) — a reactive Python notebook that is stored as a pure Python program, Git-friendly, reproducible, and designed for data.

## What is marimo?

marimo is a next-generation computational notebook that addresses many of the perceived shortcomings of Jupyter and similar tools. marimo feels like a notebook, but is stored as a pure Python `.py` file that's:

- **Reactive** — change a cell or widget and every dependent cell updates automatically
- **Reproducible** — no hidden state; self-contained environments via PEP 723 + uv
- **Git-friendly** — clean diffs, importable as a module, runnable as a script
- **AI-native** — plain Python format means AI agents write real code, not fragile JSON
- **Deployable** — serve as a web app, host on molab, or export to static HTML via WASM

## Presentation

~15 minutes: 6 minutes of slides followed by a 9-minute live demo.

### Slides

The slide deck covers:
1. Why we use computational notebooks (explore, prototype, narrate, share)
2. How marimo improves on each of those dimensions
3. AI-native design (Claude Code `--watch`, built-in AI assistant)
4. Deployment options (marimo run, molab, WASM, Cloudflare)

### Live Demo

Three pieces, one narrative — **from zero to interactive data app**:

1. **Build from scratch with AI** — start with an empty file, use Claude Code and marimo's AI assistant to author cells live
2. **[MNIST Embedding Explorer](https://molab.marimo.io/github/marimo-team/gallery-examples/blob/main/notebooks/algorithms/visualizing-embeddings.py)** — gallery notebook on [molab](https://molab.marimo.io) exploring digit embeddings with interactive 2D projections
3. **Interactive K-Means** — pre-built notebook with sliders to adjust K and watch clusters update live

## Example Notebooks

| Notebook | Description |
|----------|-------------|
| [`examples/kmeans_plus_plus.py`](examples/kmeans_plus_plus.py) | Interactive K-Means++ clustering with adjustable K |

## Getting Started

```bash
# Install marimo
uv tool install marimo

# Open a notebook
marimo edit --watch notebook.py

# Run as a web app
marimo run notebook.py

# Run in a sandboxed environment (auto-installs dependencies)
marimo edit --sandbox notebook.py
```

## Resources

- [marimo docs](https://docs.marimo.io)
- [marimo gallery](https://marimo.io/gallery)
- [marimo YouTube](https://youtube.com/@marimo-team)
- [molab — free cloud notebooks](https://molab.marimo.io)