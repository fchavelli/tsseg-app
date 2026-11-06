---
title: tsseg
emoji: 📈
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: "6.6.0"
python_version: "3.10"
app_file: app.py
pinned: false
---

# tsseg-app - Interactive Time Series Segmentation

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://gradio.app"><img src="https://img.shields.io/badge/Gradio-6.x-orange?style=for-the-badge&logo=gradio" alt="Gradio"></a>
  <a href="https://huggingface.co/spaces/fchavelli/tsseg"><img src="https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow?style=for-the-badge" alt="HF Space"></a>
  <a href="https://github.com/fchavelli/tsseg/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fchavelli/tsseg?color=blue&style=for-the-badge" alt="License"></a>
</p>

An interactive web application for **Time Series Segmentation**, powered by [tsseg](https://github.com/fchavelli/tsseg) and deployed on [HuggingFace Spaces](https://huggingface.co/spaces/fchavelli/tsseg).

## Features

- **30+ segmentation algorithms** - Change Point Detection & State Detection
- **Upload your own data** - CSV with optional ground truth labels
- **Built-in datasets** - MoCap (CMU 86) and reproducible synthetic series with recurring states
- **Dynamic parameter tuning** - interactive sliders, dropdowns, and text inputs
- **Multi-algorithm comparison** - side-by-side visual comparison
- **Standard metrics** - F1, Covering, ARI, NMI, SMS, and an SMS error-type breakdown
- **Export results** - download predictions and scores as CSV
- **Taxonomy** - the interactive map of the detectors from the [documentation](https://fchavelli.github.io/tsseg/taxonomy.html): search procedure × homogeneity criterion
- **Same look as the documentation** - palette, type, light and dark modes of [fchavelli.github.io/tsseg](https://fchavelli.github.io/tsseg/)

## Quick Start

### Online (HuggingFace Spaces)

👉 **[huggingface.co/spaces/fchavelli/tsseg](https://huggingface.co/spaces/fchavelli/tsseg)**

### Local

```bash
git clone https://github.com/fchavelli/tsseg-app.git
cd tsseg-app
make install
make run
```

The app will be available at `http://localhost:7860`.

## Architecture

```
tsseg-app/
├── app.py                  # Main Gradio application
├── requirements.txt        # Dependencies (for HF Spaces)
├── pyproject.toml          # Project metadata
├── Makefile                # Dev commands
├── src/
│   ├── algorithms.py       # Detector discovery & introspection
│   ├── about_data.py       # Licences, sources, metric and dataset tables (About tab)
│   ├── data.py             # Data loading (synthetic, MoCap, CSV)
│   ├── home.py             # Top bar, Home tab and tab headings (HTML)
│   ├── metrics.py          # Evaluation metrics wrapper
│   ├── params.py           # Dynamic parameter rendering
│   ├── plotting.py         # Plotly visualization
│   ├── style.py            # Gradio theme, CSS and head script (look of the documentation)
│   ├── taxonomy.py         # Taxonomy of the documentation (placement, names, complexity)
│   ├── taxonomy_spec.json  # Copy of docs/_ext/tsseg_taxonomy.py (scripts/sync_taxonomy.py)
│   ├── utils.py            # Helpers (state conversion, z-norm, etc.)
│   └── config/             # YAML configs with tunable parameter ranges
│       ├── pelt.yaml
│       ├── binseg.yaml
│       └── ...
├── assets/                 # CSS, SVG figures, logo, vendored taxonomy (taxonomy/)
├── scripts/
│   └── sync_taxonomy.py    # Refresh the taxonomy from a tsseg checkout
├── tests/                  # Test suite
└── .github/workflows/
    └── sync-hf.yml         # CI: auto-deploy to HuggingFace Spaces
```

## Tabs

| Tab | Description |
|-----|-------------|
| **Data** | Load synthetic, MoCap, or upload CSV data |
| **Segmentation** | Choose algorithm, tune parameters, run detection |
| **Comparison** | Compare multiple algorithm runs side by side |
| **Evaluation** | Compute F1, ARI, SMS, etc. when ground truth is available; inspect SMS delay, transition, isolation and missing errors |
| **Taxonomy** | The interactive grid of the documentation: every detector by search procedure and homogeneity criterion, with filters, search and a card per method |
| **About** | Detectors of the demo with their place in the taxonomy, time complexity, licence and source; metrics and datasets |

Detectors carry the names of the documentation (PELT, GGS, Gaussian HMM, …). In the CPD algorithm menu, state detectors carry a compact **[SD]** badge. Their state labels can still be evaluated with SD metrics after a run.

Synthetic data uses a minimum segment length, distinct channel profiles and a reproducible seed. Level matching at boundaries is optional and off by default because it can hide genuine level changes. Gaussian-process patterns are sampled on a bounded knot grid and interpolated, so the full 10,000-point range stays practical.

## GPU Algorithms on CPU

PyTorch-based algorithms (Tire, TGLAD, E2USD and Time2State) run on CPU. The app sets `CUDA_VISIBLE_DEVICES=""` automatically. TensorFlow-based detectors (TSCP2 and SNLDS) and Patss are excluded from the app.

## Taxonomy and look

The taxonomy is declared once, in the documentation of tsseg (`docs/_ext/tsseg_taxonomy.py`, rendered by
`docs/_static/js/taxonomy.js`). The app keeps a copy; refresh it after the taxonomy changes in tsseg:

```bash
python scripts/sync_taxonomy.py ../tsseg   # writes src/taxonomy_spec.json and assets/taxonomy/
```

Capabilities (unsupervised or guided, multivariate, optional extra) are read at runtime from the estimator tags of the
installed tsseg. The figure of the Home tab, `assets/demo.svg`, is the one of the tsseg README.

## Development

```bash
make install    # Create conda env + install deps
make run        # Start the app locally
make test       # Run tests
make lint       # Check style with ruff
```

## Deployment

The app auto-deploys to HuggingFace Spaces via GitHub Actions on push to `main`. Set the `HF_TOKEN` secret in your GitHub repo settings.

## License

[AGPLv3](https://github.com/fchavelli/tsseg/blob/main/LICENSE)

*Built with [Gradio](https://gradio.app) · Powered by [tsseg](https://github.com/fchavelli/tsseg)*
