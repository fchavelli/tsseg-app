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

# tsseg-app — Interactive Time Series Segmentation

<p align="center">
  <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://gradio.app"><img src="https://img.shields.io/badge/Gradio-6.x-orange?style=for-the-badge&logo=gradio" alt="Gradio"></a>
  <a href="https://huggingface.co/spaces/fchavelli/tsseg"><img src="https://img.shields.io/badge/🤗-HuggingFace%20Space-yellow?style=for-the-badge" alt="HF Space"></a>
  <a href="https://github.com/fchavelli/tsseg/blob/main/LICENSE"><img src="https://img.shields.io/github/license/fchavelli/tsseg?color=blue&style=for-the-badge" alt="License"></a>
</p>

An interactive web application for **Time Series Segmentation**, powered by [tsseg](https://github.com/fchavelli/tsseg) and deployed on [HuggingFace Spaces](https://huggingface.co/spaces/fchavelli/tsseg).

## Features

- **30+ segmentation algorithms** — Change Point Detection & State Detection
- **Upload your own data** — CSV with optional ground truth labels
- **Built-in datasets** — MoCap (CMU 86) and synthetic generators
- **Dynamic parameter tuning** — interactive sliders, dropdowns, and text inputs
- **Multi-algorithm comparison** — side-by-side visual comparison
- **Standard metrics** — F1, Covering, ARI, NMI, SMS, and more
- **Export results** — download predictions and scores as CSV

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
│   ├── data.py             # Data loading (synthetic, MoCap, CSV)
│   ├── metrics.py          # Evaluation metrics wrapper
│   ├── params.py           # Dynamic parameter rendering
│   ├── plotting.py         # Plotly visualization
│   ├── utils.py            # Helpers (state conversion, z-norm, etc.)
│   └── config/             # YAML configs with tunable parameter ranges
│       ├── pelt.yaml
│       ├── binseg.yaml
│       └── ...
├── assets/                 # Logo and static files
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
| **Evaluation** | Compute F1, ARI, SMS, etc. when ground truth is available |
| **About** | Full list of available algorithms and links |

## GPU Algorithms on CPU

Algorithms requiring PyTorch (TIRE, TGLAD, E2USD, Time2State, VQTSS) and TensorFlow (TSCP2) are included and run on CPU. They will be slower than on GPU but fully functional. The environment variable `CUDA_VISIBLE_DEVICES=""` is set automatically.

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
