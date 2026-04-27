# ultimate-utils (uutils)

[![PyPI](https://img.shields.io/pypi/v/ultimate-utils)](https://pypi.org/project/ultimate-utils/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)

A comprehensive Python utility library for machine learning research, spanning PyTorch training infrastructure, statistical analysis, meta-learning, LLM evaluation, and experiment management.

Built and maintained by [Brando Miranda](https://scholar.google.com/citations?user=_NQJoBkAAAAJ&hl=en) (Stanford, MIT, UIUC).

```python
pip install ultimate-utils
```

```python
import uutils
uutils.hello()
```

## What's Inside

| Module | Description |
|--------|-------------|
| `uutils` | Core utilities — file I/O, argument parsing, serialization (dill/pickle/JSON), git helpers, seed management, progress bars |
| `uutils.torch_uu` | PyTorch training loops, distributed training, checkpointing, optimizers, learning rate schedulers |
| `uutils.torch_uu.models` | Model utilities and HuggingFace integrations |
| `uutils.torch_uu.dataloaders` | Dataloaders for meta-learning (miniImageNet, CIFAR-FS), standard vision, and multi-dataset sampling |
| `uutils.torch_uu.metrics` | CCA/PWCCA/DCCA similarity, model complexity, task diversity (Task2Vec), accuracy with confidence intervals |
| `uutils.torch_uu.meta_learners` | Meta-learning algorithms (MAML, Prototypical Networks, etc.) |
| `uutils.stats_uu` | Hypothesis testing, p-values, effect sizes (Cohen's d), power analysis, confidence intervals, ANOVA, regression |
| `uutils.plot` | Plotting with error bands, heatmaps, bar charts, LaTeX table export |
| `uutils.hf_uu` | HuggingFace training utilities — full fine-tuning, QLoRA/Unsloth, causal and seq2seq LM training |
| `uutils.evals` | LLM evaluation — math benchmarks (MATH, Putnam, OlympiadBench), API inference (Claude, OpenAI, vLLM), answer extraction |
| `uutils.dspy_uu` | DSPy-based synthetic data generation for in-context learning and fine-tuning |
| `uutils.jax_uu` | JAX multi-head attention, layer norm, flash attention implementations |
| `uutils.numpy_uu` | Statistical moments, confidence intervals, matrix utilities |
| `uutils.logging_uu` | Weights & Biases integration — setup, logging, sweeps, model watching |
| `uutils.emailing` | SMTP email + Stanford Outlook (AppleScript) notifications with attachments |
| `uutils.discord_uu` | One-way Discord notifications via webhooks (text, embeds, file uploads) |
| `uutils.whatsapp_uu` | One-way WhatsApp notifications via Meta Cloud API or Twilio |

## Notifications vs. Interactive Agents

The messaging modules above (`emailing`, `discord_uu`, `whatsapp_uu`) are **one-way programmatic notification senders** — small libraries you import from scripts, schedulers, and watchers to push alerts like "job finished", "GPU idle", or "deploy complete". They are not chat agents; there is no conversation loop, no inbound message handling, and no LLM attached.

If you want an **interactive AI assistant that lives inside WhatsApp / Discord / Telegram / iMessage** (reads your messages, replies, executes tasks), that is a different category of tool. Projects like [OpenClaw](https://github.com/steipete/claw-bot) (hosted via [myclaw.ai](https://myclaw.ai)) exist for that use case — they bundle Baileys, grammY, Discord.py, AppleScript bridges, etc., and wire them into an agent loop.

**Rule of thumb:**

| Need | Use |
|------|-----|
| Script on my cluster pings me when training finishes | `uutils.emailing` / `discord_uu` / `whatsapp_uu` |
| I want to *chat* with an agent from my phone and have it triage email, reply to admin tasks, run commands | OpenClaw or similar (not this library) |

Today Brando's default notification channel is email (`uutils.emailing`); Discord and WhatsApp modules are available but optional. For the interactive-agent use case, see issue [#41](https://github.com/brando90/ultimate-utils/issues/41). The executable plan lives at `experiments/01_self_hosted_openclaw/cc_prompt.md` here, with the canonical home being `~/agents-config/experiments/01_self_hosted_openclaw/cc_prompt.md` (mirror it back into agents-config when convenient).

## Installation

> **Note:** PyTorch must be installed separately (with CUDA if you need GPU support).

### Development install (recommended)

```bash
conda create -n uutils python=3.11 -y
conda activate uutils
pip install -e ~/ultimate-utils
```

### pip install from PyPI

```bash
pip install ultimate-utils
```

### Verify installation

```bash
python -c "import uutils; uutils.hello()"
python -c "import uutils; uutils.torch_uu.gpu_test_torch_any_device()"
```

## Quick Examples

### Seed everything for reproducibility

```python
from uutils import seed_everything
seed_everything(42)
```

### Save and load with dill

```python
from uutils import save_with_dill, load_with_dill
save_with_dill(my_object, '~/data/my_object.pkl')
obj = load_with_dill('~/data/my_object.pkl')
```

### Plot with error bands

```python
from uutils.plot import plot_with_error_bands
plot_with_error_bands(x, y, yerr, xlabel='Steps', ylabel='Loss', title='Training Loss')
```

### Statistical testing with effect size

```python
from uutils.stats_uu.effect_size import stat_test_with_effect_size_as_emphasis
stat_test_with_effect_size_as_emphasis(group1_data, group2_data)
```

### W&B logging

```python
from uutils.logging_uu.wandb_logging.common import setup_wandb, log_2_wandb
setup_wandb(args)
log_2_wandb(metrics_dict, step=step)
```

## Publishing to PyPI

```bash
# Bump version in setup.py, then:
cd ~/ultimate-utils && bash scripts/publish_to_pypi.sh
```

## Citation

If you use `ultimate-utils` in your research, please cite:

```bibtex
@software{miranda2024uutils,
  author       = {Brando Miranda},
  title        = {ultimate-utils: A Comprehensive Utility Library for Machine Learning Research},
  year         = {2024},
  publisher    = {PyPI},
  url          = {https://github.com/brando90/ultimate-utils},
  note         = {Available at \url{https://pypi.org/project/ultimate-utils/}}
}
```

You can also find the author's publications on [Google Scholar](https://scholar.google.com/citations?user=_NQJoBkAAAAJ&hl=en).

## Related Publications

This library has supported research in the following publications (among others):

- [Beyond Scale: the Diversity Coefficient as a Data Quality Metric Demonstrates LLMs are Not Yet More Human than Human](https://arxiv.org/abs/2306.13840)
- [Does MAML Only Work via Feature Re-use? A Data Set Centric Perspective](https://arxiv.org/abs/2112.13137)

## License

Apache-2.0
