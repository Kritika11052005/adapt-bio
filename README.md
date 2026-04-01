# ADAPT-BIO

**Anticipatory Dynamic Attention Pruning Topology — Biologically Inspired**

> mHC stability + slime anticipation + RNA self-edit + starling sparsity + RLM scale + TRIBE validation

## Architecture

```
Layer 0  mHC stable residual        → stable gradients
Layer 1  ADAPT + slime anticipation → early mask signal
Layer 2  RNA-editing mask refinement → refined sparse mask      [SOMA]
Layer 3  Starling sparse topology   → sparse topology           [SOMA]
Layer 4  RLM recursive context      → mask handed to hardware
Layer 5  Hardware accelerators      → validation signal
Layer 6  TRIBE v2 bio check         → biological plausibility
         ↓
         ADAPT-BIO
```

## Novel Contribution: SOMA

**Self-Organizing Mask Anticipation** — three bio-signals unified:
1. **Slime mold** → *when* to look (anticipatory early signal)
2. **Octopus RNA** → *how long* to keep updating (not frozen at 5%)
3. **Starling k=7** → *what shape* the mask takes (local sparse)

## Quick Start

```bash
pip install -e .
python scripts/train.py --config configs/base_config.yaml
```

## Experiments

Run on Kaggle P100 (free). See `experiments/kaggle/`.
