# Soccer ML: Graph Attention Network v2 (GATv2) with $D_2$ Group Equivariance

This repository implements a **Graph Attention Network v2 (GATv2)** with **$D_2$ dihedral group equivariance** for soccer tracking and event analytics (including StatsBomb 360 data). 

The implementation faithfully reproduces the mathematical methods from:
- **TacticAI: an AI assistant for football tactics** ([arXiv:2310.10553](https://arxiv.org/abs/2310.10553), Nature Communications) by Google DeepMind & Liverpool FC.
- **DeepMind CLRS Algorithmic Reasoning Benchmark** ([google-deepmind/clrs](https://github.com/google-deepmind/clrs), `processors.py`).

---

## Key Features

1. **Pure PyTorch Implementation**:
   - Faithful translation of DeepMind's CLRS `GATv2` and `GATv2FullD2` processors into clean `torch.nn.Module` classes.
   - Zero dependency on heavy or version-locked external C++ graph libraries, enabling instant setup on macOS (Apple Silicon `mps` acceleration) and Linux (`cuda`).

2. **Geometric Deep Learning & $D_2$ Group Symmetries**:
   - Football pitches exhibit quadrant reflection symmetries represented by the Klein four-group / dihedral group $D_2 = \{\mathrm{id}, \leftrightarrow, \updownarrow, \leftrightarrow\updownarrow\}$.
   - **Group Convolutions**: Interleaves all 4 pitch reflection views through Cayley composition ($g^{-1}h$) and shared GATv2 message-passing (TacticAI Eq. 8).
   - **Frame Averaging**: Aggregates all 4 views for invariant tasks (receiver prediction, shot prediction, creativity ratings).
   - **Equivariant Generative Head**: Preserves spatial orientation in identity view $\mathbf{H}_{\mathrm{id}}$ for counterfactual position/velocity adjustments using a conditional VAE.

3. **Multi-Task Decoders**:
   - **Receiver Prediction**: Node-level classification identifying the target/recipient of a pass or set piece.
   - **Threatening Shot / Event Prediction**: Graph-level binary classification estimating the likelihood of a shot.
   - **Player Creativity Index**: Node-level continuous scoring head to evaluate individual player decision-making.
   - **Counterfactual Adjustments**: VAE-based generative head with reparameterization sampling ($\mu + \sigma \odot \epsilon$).

4. **Complete Training Pipeline**:
   - Adam optimizer ($\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$) and $L_2$ weight decay ($10^{-4}$) matching TacticAI hyperparameters.
   - Metric tracking (Top-1 & Top-3 accuracy, F1-score, MSE).
   - Checkpointing, learning rate scheduling, and gradient clipping.

---

## Repository Structure

```
soccer_ml/
├── pyproject.toml              # Dependencies in main group (torch, numpy, scipy, pyyaml, pytest, tqdm)
├── poetry.toml                 # Poetry config (virtualenvs.create = false to use soccer_ml conda env)
├── configs/
│   └── default_config.yaml     # Hyperparameters matching TacticAI S.Table 1
├── soccer_ml/
│   ├── data/
│   │   ├── schema.py           # SoccerSample & SoccerGraphBatch dataclasses
│   │   ├── transforms.py       # D2 reflections, Cayley composition, pitch normalization
│   │   └── dataset.py          # PyTorch Dataset, batch collation, synthetic soccer generator
│   ├── models/
│   │   ├── gatv2.py            # Pure PyTorch GATv2 layer (CLRS & Brody et al.)
│   │   ├── group_conv.py       # D2 Group Convolution layer (TacticAI Eq. 8)
│   │   ├── backbone.py         # Multi-layer D2 GATv2 encoder backbone
│   │   ├── heads.py            # Invariant node/graph heads & Equivariant VAE head
│   │   └── soccer_gat.py       # Full end-to-end model
│   ├── training/
│   │   ├── losses.py           # CrossEntropy, BCEWithLogits, VAE Loss (MSE + KL)
│   │   ├── metrics.py          # Top-k accuracy, F1, precision, recall, MSE
│   │   └── trainer.py          # Training loop, evaluation, checkpointing, device detection
│   └── utils/
│       └── config.py           # YAML configuration loader
├── scripts/
│   ├── train.py                # CLI script to train models
│   └── evaluate.py             # CLI script to evaluate checkpoints
└── tests/
    ├── test_transforms.py      # Symmetries and Cayley table tests
    ├── test_gatv2.py           # GATv2 attention and masking tests
    ├── test_group_conv.py      # D2 equivariance and backbone tests
    ├── test_models.py          # End-to-end forward and backward tests
    └── test_trainer.py         # Training loop integration test
```

---

## Installation & Setup

We use the existing Conda environment `soccer_ml` with Poetry:

```bash
# 1. Activate your conda environment
conda activate soccer_ml

# 2. Install dependencies using Poetry into the conda environment
poetry install
```

---

## Running Tests

Verify mathematical symmetries, GATv2 attention, equivariance, and training loop:

```bash
pytest tests/ -v
```

---

## Running Training & Evaluation

To launch training using the default configuration (or synthetic soccer dataset):

```bash
# Train for 10 epochs
python scripts/train.py --epochs 10 --batch-size 32

# Evaluate saved checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```
