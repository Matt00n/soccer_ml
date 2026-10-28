# Measuring Creativity as It Unfolds in Interdependent Teams

Code for the research proposal *Measuring Creativity as It Unfolds in Interdependent Teams* (Mosner & Lehmann),
submitted to the *Organization Science* special issue "AI-Enabled Frontiers in Organizational Science". The
proposal and its technical appendix are in [`reports/`](reports/) (`main.tex` builds both; `technical_appendix.md`
is a Markdown version of the appendix).

We measure the creativity of every on-ball decision in professional football from StatsBomb 360 data. A **surprise
model** (graph attention network over the visible players plus a GRU over the event history) estimates the
probability of each of 1,771 possible actions in a situation, and a **usefulness model** estimates each action's
value (probability of scoring minus conceding). Our creativity measure is the surprise of the chosen action among the
options that would have improved on a typical choice; the **creative opportunity** of a situation is the creativity
a typical player would produce there. All scores are out-of-sample (2-fold cross-fitting plus a reference model on
held-out competitions). This repository is a research prototype under active development.

## Setup

Python 3.12 or newer with PyTorch, NumPy, pandas and SciPy (see `pyproject.toml`), e.g.

```bash
conda create -n soccer_ml python=3.13 && conda activate soccer_ml
pip install -e .          # or: poetry install (installs into the active environment)
pytest tests/             # unit tests on small fixtures in data_test/ and data_toy_receiver/
```

The value baselines in `scripts/value/` (VAEP) additionally need `socceraction` and `xgboost`.

## Data

The data are not part of this repository (`data/` is git-ignored) and are obtained as follows.

**StatsBomb open data** (events and 360 freeze frames, about 15 GB). Use is subject to StatsBomb's terms in the
repository's `LICENSE.pdf`, which require crediting StatsBomb as the data source.

```bash
git clone --depth 1 https://github.com/statsbomb/open-data.git data/open-data
```

All scripts read `data/open-data/data`. Matches with 360 data are selected automatically (426 matches, of which 418
are usable; see Table A1 of the appendix).

**FIFA player ratings** (external validation only), from the public "FIFA 23 complete player dataset" (sofifa
scrape, Kaggle `stefanoleone992/fifa-23-complete-player-dataset`), via its Hugging Face mirror:

```bash
mkdir -p data/external/ratings
curl -L -o data/external/ratings/male_players_legacy.csv \
  "https://huggingface.co/datasets/jsulz/FIFA23/resolve/main/male_players%20(legacy).csv"
curl -L -o data/external/ratings/female_players.csv \
  "https://huggingface.co/datasets/jsulz/FIFA23/resolve/main/female_players.csv"
```

**Known creative players.** The list used for the known-groups test is versioned in
[`validation/known_creative_players.csv`](validation/known_creative_players.csv); its commit time documents that it
was fixed before any player scores were computed.

## Reproducing the results

Run from the repository root. Training uses CUDA or Apple-silicon (MPS) GPUs if available (`--device cpu` forces
the CPU); steps 1 to 3 take about a day on a single Apple M-series machine. Seeds are fixed (`--sample-seed 42`),
but GPU training is not bit-exact, so estimates may differ slightly in the last digits. The scripts
`scripts/run_crossfit_and_lambda.sh` and `scripts/run_reference_and_c51.sh` are the unattended runs we used;
the commands below are the subset needed for the reported results.

```bash
DATA=data/open-data/data
S=outputs/creativity/step2/scored
HOLDOUT=(--holdout-competitions "La Liga" "Ligue 1" "1. Bundesliga" "Major League Soccer" --holdout-teams Argentina)

# 1. Cross-fitted surprise and usefulness models (appendix A.2): per fold, the surprise model is trained on one
#    half of the matches, and the usefulness model (MC targets, C51 head) is warm-started from it.
for FOLD in 0 1; do
  python scripts/train_next_action.py --train-dir $DATA --num-folds 2 --fold $FOLD --sample-seed 42 \
    --game-state --velocities --edge-distance --epochs 16 \
    --checkpoint-dir checkpoints/crossfit/fold$FOLD/next_action
  python scripts/train_value.py --policy-checkpoint checkpoints/crossfit/fold$FOLD/next_action/best_model.pt \
    --train-dir $DATA --num-folds 2 --fold $FOLD --sample-seed 42 \
    --target mc --head decomposed --distribution categorical --epochs 3 \
    --checkpoint-dir checkpoints/crossfit/fold$FOLD/value_mc_c51
done

# 2. Reference model, trained without the club seasons and Argentina's World Cup matches (appendix A.2).
python scripts/train_next_action.py --train-dir $DATA "${HOLDOUT[@]}" --sample-seed 42 \
  --game-state --velocities --edge-distance --epochs 10 --checkpoint-dir checkpoints/reference/next_action
python scripts/train_value.py --policy-checkpoint checkpoints/reference/next_action/best_model.pt \
  --train-dir $DATA "${HOLDOUT[@]}" --sample-seed 42 --target mc --head decomposed --epochs 3 \
  --checkpoint-dir checkpoints/reference/value_mc

# 3. Score every out-of-sample decision (the script refuses matches a model was trained on).
python scripts/creativity/score_decisions.py \
  --value-checkpoint checkpoints/crossfit/fold0/value_mc_c51/best_model.pt --output $S/crossfit_c51_fold0.csv.gz
python scripts/creativity/score_decisions.py \
  --value-checkpoint checkpoints/crossfit/fold1/value_mc_c51/best_model.pt --output $S/crossfit_c51_fold1.csv.gz
python scripts/creativity/score_decisions.py \
  --value-checkpoint checkpoints/reference/value_mc/best_model.pt --output $S/reference_holdout.csv.gz
TABLES="$S/crossfit_c51_fold0.csv.gz $S/crossfit_c51_fold1.csv.gz"

# 4. Validation (appendix A.4): calibration, known creative players, annotation controls, external ratings.
python scripts/creativity/validate_measures.py --tables $TABLES \
  --known-players validation/known_creative_players.csv --output-dir outputs/creativity/validation_c51
python scripts/creativity/calibrate_policy.py --policy-checkpoint checkpoints/crossfit/fold0/next_action/best_model.pt
python scripts/creativity/external_ratings.py --tables $TABLES

# 5. Analyses (appendix A.5 and Table A2).
python scripts/creativity/build_context.py --tables $TABLES
python scripts/creativity/run_second_order.py --tables $TABLES --reference $S/reference_holdout.csv.gz
python scripts/creativity/run_step2.py --tables $TABLES --reference $S/reference_holdout.csv.gz
```

The model performance reported in A.2 is written by the training scripts to `summary.json` in each checkpoint
directory. The result summaries behind the appendix are included in this repository, so they can be checked
without rerunning the pipeline:

| Appendix | Output |
|---|---|
| A.4 calibration, known creative players, annotation controls, position variance | `outputs/creativity/validation_c51/report.md` |
| A.4 temperature scaling | `outputs/creativity/calibration/*/report.md` |
| A.4 external FIFA ratings | `outputs/creativity/step2/A1_external_ratings/report.md` |
| A.5 analyses, one folder per analysis with a `summary.md` | `outputs/creativity/step2/` (`index.md` lists all) |
| Table A2 estimates with Benjamini–Hochberg q-values | `outputs/creativity/step2/claims.csv` |

## Repository structure

| Path | Content |
|---|---|
| `soccer_ml/data/` | StatsBomb loading, decision extraction, pitch grid, match splits |
| `soccer_ml/models/` | Surprise model (`next_action.py`), usefulness model (`value.py`), GATv2 layers |
| `soccer_ml/training/` | Training loops for both models |
| `soccer_ml/creativity/` | Scoring, measures, validation and all analyses |
| `scripts/`, `scripts/creativity/` | Command-line entry points used above |
| `validation/` | Pre-registered validation inputs (known creative players) |
| `reports/` | Proposal and technical appendix (LaTeX and Markdown) |
| `*_README.md` | Notes on individual components (next-action model, value model, validation) |

---

## Background: GATv2 backbone with $D_2$ group equivariance

The sections below document the graph backbone this project started from.

This repository implements a **Graph Attention Network v2 (GATv2)** with **$D_2$ dihedral group equivariance** for soccer tracking and event analytics (including StatsBomb 360 data). 

The implementation faithfully reproduces the mathematical methods from:
- **TacticAI: an AI assistant for football tactics** ([arXiv:2310.10553](https://arxiv.org/abs/2310.10553), Nature Communications) by Google DeepMind & Liverpool FC.
- **DeepMind CLRS Algorithmic Reasoning Benchmark** ([google-deepmind/clrs](https://github.com/google-deepmind/clrs), `processors.py`).

---

### Key Features

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

### Repository Structure

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

### Installation & Setup

We use the existing Conda environment `soccer_ml` with Poetry:

```bash
# 1. Activate your conda environment
conda activate soccer_ml

# 2. Install dependencies using Poetry into the conda environment
poetry install
```

---

### Running Tests

Verify mathematical symmetries, GATv2 attention, equivariance, and training loop:

```bash
pytest tests/ -v
```

---

### Running Training & Evaluation

To launch training using the default configuration (or synthetic soccer dataset):

```bash
# Train for 10 epochs
python scripts/train.py --epochs 10 --batch-size 32

# Evaluate saved checkpoint
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt
```
