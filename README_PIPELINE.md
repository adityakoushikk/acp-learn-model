# ACP Learn – Experimental Pipeline (Hydra + PyTorch Lightning + WANDB)

This document describes the refactored ACP (Anticancer Peptide) classification pipeline with configurable data splits, iFeature types, models, and WANDB logging.

## Setup

### 1. Install dependencies

```bash
cd /path/to/acp-learn-model
pip install torch lightning pytorch-lightning torchmetrics hydra-core omegaconf pandas numpy scikit-learn pyyaml
# Optional: for WANDB
pip install wandb
```

### 2. WANDB integration (optional but recommended)

1. **Create an account**: Sign up at [wandb.ai](https://wandb.ai).
2. **Get your API key**: In the WANDB UI, go to **Settings → API keys** and copy your key.
3. **Login** (one-time per machine):
   ```bash
   wandb login
   ```
   Paste your API key when prompted. Alternatively set the environment variable:
   ```bash
   export WANDB_API_KEY=your_key_here
   ```
4. **Offline mode**: To log locally without uploading, set in config or CLI:
   ```yaml
   logger.wandb.offline: true
   ```
   or run:
   ```bash
   python -m acp_learn.train logger.wandb.offline=true
   ```

Runs will appear under the project name set in `configs/logger/wandb.yaml` (default: `acp-learn`). You can set **entity** (team/user) there or via CLI: `logger.wandb.entity=your_team`.

---

## Project layout

- **`configs/`** – Hydra configs
  - `train.yaml` – main entry; defaults for data, model, trainer, logger
  - `data/default.yaml` – datamodule, feature types, paths, imbalance, scaler
  - `data/splitter/` – random, stratified_kfold, kmeans_clustered, fixed_test
  - `model/` – mlp, mlp_small (more can be added)
  - `logger/wandb.yaml` – WANDB project and options
- **`src/acp_learn/`** – package
  - `train.py` – Hydra entrypoint
  - `data/` – ACPDataModule, dataset, splitters, iFeature runner
  - `models/` – ACPLitModule, architectures (e.g. DeepMLP)
  - `utils/` – logging, instantiators
- **`scripts/run_ifeature.py`** – standalone iFeature runner (YAML or CLI); used by the pipeline to compute CTDC/CKSAAGP/CTDD (or other types) and merge CSVs.

---

## Running training

From the **project root** (directory containing `src` and `configs`):

```bash
export PROJECT_ROOT=$(pwd)
python -m acp_learn.train
```

Or with overrides:

```bash
# Different splitter
python -m acp_learn.train data=splitter/stratified_kfold data.splitter.n_splits=5 data.splitter.fold_index=0

# Different feature types
python -m acp_learn.train data.feature_types=[CTDC,CTDD,CKSAAGP]

# Fixed test set (e.g. acp-240)
python -m acp_learn.train data=splitter/fixed_test data.test_fasta=${PROJECT_ROOT}/data/acp240.txt

# K-means clustered split
python -m acp_learn.train data=splitter/kmeans_clustered data.splitter.n_clusters=10

# Model and training
python -m acp_learn.train model=mlp_small trainer.max_epochs=50 data.batch_size=32
```

### Multirun (sweeps)

```bash
# Stratified K-Fold over 5 folds
python -m acp_learn.train --multirun data=splitter/stratified_kfold data.splitter.fold_index=0,1,2,3,4

# Multiple feature sets
python -m acp_learn.train --multirun data.feature_types=[CTDC,CTDD],[CTDC,CKSAAGP,CTDD]
```

---

## Tunable parameters (summary)

| Area | Parameter | Config path | Examples |
|------|-----------|-------------|----------|
| **Data** | Splitter | `data=splitter/<name>` | random, stratified_kfold, kmeans_clustered, fixed_test |
| | Feature types | `data.feature_types` | [CTDC, CKSAAGP, CTDD] |
| | Test set | `data.test_fasta` | path to FASTA for fixed test (e.g. acp-240) |
| | Batch size | `data.batch_size` | 64 |
| | Imbalance | `data.imbalance` | class_weight, weighted_sampler, none |
| | Scaler | `data.scaler` | standardize, minmax, none |
| **Model** | Architecture | `model=mlp` or `model=mlp_small` | |
| | Layers / dropout | `model.net.hidden_dims`, `model.net.dropout_rates` | |
| **Trainer** | Epochs / devices | `trainer.max_epochs`, `trainer.devices` | |

---

## Data preparation

1. **Train FASTA**: Place a FASTA file at `data/train.fasta` (or set `data.train_fasta`). Headers determine labels:
   - If header contains the **negative** pattern (default `TRNEGATIVE`) → label 0.
   - If header contains the **positive** pattern (default `seq`) → label 1.
2. **Optional fixed test set**: Set `data.test_fasta` to a path (e.g. `data/acp240.txt`) and use `data=splitter/fixed_test` so the test set is always that file.
3. **iFeature**: The pipeline runs iFeature for each type in `data.feature_types` and merges the tables. Ensure `iFeature/` (with `iFeature.py`) is at `paths.ifeature_dir` (default: project root `iFeature`). You can precompute features by running `scripts/run_ifeature.py` and then pointing `data.features_cache` to the merged CSV to skip recomputation.

---

## iFeature runner (standalone)

Without running the full pipeline, you can generate merged features from a FASTA:

```bash
python scripts/run_ifeature.py --input data/train.fasta --types CTDC CKSAAGP CTDD --out data/processed/features.csv
```

Or with a YAML config:

```yaml
# ifeature_config.yaml
input_fasta: data/train.fasta
feature_types: [CTDC, CKSAAGP, CTDD]
output_csv: data/processed/features.csv
ifeature_dir: iFeature   # or path to iFeature
```

```bash
python scripts/run_ifeature.py --config ifeature_config.yaml
```

---

## Adding new architectures

1. Add a new class in `src/acp_learn/models/architectures/` (e.g. `resnet.py`).
2. Add a config in `configs/model/` (e.g. `resnet.yaml`) with `_target_` pointing to your module and `net._target_` to the new architecture.
3. Run with `model=resnet`.

---

## Logs and checkpoints

- Hydra writes run dirs under `logs/acp_train/runs/<date>_<time>/`.
- WANDB logs metrics and (optionally) config; checkpoints are saved under `checkpoints/` in the same run dir.
- Multiruns go to `logs/acp_train/multiruns/<date>_<time>/0`, `1`, …
