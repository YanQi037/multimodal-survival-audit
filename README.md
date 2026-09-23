# Multimodal Survival Audit

This repository reproduces prediction-level analyses for multimodal survival models. It audits whether a fused predictor uses complementary information from pathology and molecular reference models. It contains no training code, TCGA data, slide images, features, checkpoints, or real patient predictions.

Any model can be audited after its held-out predictions are converted to the common CSV format. This includes MCAT, MOTCat, SurvPath, and other survival models.

## Core idea

Harrell-comparable patient pairs are constructed independently inside each held-out fold. Higher risk denotes worse prognosis. A model receives pair credit 1 for a correct order, 0.5 for a risk tie, and 0 for an incorrect order.

The two unimodal references partition every comparable pair into five exclusive groups:

- `PM`: both pathology and molecular references are correct.
- `P`: only pathology is correct.
- `M`: only molecular is correct.
- `empty`: both references are wrong.
- `tie`: at least one reference assigns tied risks.

`pi_g` is the pooled proportion of pairs in group `g`. `R_P` and `R_M` are the evaluated model's mean pair credits on the `P` and `M` groups. `A_PM`, `A_empty`, and `A_tie` are its mean credits on the corresponding groups. The exact decomposition expresses the pooled C-index difference between a trained fused model and Simple Fusion as the sum of the five group contributions.

## Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows, activate with `.venv\\Scripts\\activate`.

## Input CSV

One row represents one held-out patient. Every patient must occur in exactly one fold.

| Column | Meaning |
|---|---|
| `patient_id` | Arbitrary patient identifier |
| `time` | Observed survival or censoring time |
| `event` | `1` for observed event, `0` for censored |
| `fold` | Official held-out fold identifier |
| `fused_risk` | Trained multimodal model risk |
| `pathology_risk` | Pathology-only risk |
| `molecular_risk` | Molecular-only risk |

Simple Fusion and weight-based analyses additionally need risks standardized using statistics from the corresponding outer-training fold. Supply either:

- `pathology_risk_z` and `molecular_risk_z`; or
- `pathology_train_mean`, `pathology_train_std`, `molecular_train_mean`, and `molecular_train_std`.

The four training-statistic values must be constant within each fold. If `pathology_risk` and `molecular_risk` are already standardized with outer-training statistics, pass `--risks-already-standardized`. The scripts never estimate normalization statistics from held-out patients.

See `examples/example_predictions.csv` for synthetic, non-patient data.

## Commands

Run commands from the repository root.

### 1. Pair audit and exact decomposition

```bash
python scripts/run_pair_audit.py \
  --predictions examples/example_predictions.csv \
  --output-dir outputs/pair
```

Outputs the five-group partition, pooled metrics, exact five-term decomposition, fold-level rescue asymmetry, a pooled rescue summary, and numerical checks.

### 2. Modality dependency audit

Generate a prediction-level Simple Fusion shuffle:

```bash
python scripts/run_dependency_audit.py \
  --predictions examples/example_predictions.csv \
  --output-dir outputs/dependency \
  --generate-simple-fusion-shuffle molecular \
  --donor-seeds 11 22 33 44 55
```

This reports shuffled C-index, C-index change, risk correlations, standardized mean absolute effect, and ranking flip rate. Seeds are averaged within each fold before fold-level mean and population standard deviation are calculated.

For a trained model, inference-time modality shuffling must happen in that model's own forward path. Export each shuffled held-out risk as an additional CSV column, then audit it without rerunning the model:

```bash
python scripts/run_dependency_audit.py \
  --predictions predictions_with_shuffles.csv \
  --output-dir outputs/dependency \
  --shuffled-risk molecular_seed11=molecular_shuffle_seed11_risk \
  --shuffled-risk pathology_seed11=pathology_shuffle_seed11_risk
```

### 3. Less-used-modality permutation null

```bash
python scripts/run_permutation_null.py \
  --predictions examples/example_predictions.csv \
  --output-dir outputs/permutation \
  --dominant pathology \
  --donor-seeds 11 22 33 44 55 \
  --setting-name synthetic_example
```

This preserves the dominant standardized risk, deranges the less-used risk within each fold, and outputs fold-seed results, seed-within-fold summaries, and overall descriptive summaries. It also reports complementary-pair proportions and the pairwise oracle under the real and null references.

### 4. Global weight sweep

```bash
python scripts/run_weight_sweep.py \
  --predictions examples/example_predictions.csv \
  --output-dir outputs/weight_sweep \
  --step 0.05
```

For `risk(w) = w * pathology_z + (1-w) * molecular_z`, this outputs pooled and fold-mean C-index, fold standard deviation (`ddof=0`), `R_P`, and `R_M` at every weight. The summary identifies the best pooled weight and compares it with the trained fused predictor. A numerical check confirms that `w=0.5` exactly reproduces Simple Fusion.

## Scope

All pair construction is fold-local; patients from different folds are never paired. Pooled metrics sum pair credits and denominators across held-out folds. The code only reads prediction CSVs and writes audit tables. It does not download data, train models, or modify prediction files.
