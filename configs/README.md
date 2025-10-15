# Configuration Files

This directory contains YAML configuration files for training the soccer match predictor.

## Quick Start

### 1. Install PyYAML (if not already installed)

```bash
pip install PyYAML
```

### 2. Run with config file

```bash
python src/models/match_predictor.py --config configs/train_config.yaml
```

## Usage Examples

### Use config file only
```bash
python src/models/match_predictor.py --config configs/train_config.yaml
```

### Override specific values from CLI
```bash
# Change test season
python src/models/match_predictor.py --config configs/train_config.yaml --test-season 2026-2027

# Change model parameters
python src/models/match_predictor.py --config configs/train_config.yaml --n-estimators 800

# Change output path
python src/models/match_predictor.py --config configs/train_config.yaml --save-upcoming data/predictions_v2.csv
```

### Use CLI only (no config file)
```bash
python src/models/match_predictor.py \
  --glob "data/processed/matches_with_reports_*.csv" \
  --train-seasons 2021-2022 2022-2023 2023-2024 \
  --val-season 2024-2025 \
  --test-season 2025-2026 \
  --save-upcoming data/upcoming_preds.csv
```

## Config File Structure

### Data Sources
```yaml
data:
  glob: "data/processed/matches_with_reports_*.csv"
```

### Seasons
```yaml
seasons:
  train: ["2021-2022", "2022-2023", "2023-2024"]
  val: "2024-2025"
  test: "2025-2026"
```

### Model Hyperparameters
```yaml
model:
  n_estimators: 600          # Number of trees in Random Forest
  min_samples_split: 14      # Minimum samples to split a node
  min_samples_leaf: 3        # Minimum samples in a leaf
  random_state: 42           # For reproducibility
```

### Threshold Tuning
```yaml
tuning:
  use_thresholds: true       # Enable/disable probability thresholds
  fallback_class: 2          # Default class when uncertain (2=Home Win)
```

### Output Settings
```yaml
output:
  predictions: "data/upcoming_preds.csv"
  features: "data/upcoming_feature_rows.csv"
  feature_importance_plot: true
  plot_path: "feature_importances.png"
```

## Creating Custom Configs

Copy `train_config.yaml` and modify for different experiments:

```bash
# Create experiment config
cp configs/train_config.yaml configs/experiment_v2.yaml

# Edit the new config
# ... modify values ...

# Run with new config
python src/models/match_predictor.py --config configs/experiment_v2.yaml
```

## Priority Rules

**CLI arguments always override config file values.**

Example:
```yaml
# config.yaml has:
model:
  n_estimators: 600
```

```bash
# This will use 800 estimators (CLI overrides config)
python src/models/match_predictor.py --config configs/train_config.yaml --n-estimators 800
```

## Tips

- Keep default settings in `train_config.yaml`
- Create experiment-specific configs for different setups
- Version control your config files to track experiments
- Use comments in YAML files to document choices

