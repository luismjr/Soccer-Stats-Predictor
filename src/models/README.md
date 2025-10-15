# Machine Learning Models

ML pipeline for match outcome prediction.

## Files

- **`match_predictor.py`** - Complete ML pipeline (feature engineering, training, prediction)

## Features

**Feature Engineering:**
- Venue-specific rolling form (last 3 home/away games)
- Overall rolling form (last 5 games regardless of venue)
- Advanced match statistics (possession, corners, tackles, etc.)
- Match timing and team encoding

**Model:**
- Random Forest Classifier (600 trees)
- Balanced class weighting for imbalanced data
- Optional probability threshold optimization
- Handles newly promoted teams via NaN imputation

**Configuration:**
- YAML-based configuration for easy experimentation
- Separate configs for training vs production prediction
- Configurable hyperparameters and output paths

## Usage

```bash
# Training mode (with validation)
python match_predictor.py --config ../../configs/train_config.yaml

# Production predictions (all available data)
python match_predictor.py --config ../../configs/predict_config.yaml
```

## Outputs

- Predictions CSV (with probabilities)
- Feature table (for debugging)
- Feature importance plot (top 5 features)

