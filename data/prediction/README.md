# Prediction Outputs

Model predictions and feature tables for upcoming matches.

## Files

- `prediction_data.csv` - Predicted outcomes for next 10 upcoming fixtures
- `prediction_features.csv` - Engineered features used to generate predictions

## Contents

**prediction_data.csv:**
- Match details (date, home team, away team)
- Predicted outcome (Home Win, Draw, Away Win)
- Probability scores for each outcome

**prediction_features.csv:**
- All features used by the model
- Rolling form statistics (last 3 venue-specific, last 5 overall)
- Useful for debugging and understanding predictions

## Generation

Created by running:
```bash
python src/models/match_predictor.py --config configs/predict_config.yaml
```

