# Data Directory

This folder contains all data used by the Soccer Stats Predictor.

## Subdirectories

- **`raw/`** - Original scraped data from FBref (match schedules, team stats)
- **`processed/`** - Cleaned and merged data ready for machine learning
- **`prediction/`** - Model outputs (predictions and feature tables)
- **`tmp/`** - Temporary checkpoint files for resume capability

## Data Flow

```
raw/ → processed/ → [ML Model] → prediction/
```

Checkpoint files in `tmp/` allow scraping to resume if interrupted.

