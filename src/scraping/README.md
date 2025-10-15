# Web Scraping Scripts

Scripts for collecting Premier League data from FBref.

## Files

- **`matches.py`** - Scrapes basic match data (scores, xG, schedules)
- **`matches_adv.py`** - Scrapes advanced match statistics with checkpointing
- **`get_season.py`** - Scrapes season-level team statistics
- **`clean_season.py`** - Cleans and processes squad data

## Features

**Resilient Scraping:**
- Automatic retry logic with exponential backoff
- Browser-like headers to avoid blocking
- Polite delays between requests (3-8 seconds)

**Checkpointing:**
- `matches_adv.py` saves progress every N matches
- Automatically resumes if interrupted
- Atomic CSV writes prevent corruption

## Usage

```bash
# Get basic matches
python matches.py

# Get advanced stats (with checkpointing)
python matches_adv.py --season 2025-2026 --checkpoint-every 10

# Get season statistics
python get_season.py --season 2024-2025
```

