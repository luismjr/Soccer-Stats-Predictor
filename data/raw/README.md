# Raw Data

Original scraped data from FBref before any processing.

## Subdirectories

- **`match/`** - Match-level data (scores, xG, schedules, match report URLs)
- **`season/`** - Season-level team statistics (aggregated performance metrics)

## Files in `match/`

- `match_stats_YYYY-YYYY.csv` - Individual season schedules
- `match_stats_last5.csv` - Combined last 5 seasons

## Files in `season/`

- `team_stats_YYYY-YYYY.csv` - Team aggregates per season

**Note:** Match files are updated by running `src/scraping/matches.py`

