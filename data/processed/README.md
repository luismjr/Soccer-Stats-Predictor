# Processed Data

Match and team data enriched with advanced statistics, ready for machine learning.

## Subdirectories

- **`match/`** - Match-level data with advanced stats (possession, corners, tackles, etc.)
- **`season/`** - Season-level team aggregates

## Files in `match/`

- `match_stats_YYYY-YYYY.csv` - Individual seasons with advanced statistics
- `match_stats.csv` - Combined processed data (optional)

## Files in `season/`

- `team_stats_YYYY-YYYY.csv` - Team performance aggregates per season

## Match File Contents

Each match file contains:
- Basic match info (date, teams, score, xG)
- Advanced stats (possession, corners, fouls, crosses, tackles, touches, interceptions, etc.)
- Match report URLs

## Generation

Match files are created by running:
```bash
python src/scraping/matches_adv.py --season YYYY-YYYY --checkpoint-every 10
```

This merges basic match data from `data/raw/match/` with scraped advanced statistics.

