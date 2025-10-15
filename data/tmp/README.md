# Temporary Checkpoint Files

This folder stores partial scraping results for resume capability.

## Files

- `match_reports_YYYY-YYYY_partial.csv` - Checkpoint file with already-scraped matches

## Purpose

If web scraping is interrupted, the script resumes from the last checkpoint instead of re-scraping all matches from the beginning. This saves time and reduces server load.

## Cleanup

These files can be safely deleted after scraping is complete. They're automatically recreated on the next scraping run.

