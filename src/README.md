# Source Code

This folder contains all Python source code for the project.

## Subdirectories

- **`scraping/`** - Web scrapers for FBref data (matches, advanced stats, team stats)
- **`models/`** - Machine learning pipeline (feature engineering, training, prediction)

## Key Components

**Scraping:**
- Resilient web scraping with retry logic and checkpointing
- Handles both basic and advanced match statistics

**Models:**
- Random Forest classifier with custom threshold optimization
- Rolling window features for recent form analysis

