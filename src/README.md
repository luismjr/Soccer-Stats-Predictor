# Source Code

This folder contains all Python source code for the Soccer Stats Predictor project. The codebase is organized into specialized modules for data collection, machine learning, and visualization.

## 📁 Directory Structure

```
src/
├── scraping/          # Web scraping modules for FBref data
├── models/            # Machine learning pipeline and prediction models
├── tableau/           # Tableau export and visualization utilities
└── README.md         # This file
```

## 🔧 Modules Overview

### 📊 Scraping Module (`scraping/`)
**Purpose:** Automated data collection from FBref.com for Premier League statistics

**Key Features:**
- **Resilient scraping** with retry logic and exponential backoff
- **Checkpointing system** for large data collection tasks
- **Browser-like headers** to avoid detection and blocking
- **Polite delays** (3-8 seconds) between requests
- **Atomic CSV writes** to prevent data corruption

**Scripts:**
- `matches.py` - Basic match data (scores, xG, schedules)
- `matches_adv.py` - Advanced match statistics with checkpointing
- `get_season.py` - Season-level team statistics
- `clean_season.py` - Squad data cleaning and processing

### 🤖 Models Module (`models/`)
**Purpose:** Machine learning pipeline for match outcome prediction

**Key Features:**
- **Feature Engineering:** Venue-specific rolling form, advanced statistics
- **Random Forest Classifier:** 600 trees with balanced class weighting
- **Threshold Optimization:** Custom probability thresholds for better accuracy
- **New Team Handling:** NaN imputation for newly promoted teams
- **YAML Configuration:** Easy experimentation with different parameters

**Core Components:**
- Rolling window features (last 3 home/away games, last 5 overall)
- Advanced match statistics (possession, corners, tackles, etc.)
- Match timing and team encoding
- Probability-based predictions with confidence scores

### 📈 Tableau Module (`tableau/`)
**Purpose:** Advanced data export and visualization utilities for Tableau dashboards

**Key Features:**
- **Comprehensive CSV exports** with calculated metrics
- **Team strength analysis** (attack/defense ratings)
- **Confidence categorization** and risk assessment
- **Feature importance analysis** for model interpretation
- **Time-based analysis** (weekend matches, time categories)

**Outputs:**
- Main predictions dataset with confidence metrics
- Team performance analysis (per-team perspective)
- Feature analysis for model debugging
- Summary statistics for dashboard overview

## 🚀 Quick Start

### Data Collection
```bash
# Basic match data
python src/scraping/matches.py

# Advanced statistics (with checkpointing)
python src/scraping/matches_adv.py --season 2025-2026 --checkpoint-every 10

# Season statistics
python src/scraping/get_season.py --season 2024-2025
```

### Model Training & Prediction
```bash
# Training mode (with validation)
python src/models/match_predictor.py --config ../configs/train_config.yaml

# Production predictions
python src/models/match_predictor.py --config ../configs/predict_config.yaml
```

### Tableau Export
```bash
# Generate advanced Tableau exports
python src/tableau/export_tableau.py
```

## 📋 Data Flow

1. **Data Collection** (`scraping/`) → Raw CSV files in `data/raw/`
2. **Data Processing** → Cleaned data in `data/processed/`
3. **Model Training** (`models/`) → Trained model + predictions
4. **Visualization** (`tableau/`) → Dashboard-ready exports in `data/tableau/`

## 🔧 Dependencies

- **Web Scraping:** `requests`, `beautifulsoup4`, `pandas`
- **Machine Learning:** `scikit-learn`, `numpy`, `pandas`
- **Configuration:** `pyyaml`
- **Time Handling:** `zoneinfo` (Python 3.9+)

## 📊 Model Performance

The Random Forest model achieves:
- **Accuracy:** ~55-60% (significantly better than random 33%)
- **Feature Importance:** Recent form and venue-specific performance
- **Confidence Scoring:** Probability-based predictions with risk assessment

## 🎯 Key Features

- **Automated Pipeline:** End-to-end from data collection to predictions
- **Robust Error Handling:** Graceful handling of missing data and network issues
- **Configurable:** YAML-based configuration for easy experimentation
- **Production Ready:** Checkpointing and atomic operations for reliability
- **Visualization Ready:** Pre-processed exports for Tableau dashboards

