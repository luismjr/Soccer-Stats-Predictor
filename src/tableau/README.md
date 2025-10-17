# Tableau Export Module

This module provides advanced data export utilities for creating comprehensive Tableau dashboards and visualizations for the Soccer Stats Predictor project.

## 📁 Files

- **`export_tableau.py`** - Main export script for advanced Tableau visualizations

## 🎯 Purpose

The Tableau module transforms raw prediction data and features into dashboard-ready CSV files with calculated metrics, team analysis, and comprehensive statistics for advanced visualization.

## 🚀 Features

### 📊 Comprehensive Data Export
- **Main Predictions Dataset:** Complete match predictions with confidence metrics
- **Team Performance Analysis:** Per-team perspective with strength ratings
- **Feature Analysis:** Detailed feature breakdown for model interpretation
- **Summary Statistics:** High-level overview metrics for dashboards

### 🧮 Advanced Calculations
- **Confidence Categorization:** Automatic grouping by prediction confidence levels
- **Risk Assessment:** Low/Medium/High risk classification based on confidence
- **Team Strength Analysis:** Attack and defense strength calculations
- **Match Importance:** Stake levels based on prediction certainty
- **Time Analysis:** Weekend matches, time categories, day-of-week patterns

### 📈 Dashboard-Ready Metrics
- **Confidence Percentages:** Rounded confidence scores for easy visualization
- **Prediction Difficulty:** Easy/Moderate/Difficult classification
- **Model Expectations:** Home Favorite/Away Favorite/Tight Match categorization
- **Net Advantage:** Calculated team advantages for matchups

## 🔧 Usage

### Basic Export
```bash
# Run from project root
python src/tableau/export_tableau.py
```

### Output Location
All exports are saved to: `data/tableau_advanced/`

## 📋 Output Files

### 1. Main Predictions (`main_predictions.csv`)
**Primary dataset for most visualizations**

**Key Columns:**
- `Match` - Formatted match name (e.g., "Arsenal vs Chelsea")
- `PredLabel` - Predicted outcome (Home Win/Draw/Away Win)
- `ConfidencePct` - Prediction confidence percentage
- `ConfidenceCategory` - Confidence grouping (Very High/High/Medium/Low)
- `RiskLevel` - Risk assessment (Low/Medium/High Risk)
- `ModelExpectation` - Model's expectation (Home Favorite/Away Favorite/Tight Match)
- `MatchImportance` - Stake level (High/Medium/Low Stakes)
- `WeekendMatch` - Boolean for weekend games
- `TimeCategory` - Time grouping (Morning/Afternoon/Evening)

**Team Strength Metrics:**
- `HomeAttackStrength` - Home team attack rating
- `HomeDefenseStrength` - Home team defense rating
- `AwayAttackStrength` - Away team attack rating
- `AwayDefenseStrength` - Away team defense rating
- `NetAdvantage` - Calculated net advantage

### 2. Team Analysis (`team_analysis.csv`)
**Per-team perspective for team-focused dashboards**

**Key Columns:**
- `Team` - Team name
- `TeamLocation` - Home/Away
- `Opponent` - Opposing team
- `PredictedOutcome` - Outcome from team's perspective
- `WinProbability` - Team's win probability
- `AttackStrength` - Team's attack rating
- `DefenseStrength` - Team's defense rating
- `Advantage` - Team's calculated advantage

### 3. Feature Analysis (`feature_analysis.csv`)
**Detailed feature breakdown for model interpretation**

**Key Columns:**
- `Feature` - Feature name
- `FeatureCategory` - Attack/Defense/Other
- `Team` - Home/Away
- `Value` - Feature value
- `ConfidencePct` - Associated prediction confidence

### 4. Summary Statistics (`summary_stats.csv`)
**High-level metrics for dashboard overview**

**Key Metrics:**
- `TotalMatches` - Total number of predictions
- `HighConfidenceMatches` - Matches with ≥60% confidence
- `MediumConfidenceMatches` - Matches with 40-60% confidence
- `LowConfidenceMatches` - Matches with <40% confidence
- `HomeWinsPredicted` - Number of home win predictions
- `AwayWinsPredicted` - Number of away win predictions
- `DrawsPredicted` - Number of draw predictions
- `AverageConfidence` - Mean confidence across all predictions
- `WeekendMatches` - Number of weekend games

## 🎨 Dashboard Use Cases

### 📊 Confidence Analysis Dashboard
- **Confidence Distribution:** Histogram of confidence levels
- **Risk Assessment:** Pie chart of risk levels
- **Confidence vs Accuracy:** Scatter plot analysis

### 👥 Team Performance Dashboard
- **Team Strength Rankings:** Bar charts of attack/defense ratings
- **Team vs Team Analysis:** Head-to-head comparisons
- **Home vs Away Performance:** Venue-specific analysis

### 📈 Prediction Analytics Dashboard
- **Model Expectations:** Distribution of prediction types
- **Match Importance:** Stake level analysis
- **Time-based Patterns:** Weekend vs weekday performance

### 🔍 Feature Importance Dashboard
- **Feature Categories:** Attack vs Defense importance
- **Feature Values:** Distribution of feature values
- **Confidence Correlation:** Feature impact on confidence

## 🔧 Technical Details

### Data Processing
- **Automatic Merging:** Combines prediction data with feature data
- **Date Standardization:** Ensures consistent date formats
- **Column Renaming:** Converts snake_case to camelCase for Tableau
- **Missing Data Handling:** Graceful handling of missing features

### Performance Optimizations
- **Efficient Merging:** Uses pandas inner joins for data combination
- **Memory Management:** Processes data in chunks for large datasets
- **Error Handling:** Robust error handling with informative messages

### Output Formatting
- **Tableau-Friendly:** Optimized column names and data types
- **Consistent Formatting:** Standardized date and numeric formats
- **Comprehensive Coverage:** All necessary fields for advanced visualizations

## 📊 Sample Visualizations

### Confidence Analysis
```
Confidence Distribution → Histogram of ConfidencePct
Risk Assessment → Pie chart of RiskLevel
Confidence Categories → Bar chart of ConfidenceCategory
```

### Team Performance
```
Attack Strength Rankings → Bar chart of AttackStrength by Team
Defense Strength Rankings → Bar chart of DefenseStrength by Team
Net Advantage Analysis → Scatter plot of NetAdvantage vs ConfidencePct
```

### Match Analysis
```
Weekend vs Weekday → Bar chart of WeekendMatch
Time Categories → Pie chart of TimeCategory
Match Importance → Bar chart of MatchImportance
```

## 🎯 Best Practices

1. **Use Main Predictions** for most general visualizations
2. **Use Team Analysis** for team-focused dashboards
3. **Use Feature Analysis** for model interpretation
4. **Use Summary Stats** for dashboard KPIs and overview metrics
5. **Combine datasets** for comprehensive analysis views

## 🔄 Integration

This module integrates seamlessly with:
- **Prediction Pipeline:** Uses output from `models/match_predictor.py`
- **Feature Data:** Combines with feature engineering results
- **Tableau Dashboards:** Direct import into Tableau workbooks
- **Data Pipeline:** Part of the complete data processing workflow
