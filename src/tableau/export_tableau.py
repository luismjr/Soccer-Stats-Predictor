#!/usr/bin/env python3
"""
Advanced Tableau Export Script
Creates comprehensive CSV files with detailed analysis and calculated metrics for advanced dashboarding.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def create_advanced_tableau_export():
    """Create advanced CSV exports for Tableau with comprehensive analysis."""
    
    print("🚀 Creating Advanced Tableau Export...")
    
    # Read prediction data
    prediction_file = "data/prediction/prediction_data.csv"
    features_file = "data/prediction/prediction_features.csv"
    
    if not os.path.exists(prediction_file):
        print(f"❌ Prediction file not found: {prediction_file}")
        return
    
    if not os.path.exists(features_file):
        print(f"❌ Features file not found: {features_file}")
        return
    
    # Read the data
    predictions = pd.read_csv(prediction_file)
    features = pd.read_csv(features_file)
    
    print(f"📊 Loaded {len(predictions)} predictions and {len(features)} feature rows")
    
    # Standardize date formats for merging
    predictions['Date'] = pd.to_datetime(predictions['Date']).dt.date
    features['Date'] = pd.to_datetime(features['Date']).dt.date
    
    # Merge the data
    merged_data = pd.merge(predictions, features, on=['Season', 'Date', 'Home', 'Away'], how='inner')
    
    print(f"🔗 Merged data: {len(merged_data)} rows")
    
    # Create comprehensive dataset
    df = merged_data.copy()
    
    # Rename snake_case columns to camelCase
    df = df.rename(columns={
        'pred_class': 'PredClass',
        'pred_label': 'PredLabel',
        'p_home_win': 'PHomeWin',
        'p_draw': 'PDraw',
        'p_home_loss': 'PHomeLoss'
    })
    
    # Basic calculations
    df['Match'] = df['Home'] + ' vs ' + df['Away']
    df['Confidence'] = df[['PHomeWin', 'PDraw', 'PHomeLoss']].max(axis=1)
    df['ConfidencePct'] = (df['Confidence'] * 100).round(1)
    
    # Confidence categories
    def get_confidence_category(conf_pct):
        if conf_pct >= 70:
            return 'Very High (≥70%)'
        elif conf_pct >= 60:
            return 'High (60-70%)'
        elif conf_pct >= 50:
            return 'Medium-High (50-60%)'
        elif conf_pct >= 40:
            return 'Medium (40-50%)'
        else:
            return 'Low (<40%)'
    
    df['ConfidenceCategory'] = df['ConfidencePct'].apply(get_confidence_category)
    
    # Risk assessment
    df['RiskLevel'] = df['ConfidencePct'].apply(lambda x: 'Low Risk' if x >= 60 else 'Medium Risk' if x >= 40 else 'High Risk')
    
    # Match importance (based on confidence spread)
    confidence_spread = df[['PHomeWin', 'PDraw', 'PHomeLoss']].max(axis=1) - df[['PHomeWin', 'PDraw', 'PHomeLoss']].min(axis=1)
    df['PredictionCertainty'] = confidence_spread.round(3)
    df['MatchImportance'] = confidence_spread.apply(lambda x: 'High Stakes' if x >= 0.4 else 'Medium Stakes' if x >= 0.2 else 'Low Stakes')
    
    # Team strength analysis (using available features)
    # Include both 3-game venue-specific (_L3) and 5-game overall (_L5) rolling features
    feature_cols = [
        col for col in df.columns
        if (('_L5' in col or '_L3' in col) and ('xG' in col or 'G' in col or 'ShotsOnTarget' in col))
    ]
    
    if feature_cols:
        # Calculate home team strength
        home_attack_cols = [col for col in feature_cols if 'Home' in col and ('xG' in col or 'G' in col)]
        home_defense_cols = [col for col in feature_cols if 'Home' in col and ('xGA' in col or 'GA' in col)]
        
        if home_attack_cols:
            df['HomeAttackStrength'] = df[home_attack_cols].mean(axis=1).round(2)
        if home_defense_cols:
            df['HomeDefenseStrength'] = df[home_defense_cols].mean(axis=1).round(2)
        
        # Calculate away team strength
        away_attack_cols = [col for col in feature_cols if 'Away' in col and ('xG' in col or 'G' in col)]
        away_defense_cols = [col for col in feature_cols if 'Away' in col and ('xGA' in col or 'GA' in col)]
        
        if away_attack_cols:
            df['AwayAttackStrength'] = df[away_attack_cols].mean(axis=1).round(2)
        if away_defense_cols:
            df['AwayDefenseStrength'] = df[away_defense_cols].mean(axis=1).round(2)
        
        # Net advantage calculation
        if 'HomeAttackStrength' in df.columns and 'AwayDefenseStrength' in df.columns:
            df['HomeAdvantage'] = (df['HomeAttackStrength'] - df['AwayDefenseStrength']).round(2)
        if 'AwayAttackStrength' in df.columns and 'HomeDefenseStrength' in df.columns:
            df['AwayAdvantage'] = (df['AwayAttackStrength'] - df['HomeDefenseStrength']).round(2)
        if 'HomeAdvantage' in df.columns and 'AwayAdvantage' in df.columns:
            df['NetAdvantage'] = (df['HomeAdvantage'] - df['AwayAdvantage']).round(2)
    
    # Time-based analysis
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    df['DayOfWeek'] = pd.to_datetime(df['Date']).dt.day_name()
    df['WeekendMatch'] = df['DayOfWeek'].isin(['Saturday', 'Sunday'])
    
    # Hour analysis if available
    if 'Hour' in df.columns:
        df['TimeCategory'] = df['Hour'].apply(lambda x: 'Evening' if x >= 19 else 'Afternoon' if x >= 14 else 'Morning')
    else:
        df['TimeCategory'] = 'Unknown'
    
    # Prediction accuracy indicators
    df['PredictionDifficulty'] = df['ConfidencePct'].apply(
        lambda x: 'Easy' if x >= 65 else 'Moderate' if x >= 45 else 'Difficult'
    )
    
    # Model expectation analysis
    df['ModelExpectation'] = df.apply(lambda row: 
        'Home Favorite' if row['PHomeWin'] > 0.45 else
        'Away Favorite' if row['PHomeLoss'] > 0.45 else
        'Tight Match', axis=1)
    
    # Create output directory
    output_dir = "data/tableau_advanced"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Main predictions dataset
    main_columns = [
        'Season', 'Date', 'DayOfWeek', 'WeekendMatch', 'TimeCategory',
        'Home', 'Away', 'Match',
        'PredClass', 'PredLabel', 'ModelExpectation',
        'PHomeWin', 'PDraw', 'PHomeLoss',
        'Confidence', 'ConfidencePct', 'ConfidenceCategory',
        'RiskLevel', 'PredictionCertainty', 'MatchImportance', 'PredictionDifficulty'
    ]
    
    # Add team strength columns if they exist
    strength_cols = [col for col in df.columns if 'Strength' in col or 'Advantage' in col]
    main_columns.extend(strength_cols)
    
    main_data = df[main_columns].copy()
    main_file = os.path.join(output_dir, "main_predictions.csv")
    main_data.to_csv(main_file, index=False)
    
    # 2. Team performance analysis
    team_analysis = []
    for _, row in df.iterrows():
        # Home team analysis
        team_analysis.append({
            'Match': row['Match'],
            'Team': row['Home'],
            'TeamLocation': 'Home',
            'Opponent': row['Away'],
            'PredictedOutcome': row['PredLabel'],
            'WinProbability': row['PHomeWin'],
            'ConfidencePct': row['ConfidencePct'],
            'AttackStrength': row.get('HomeAttackStrength', 0),
            'DefenseStrength': row.get('HomeDefenseStrength', 0),
            'Advantage': row.get('HomeAdvantage', 0)
        })
        
        # Away team analysis
        team_analysis.append({
            'Match': row['Match'],
            'Team': row['Away'],
            'TeamLocation': 'Away',
            'Opponent': row['Home'],
            'PredictedOutcome': 'Away Win' if row['PredLabel'] == 'Away Win' else 'Home Win' if row['PredLabel'] == 'Home Win' else 'Draw',
            'WinProbability': row['PHomeLoss'] if row['PredLabel'] == 'Away Win' else row['PHomeWin'] if row['PredLabel'] == 'Home Win' else row['PDraw'],
            'ConfidencePct': row['ConfidencePct'],
            'AttackStrength': row.get('AwayAttackStrength', 0),
            'DefenseStrength': row.get('AwayDefenseStrength', 0),
            'Advantage': row.get('AwayAdvantage', 0)
        })
    
    team_df = pd.DataFrame(team_analysis)
    team_file = os.path.join(output_dir, "team_analysis.csv")
    team_df.to_csv(team_file, index=False)
    
    # 3. Feature importance analysis
    if feature_cols:
        feature_analysis = []
        for _, row in df.iterrows():
            for col in feature_cols:
                if pd.notna(row[col]):
                    feature_analysis.append({
                        'Match': row['Match'],
                        'Feature': col,
                        'FeatureCategory': 'Attack' if 'xG' in col or 'G' in col else 'Defense' if 'xGA' in col or 'GA' in col else 'Other',
                        'Team': 'Home' if 'Home' in col else 'Away',
                        'Value': row[col],
                        'ConfidencePct': row['ConfidencePct'],
                        'PredictedOutcome': row['PredLabel']
                    })
        
        feature_df = pd.DataFrame(feature_analysis)
        feature_file = os.path.join(output_dir, "feature_analysis.csv")
        feature_df.to_csv(feature_file, index=False)
    
    # 4. Summary statistics
    summary_stats = {
        'TotalMatches': len(df),
        'HighConfidenceMatches': len(df[df['ConfidencePct'] >= 60]),
        'MediumConfidenceMatches': len(df[(df['ConfidencePct'] >= 40) & (df['ConfidencePct'] < 60)]),
        'LowConfidenceMatches': len(df[df['ConfidencePct'] < 40]),
        'HomeWinsPredicted': len(df[df['PredLabel'] == 'Home Win']),
        'AwayWinsPredicted': len(df[df['PredLabel'] == 'Away Win']),
        'DrawsPredicted': len(df[df['PredLabel'] == 'Draw']),
        'AverageConfidence': df['ConfidencePct'].mean(),
        'WeekendMatches': len(df[df['WeekendMatch'] == True])
    }
    
    summary_df = pd.DataFrame([summary_stats])
    summary_file = os.path.join(output_dir, "summary_stats.csv")
    summary_df.to_csv(summary_file, index=False)
    
    print(f"✅ Advanced Tableau exports created in: {output_dir}")
    print(f"📁 Files created:")
    print(f"   📊 Main predictions: {main_file}")
    print(f"   👥 Team analysis: {team_file}")
    if feature_cols:
        print(f"   🔍 Feature analysis: {feature_file}")
    print(f"   📈 Summary stats: {summary_file}")
    
    print(f"\n📋 Main dataset preview:")
    print(main_data.head())
    
    return output_dir

if __name__ == "__main__":
    create_advanced_tableau_export()
