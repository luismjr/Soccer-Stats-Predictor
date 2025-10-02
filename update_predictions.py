#!/usr/bin/env python3
"""
Script to generate fresh predictions and update the web app data.
Run this to get the latest predictions from your model.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔄 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def main():
    """Generate fresh predictions and update the app"""
    print("🚀 Updating Premier League Predictions...")
    print("=" * 50)
    
    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    # Activate virtual environment and run predictions
    venv_python = project_dir / "venv" / "bin" / "python"
    
    # Command to generate predictions
    cmd = f"""{venv_python} src/models/match_predictor.py \
  --glob "data/processed/matches_*.csv" \
  --train-seasons 2021-2022 2022-2023 2023-2024 2024-2025 \
  --val-season 2025-2026 \
  --test-season 2025-2026 \
  --n-estimators 800 \
  --min-samples-split 8 \
  --min-samples-leaf 3 \
  --max-depth 20 \
  --max-features "sqrt" \
  --save-upcoming data/upcoming_preds.csv"""
    
    if not run_command(cmd, "Generating predictions"):
        print("❌ Failed to generate predictions. Check your data and model.")
        sys.exit(1)
    
    # Check if predictions were created
    pred_file = project_dir / "data" / "upcoming_preds.csv"
    if pred_file.exists():
        print(f"✅ Predictions saved to: {pred_file}")
        
        # Show some stats
        import pandas as pd
        df = pd.read_csv(pred_file)
        print(f"📊 Generated {len(df)} predictions")
        print(f"🏠 Home wins: {len(df[df['pred_class'] == 2])}")
        print(f"🤝 Draws: {len(df[df['pred_class'] == 1])}")
        print(f"✈️ Away wins: {len(df[df['pred_class'] == 0])}")
        
    else:
        print("❌ Predictions file not found after generation")
        sys.exit(1)
    
    print("\n🎉 Predictions updated successfully!")
    print("🌐 Run 'python app.py' to start the web server")
    print("📱 Visit http://127.0.0.1:3000 to view predictions")

if __name__ == "__main__":
    main()

