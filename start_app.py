#!/usr/bin/env python3
"""
Quick start script for the Premier League Predictor app.
This will update predictions and start the web server.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Start the app with fresh predictions"""
    project_dir = Path(__file__).parent
    os.chdir(project_dir)
    
    print("🚀 Starting Premier League Predictor App...")
    print("=" * 50)
    
    # Check if we have predictions data
    pred_file = project_dir / "data" / "upcoming_preds.csv"
    if not pred_file.exists():
        print("📊 No predictions found. Generating fresh predictions...")
        try:
            subprocess.run([sys.executable, "update_predictions.py"], check=True)
        except subprocess.CalledProcessError:
            print("❌ Failed to generate predictions. Please run 'python update_predictions.py' manually.")
            sys.exit(1)
    else:
        print("✅ Found existing predictions data")
    
    print("\n🌐 Starting web server...")
    print("📱 Visit http://127.0.0.1:3000 to view predictions")
    print("🛑 Press Ctrl+C to stop the server")
    print("=" * 50)
    
    # Start the Flask app
    try:
        subprocess.run([sys.executable, "app.py"], check=True)
    except KeyboardInterrupt:
        print("\n👋 Server stopped. Goodbye!")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

