from flask import Flask, render_template, send_from_directory, jsonify
import os
import pandas as pd
from datetime import datetime, timezone
import json

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

def load_predictions():
    """Load and process predictions data"""
    try:
        # Try to load from the most recent predictions file
        pred_file = os.path.join(BASE_DIR, "data", "upcoming_preds.csv")
        if os.path.exists(pred_file):
            df = pd.read_csv(pred_file)
        else:
            # Fallback to any predictions file
            pred_files = [f for f in os.listdir(os.path.join(BASE_DIR, "data")) if f.endswith("preds.csv")]
            if pred_files:
                pred_file = os.path.join(BASE_DIR, "data", pred_files[0])
                df = pd.read_csv(pred_file)
            else:
                return None
        
        # Convert date to datetime and filter future matches
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        now = datetime.now(timezone.utc)
        future_matches = df[df['Date'] >= now].copy()
        
        # Sort by date
        future_matches = future_matches.sort_values('Date')
        
        return future_matches
    except Exception as e:
        print(f"Error loading predictions: {e}")
        return None

@app.get("/")
def home():
    """Main page showing predictions"""
    predictions = load_predictions()
    return render_template("index.html", predictions=predictions)

@app.get("/api/predictions")
def api_predictions():
    """API endpoint for predictions data"""
    predictions = load_predictions()
    if predictions is None:
        return jsonify({"error": "No predictions data available"}), 404
    
    # Convert to JSON-serializable format
    data = predictions.to_dict('records')
    return jsonify(data)

@app.get("/data/<path:filename>")
def data_files(filename):
    """Serve data files"""
    return send_from_directory(os.path.join(BASE_DIR, "data"), filename, as_attachment=False)

@app.get("/api/stats")
def api_stats():
    """API endpoint for prediction statistics"""
    predictions = load_predictions()
    if predictions is None:
        return jsonify({"error": "No predictions data available"}), 404
    
    stats = {
        "total_matches": len(predictions),
        "home_wins": len(predictions[predictions['pred_class'] == 2]),
        "draws": len(predictions[predictions['pred_class'] == 1]),
        "away_wins": len(predictions[predictions['pred_class'] == 0]),
        "avg_confidence": predictions[['p_home_win', 'p_draw', 'p_home_loss']].max(axis=1).mean(),
        "next_match": predictions.iloc[0]['Date'].isoformat() if len(predictions) > 0 else None
    }
    
    return jsonify(stats)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=3000, debug=True)
