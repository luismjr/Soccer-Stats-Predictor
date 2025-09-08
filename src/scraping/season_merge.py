import re
import pandas as pd
from pathlib import Path

def clean_col(c: str) -> str:
    c = str(c).strip()
    # turn "Unnamed: 0_level_0 Squad" -> "Squad"
    c = re.sub(r"^Unnamed:\s*\d+_level_\d+\s*", "", c)
    # unify spaces
    c = re.sub(r"\s+", " ", c)
    # optional: shorter names
    replacements = {
        "Playing Time ": "",
        "Per 90 Minutes ": "Per90 ",
        "Performance ": "",
        "Expected ": "",
        "Progression ": "",
    }
    for k, v in replacements.items():
        if c.startswith(k):
            c = v + c[len(k):]
    # example: rename common fields
    c = c.replace("# Pl", "NumPlayers").replace("MP", "Matches")
    return c

# Ensure processed directory exists
Path("data/processed").mkdir(parents=True, exist_ok=True)

# Loop through 2020–2026 seasons
for end_year in range(2020, 2027):  # includes 2026
    season = f"{end_year-1}-{end_year}"
    in_path = f"data/raw/squad_standard_{season}.csv"
    out_path = f"data/processed/squad_standard_{season}_clean.csv"

    try:
        df = pd.read_csv(in_path)

        # Clean columns
        df.columns = [clean_col(c) for c in df.columns]

        # Ensure numeric columns except Squad/Season
        for col in df.columns:
            if col not in ("Squad", "Season"):
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # If conversion fails, leave column as-is
                    pass

        # Save
        df.to_csv(out_path, index=False)
        print(f"✔ Cleaned {season} → {out_path}")

    except FileNotFoundError:
        print(f"⚠ Skipped {season}: file not found at {in_path}")
