"""
Cleans season-wide team stats from data/raw/season and exports to data/processed/season
"""

from pathlib import Path
import pandas as pd


def clean_team_stats(raw_path: Path) -> pd.DataFrame:
    """
    Cleans a team stats CSV with multi-level headers.
    
    Params:
    - raw_path (Path): Path to the raw CSV file
    Returns:
    - pd.DataFrame: Cleaned dataframe with flattened columns and Season column
    """
    # Read existing data frame with multi-level header
    df = pd.read_csv(raw_path, header=[0, 1])
    
    # Flatten two level header names: into single level header names
    # If cat (first level header) is Unnamed, then use col (second level header)
    # Otherwise, join cat and col with a space and strip the result
    df.columns = [
        f"{cat} {col}".strip() if not cat.startswith("Unnamed") else col 
        for cat, col in df.columns
    ]
    
    # Remove unnecessary columns
    columns_to_drop = ["Playing Time 90s", "Playing Time Min", "Playing Time Starts"]
    df.drop(columns=[col for col in columns_to_drop if col in df.columns], inplace=True)
    
    # Remove category prefixes from column names
    prefixes_to_remove = ["Performance ", "Progression ", "Playing Time ", "Expected "]
    new_columns = []
    for col in df.columns:
        for prefix in prefixes_to_remove:
            if col.startswith(prefix):
                col = col.replace(prefix, "").strip()
                break
        new_columns.append(col)
    df.columns = new_columns
    
    # Rename common abbreviations
    abbreviations = {
        "Squad": "Team",
        "Age": "Avg_Age",
        "# Pl": "Players_Used",
        "Poss": "Avg_Possesion",
        "MP": "Matches_Played",
        "Gls": "Goals",
        "Ast": "Assists",
        "G+A": "Goals_And_Assists",
        "G-PK": "Non_Penalty_Goals",
        "PK": "Penalty_Goals",
        "PKatt": "Penalty_Attempts",
        "CrdY": "Yellow_Cards",
        "CrdR": "Red_Cards",
        "PrgC": "Progressive_Carries",
        "PrgP": "Progressive_Passes",
        "xG": "Expected_Goals",
        "npxG": "Non_Penalty_Expected_Goals",
        "xAG": "Expected_Goals_And_Assists",
        "npxG+xAG": "Non_Penalty_Expected_Goals_And_Assists",
    }
    df.rename(columns=abbreviations, inplace=True)
    
    # Extract season from filename (e.g., "team_stats_2024-2025.csv" -> "2024-2025")
    season = raw_path.stem.replace("team_stats_", "")

    # Add season column
    df["Season"] = season
    
    # Drop empty rows
    df = df.dropna(how="all").reset_index(drop=True)
    
    # Return the dataframe
    return df


if __name__ == "__main__":
    raw_dir = Path("data/raw/season")
    processed_dir = Path("data/processed/season")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all team_stats CSV files
    for raw_file in sorted(raw_dir.glob("team_stats_*.csv")):
        print(f"Cleaning {raw_file.name}...")
        
        df = clean_team_stats(raw_file)
        output_path = processed_dir / raw_file.name
        df.to_csv(output_path, index=False)
        
        print(f"Saved to {output_path}")
    
    print("Done ✓")

