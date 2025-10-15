import argparse
import glob
import logging
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score
from zoneinfo import ZoneInfo

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

UK_TZ = ZoneInfo("Europe/London")

# Set to WARNING to reduce verbosity - only shows warnings, errors, and print statements
# Change to logging.INFO for detailed debugging
logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")


# =========================
# Feature engineering
# This is where we engineer the features that will be used to train the model.
# =========================
def prepare_basic_features(
    df: pd.DataFrame,
    *,
    drop_unknown_target: bool = True
) -> pd.DataFrame:
    out = df.copy()

    # --- Date ---
    if "Date" not in out.columns:
        raise ValueError("Expected 'Date' column")
    out["Date"] = pd.to_datetime(out["Date"], utc=True, errors="coerce")
    if out["Date"].isna().any():
        logging.warning("Some dates could not be parsed and will be dropped.")
        out = out.dropna(subset=["Date"])

    # --- Time -> minutes since midnight (continuous) + hour bins ---
    if "Time" not in out.columns:
        raise ValueError("Expected 'Time' column like '12:30'")
    parsed_time = pd.to_datetime(out["Time"], format="%H:%M", errors="coerce")

    out["MinutesSinceMidnight"] = parsed_time.dt.hour * 60 + parsed_time.dt.minute
    if out["MinutesSinceMidnight"].isna().any():
        logging.warning("Some times could not be parsed; filling MinutesSinceMidnight with -1.")
        out["MinutesSinceMidnight"] = out["MinutesSinceMidnight"].fillna(-1).astype(int)
    else:
        out["MinutesSinceMidnight"] = out["MinutesSinceMidnight"].astype(int)

    out["Hour"] = parsed_time.dt.hour.fillna(-1).astype(int)
    out["HourCat"] = out["Hour"].astype("category")

    # --- Opponent / Home / Venue ---
    if "Away" not in out.columns:
        raise ValueError("Expected 'Away' column for opponent")
    out["OppCode"] = out["Away"].astype("category").cat.codes

    if "Home" in out.columns:
        out["HomeCode"] = out["Home"].astype("category").cat.codes
    else:
        out["HomeCode"] = -1

    if "Venue" in out.columns:
        out["VenueCode"] = out["Venue"].astype("category").cat.codes
    else:
        out["VenueCode"] = -1

    # --- Day of week ---
    out["DayCode"] = out["Date"].dt.dayofweek  # Mon=0 .. Sun=6

    # rolling recent form
    out = add_home_recent_form(out, window=3)  # Last 3 home games

    # Away team recent form (venue-specific - last 3 away games only)
    out = add_away_recent_form(out, window=3)  # Last 3 away games

    # Overall form for both teams (regardless of venue - all games)
    out = add_overall_recent_form(out, window=5)  # Last 5 games overall

    # --- Target: 2=HomeWin, 1=Draw, 0=HomeLoss ---
    if "HomeGoals" not in out.columns or "AwayGoals" not in out.columns:
        raise ValueError("Expected 'HomeGoals' and 'AwayGoals' columns to create target")

    def _result_row(r):
        if pd.isna(r["HomeGoals"]) or pd.isna(r["AwayGoals"]):
            return pd.NA
        if r["HomeGoals"] > r["AwayGoals"]:
            return 2
        if r["HomeGoals"] < r["AwayGoals"]:
            return 0
        return 1

    out["target"] = out.apply(_result_row, axis=1).astype("Int64")

    if drop_unknown_target:
        out = out.dropna(subset=["target"])

    return out


def add_home_recent_form(out: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    # derive conceded versions if missing
    if "Home_xGA" not in out.columns and "AwayxG" in out.columns:
        out["Home_xGA"] = pd.to_numeric(out["AwayxG"], errors="coerce")

    if "Home_GA" not in out.columns and "AwayGoals" in out.columns:
        out["Home_GA"] = pd.to_numeric(out["AwayGoals"], errors="coerce")

    if "CrossesConceded" not in out.columns and "AwayCrossesMade" in out.columns:
        out["CrossesConceded"] = pd.to_numeric(out["AwayCrossesMade"], errors="coerce")

    if "ShotsOnTargetConceded" not in out.columns and "AwayShotsOnTargetMade" in out.columns:
        out["ShotsOnTargetConceded"] = pd.to_numeric(out["AwayShotsOnTargetMade"], errors="coerce")

    if "LongBallsConceded" not in out.columns and "AwayLongBallsMade" in out.columns:
        out["LongBallsConceded"] = pd.to_numeric(out["AwayLongBallsMade"], errors="coerce")

    if "TouchesConceded" not in out.columns and "AwayTouchesMade" in out.columns:
        out["TouchesConceded"] = pd.to_numeric(out["AwayTouchesMade"], errors="coerce")

    if "TacklesConceded" not in out.columns and "AwayTacklesMade" in out.columns:
        out["TacklesConceded"] = pd.to_numeric(out["AwayTacklesMade"], errors="coerce")

    cols = {
        # for (home made)
        "HomexG": "xG_L5",
        "HomeGoals": "G_L5",
        "HomeShotsOnTargetMade": "ShotsOnTarget_L5",
        "HomeSavesMade": "Saves_L5",
        "HomeTacklesMade": "Tackles_L5",
        "HomeCrossesMade": "Crosses_L5",
        "HomeTouchesMade": "Touches_L5",
        "HomeLongBallsMade": "LongBalls_L5",

        # against (conceded by home)
        "Home_xGA": "xGA_L5",
        "Home_GA": "GA_L5",
        "ShotsOnTargetConceded": "ShotsOnTargetA_L5",
        "TacklesConceded": "TacklesA_L5",
        "CrossesConceded": "CrossesA_L5",
        "TouchesConceded": "TouchesA_L5",
        "LongBallsConceded": "LongBallsA_L5",
    }

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = pd.NA

    # deterministic order for rolling
    out = out.sort_values(["Home", "Date"]).reset_index(drop=True)

    # per-team rolling means of *previous* home games (shift(1) avoids leakage)
    for src, dest in cols.items():
        prev = out.groupby("Home")[src].shift(1)
        rolled = (
            prev
            .groupby(out["Home"])
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[dest] = rolled

    # restore global chronological order
    out = out.sort_values("Date").reset_index(drop=True)
    return out


def add_away_recent_form(out: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Rolling means over the away team's last N *away* matches.
    Creates features like Away_xG_L5, Away_G_L5, Away_GA_L5, etc.
    based ONLY on the away team's recent away game performance.
    """
    if "Away_xGA" not in out.columns and "HomexG" in out.columns:
        out["Away_xGA"] = pd.to_numeric(out["HomexG"], errors="coerce")
    if "Away_GA" not in out.columns and "HomeGoals" in out.columns:
        out["Away_GA"] = pd.to_numeric(out["HomeGoals"], errors="coerce")

    if "A_ShotsOnTargetConceded" not in out.columns and "HomeShotsOnTargetMade" in out.columns:
        out["A_ShotsOnTargetConceded"] = pd.to_numeric(out["HomeShotsOnTargetMade"], errors="coerce")
    if "A_TacklesConceded" not in out.columns and "HomeTacklesMade" in out.columns:
        out["A_TacklesConceded"] = pd.to_numeric(out["HomeTacklesMade"], errors="coerce")
    if "A_CrossesConceded" not in out.columns and "HomeCrossesMade" in out.columns:
        out["A_CrossesConceded"] = pd.to_numeric(out["HomeCrossesMade"], errors="coerce")
    if "A_TouchesConceded" not in out.columns and "HomeTouchesMade" in out.columns:
        out["A_TouchesConceded"] = pd.to_numeric(out["HomeTouchesMade"], errors="coerce")
    if "A_LongBallsConceded" not in out.columns and "HomeLongBallsMade" in out.columns:
        out["A_LongBallsConceded"] = pd.to_numeric(out["HomeLongBallsMade"], errors="coerce")

    cols = {
        "AwayxG": "Away_xG_L5",
        "AwayGoals": "Away_G_L5",
        "AwayShotsOnTargetMade": "Away_ShotsOnTarget_L5",
        "AwaySavesMade": "Away_Saves_L5",
        "AwayTacklesMade": "Away_Tackles_L5",
        "AwayCrossesMade": "Away_Crosses_L5",
        "AwayTouchesMade": "Away_Touches_L5",
        "AwayLongBallsMade": "Away_LongBalls_L5",
        "Away_xGA": "Away_xGA_L5",
        "Away_GA": "Away_GA_L5",
        "A_ShotsOnTargetConceded": "Away_ShotsOnTargetA_L5",
        "A_TacklesConceded": "Away_TacklesA_L5",
        "A_CrossesConceded": "Away_CrossesA_L5",
        "A_TouchesConceded": "Away_TouchesA_L5",
        "A_LongBallsConceded": "Away_LongBallsA_L5",
    }

    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        else:
            out[c] = pd.NA

    out = out.sort_values(["Away", "Date"]).reset_index(drop=True)
    for src, dest in cols.items():
        prev = out.groupby("Away")[src].shift(1)
        rolled = (
            prev
            .groupby(out["Away"])
            .rolling(window=window, min_periods=1)
            .mean()
            .reset_index(level=0, drop=True)
        )
        out[dest] = rolled

    out = out.sort_values("Date").reset_index(drop=True)
    return out


def add_overall_recent_form(out: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Rolling means over each team's last N matches regardless of venue (home or away).
    Produces:
      - home_overall_*_l5
      - away_overall_*_l5
    """
    out = out.copy()

    # Ensure conceded columns exist for mapping (fallbacks if not present)
    if "Home_xGA" not in out.columns and "AwayxG" in out.columns:
        out["Home_xGA"] = pd.to_numeric(out["AwayxG"], errors="coerce")
    if "Away_xGA" not in out.columns and "HomexG" in out.columns:
        out["Away_xGA"] = pd.to_numeric(out["HomexG"], errors="coerce")

    if "Home_GA" not in out.columns and "AwayGoals" in out.columns:
        out["Home_GA"] = pd.to_numeric(out["AwayGoals"], errors="coerce")
    if "Away_GA" not in out.columns and "HomeGoals" in out.columns:
        out["Away_GA"] = pd.to_numeric(out["HomeGoals"], errors="coerce")

    # Build a "long" per-team table (one row per team per match)
    home_part = pd.DataFrame({
        "team": out["Home"].astype(str),
        "Date": out["Date"],
        "xG": pd.to_numeric(out.get("HomexG"), errors="coerce"),
        "G": pd.to_numeric(out.get("HomeGoals"), errors="coerce"),
        "ShotsOnTarget": pd.to_numeric(out.get("HomeShotsOnTargetMade"), errors="coerce"),
        "Saves": pd.to_numeric(out.get("HomeSavesMade"), errors="coerce"),
        "Tackles": pd.to_numeric(out.get("HomeTacklesMade"), errors="coerce"),
        "Crosses": pd.to_numeric(out.get("HomeCrossesMade"), errors="coerce"),
        "Touches": pd.to_numeric(out.get("HomeTouchesMade"), errors="coerce"),
        "LongBalls": pd.to_numeric(out.get("HomeLongBallsMade"), errors="coerce"),

        "xGA": pd.to_numeric(out.get("Home_xGA"), errors="coerce"),
        "GA": pd.to_numeric(out.get("Home_GA"), errors="coerce"),
        "ShotsOnTargetA": pd.to_numeric(out.get("AwayShotsOnTargetMade"), errors="coerce"),
        "TacklesA": pd.to_numeric(out.get("AwayTacklesMade"), errors="coerce"),
        "CrossesA": pd.to_numeric(out.get("AwayCrossesMade"), errors="coerce"),
        "TouchesA": pd.to_numeric(out.get("AwayTouchesMade"), errors="coerce"),
        "LongBallsA": pd.to_numeric(out.get("AwayLongBallsMade"), errors="coerce"),
    })

    away_part = pd.DataFrame({
        "team": out["Away"].astype(str),
        "Date": out["Date"],
        "xG": pd.to_numeric(out.get("AwayxG"), errors="coerce"),
        "G": pd.to_numeric(out.get("AwayGoals"), errors="coerce"),
        "ShotsOnTarget": pd.to_numeric(out.get("AwayShotsOnTargetMade"), errors="coerce"),
        "Saves": pd.to_numeric(out.get("AwaySavesMade"), errors="coerce"),
        "Tackles": pd.to_numeric(out.get("AwayTacklesMade"), errors="coerce"),
        "Crosses": pd.to_numeric(out.get("AwayCrossesMade"), errors="coerce"),
        "Touches": pd.to_numeric(out.get("AwayTouchesMade"), errors="coerce"),
        "LongBalls": pd.to_numeric(out.get("AwayLongBallsMade"), errors="coerce"),

        "xGA": pd.to_numeric(out.get("Away_xGA"), errors="coerce"),
        "GA": pd.to_numeric(out.get("Away_GA"), errors="coerce"),
        "ShotsOnTargetA": pd.to_numeric(out.get("HomeShotsOnTargetMade"), errors="coerce"),
        "TacklesA": pd.to_numeric(out.get("HomeTacklesMade"), errors="coerce"),
        "CrossesA": pd.to_numeric(out.get("HomeCrossesMade"), errors="coerce"),
        "TouchesA": pd.to_numeric(out.get("HomeTouchesMade"), errors="coerce"),
        "LongBallsA": pd.to_numeric(out.get("HomeLongBallsMade"), errors="coerce"),
    })

    long_df = pd.concat([home_part, away_part], ignore_index=True)
    long_df = long_df.sort_values(["team", "Date"]).reset_index(drop=True)

    base_feats = ["xG","G","ShotsOnTarget","Saves","Tackles","Crosses","Touches","LongBalls",
                  "xGA","GA","ShotsOnTargetA","TacklesA","CrossesA","TouchesA","LongBallsA"]

    # Rolling means over previous matches (shift(1) prevents leakage)
    rolled = {}
    for feat in base_feats:
        prev = long_df.groupby("team")[feat].shift(1)
        rolled_feat = (
            prev.groupby(long_df["team"])
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
        )
        rolled[f"{feat}_overall_l5"] = rolled_feat

    long_roll = pd.concat([long_df[["team","Date"]], pd.DataFrame(rolled)], axis=1)

    # Merge back for HOME team rows
    home_cols = [c for c in long_roll.columns if c.endswith("_overall_l5")]
    home_merge = long_roll.rename(columns={c: f"HomeOverall_{c[:-11]}_L5" for c in home_cols})
    out = out.merge(
        home_merge.rename(columns={"team": "Home", "Date": "Date"}),
        on=["Home", "Date"], how="left"
    )

    # Merge back for AWAY team rows
    away_merge = long_roll.rename(columns={c: f"AwayOverall_{c[:-11]}_L5" for c in home_cols})
    out = out.merge(
        away_merge.rename(columns={"team": "Away", "Date": "Date"}),
        on=["Away", "Date"], how="left"
    )

    return out


# =========================
# NEW: Build UTC kickoff from Date + Time (UK) for upcoming
# =========================
def combine_date_time_to_utc(df: pd.DataFrame) -> pd.Series:
    """
    Build a UTC kickoff timestamp from the match-day Date (tz-aware UTC)
    and the local UK Time column (HH:MM). Handles DST correctly.
    """
    # normalize match-day to date string in UTC (YYYY-MM-DD)
    date_str = df["Date"].dt.tz_convert("UTC").dt.strftime("%Y-%m-%d")

    # extract HH:MM from Time, fallback to 00:00
    time_str = (
        pd.Series(df.get("Time", "00:00"), index=df.index)
          .astype(str)
          .str.extract(r"^(\d{1,2}:\d{2})")[0]
          .fillna("00:00")
    )

    # create local UK datetime, then convert to UTC
    dt_local = pd.to_datetime(date_str + " " + time_str, errors="coerce")
    dt_local = dt_local.dt.tz_localize(UK_TZ, ambiguous="infer", nonexistent="shift_forward")
    return dt_local.dt.tz_convert("UTC")


# =========================
# Data loading / combining
# =========================
def load_multiple_csvs(files: Optional[List[str]] = None, glob_pattern: Optional[str] = None) -> pd.DataFrame:
    if files:
        paths = files
    elif glob_pattern:
        paths = sorted(glob.glob(glob_pattern))
    else:
        raise ValueError("Provide either --files or --glob to load data.")

    if not paths:
        raise FileNotFoundError("No CSV files matched the given --files/--glob.")

    logging.info(f"Loading {len(paths)} file(s):")
    for p in paths:
        logging.info(f"  - {p}")

    dfs = [pd.read_csv(p) for p in paths]
    df_all = pd.concat(dfs, ignore_index=True)
    logging.info(f"Combined rows (pre-clean): {len(df_all):,}")

    # Ensure Season exists
    if "Season" not in df_all.columns:
        raise ValueError("Expected a 'Season' column (e.g., '2024-2025').")

    # Parse/sort dates; drop bad dates
    df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce", utc=True)
    before_drop = len(df_all)
    df_all = df_all.dropna(subset=["Date"])
    if len(df_all) < before_drop:
        logging.warning(f"Dropped {before_drop - len(df_all)} rows with unparseable Date.")
    df_all = df_all.sort_values("Date").reset_index(drop=True)

    # Optional dedupe (common key)
    if {"Date", "Home", "Away"}.issubset(df_all.columns):
        d0 = len(df_all)
        df_all = df_all.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
        if len(df_all) < d0:
            logging.info(f"De-duplicated {d0 - len(df_all)} overlapping rows.")

    logging.info(f"Final combined rows: {len(df_all):,}")
    return df_all


# =========================
# Season splits
# =========================
def season_split(
    df_all: pd.DataFrame,
    train_seasons: List[str],
    val_season: str,
    test_season: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Known outcomes only for model training/evaluation
    feat_all = prepare_basic_features(df_all, drop_unknown_target=True)

    train_df = feat_all[feat_all["Season"].isin(train_seasons)].sort_values("Date")
    val_df   = feat_all[feat_all["Season"] == val_season].sort_values("Date")
    test_df  = feat_all[feat_all["Season"] == test_season].sort_values("Date")

    logging.info(
        f"Split counts | Train: {len(train_df):,}  Val: {len(val_df):,}  Test (played): {len(test_df):,}"
    )
    return train_df, val_df, test_df


# =========================
# Training / evaluation
# =========================
PREDICTORS = [
    # timing / ids
    "MinutesSinceMidnight", "Hour",
    "OppCode", "HomeCode", "VenueCode", "DayCode",

    # recent HOME form (for)
    "xG_L5", "G_L5", "ShotsOnTarget_L5", "Saves_L5",
    "Tackles_L5", "Crosses_L5", "Touches_L5", "LongBalls_L5",

    # recent HOME form (against)
    "xGA_L5", "GA_L5", "ShotsOnTargetA_L5", "TacklesA_L5",
    "CrossesA_L5", "TouchesA_L5", "LongBallsA_L5",

    # overall recent form (home team)
    "HomeOverall_xG_L5", "HomeOverall_G_L5", "HomeOverall_ShotsOnTarget_L5",
    "HomeOverall_Saves_L5", "HomeOverall_Tackles_L5", "HomeOverall_Crosses_L5",
    "HomeOverall_Touches_L5", "HomeOverall_LongBalls_L5",
    "HomeOverall_xGA_L5", "HomeOverall_GA_L5", "HomeOverall_ShotsOnTargetA_L5",
    "HomeOverall_TacklesA_L5", "HomeOverall_CrossesA_L5",
    "HomeOverall_TouchesA_L5", "HomeOverall_LongBallsA_L5",

    # overall recent form (away team)
    "AwayOverall_xG_L5", "AwayOverall_G_L5", "AwayOverall_ShotsOnTarget_L5",
    "AwayOverall_Saves_L5", "AwayOverall_Tackles_L5", "AwayOverall_Crosses_L5",
    "AwayOverall_Touches_L5", "AwayOverall_LongBalls_L5",
    "AwayOverall_xGA_L5", "AwayOverall_GA_L5", "AwayOverall_ShotsOnTargetA_L5",
    "AwayOverall_TacklesA_L5", "AwayOverall_CrossesA_L5",
    "AwayOverall_TouchesA_L5", "AwayOverall_LongBallsA_L5",

    # venue-specific away form
    "Away_xG_L5", "Away_G_L5", "Away_ShotsOnTarget_L5", "Away_Saves_L5",
    "Away_Tackles_L5", "Away_Crosses_L5", "Away_Touches_L5", "Away_LongBalls_L5",
    "Away_xGA_L5", "Away_GA_L5", "Away_ShotsOnTargetA_L5", "Away_TacklesA_L5",
    "Away_CrossesA_L5", "Away_TouchesA_L5", "Away_LongBallsA_L5",
]

LABEL_MAP = {0: "Away Win", 1: "Draw", 2: "Home Win"}


# -------- Fast thresholding helpers (precompute proba once) --------
def predict_with_thresholds_from_proba(proba: np.ndarray, labels, thresholds=None, default=None):
    """labels: list(model.classes_). thresholds: {class_label: min_prob}."""
    cls_idx = {int(c): i for i, c in enumerate(labels)}
    t = np.zeros(proba.shape[1], dtype=float)
    if thresholds:
        for c, th in thresholds.items():
            t[cls_idx[int(c)]] = float(th)
    preds = []
    for row in proba:
        meets = [j for j in range(len(row)) if row[j] >= t[j]]
        if meets:
            j_best = max(meets, key=lambda j: row[j])
            preds.append(int(labels[j_best]))
        else:
            preds.append(int(default) if default is not None else int(labels[int(row.argmax())]))
    return np.array(preds, dtype=int)


def search_thresholds_for_precision_fast(proba: np.ndarray, y_val: np.ndarray, labels, default=None):
    """
    Coarse-to-fine grid search on precomputed probabilities.
    Returns {'score': best_macro_precision, 'thr': {class: threshold}}
    """
    # coarse grid (fewer combos)
    coarse = np.linspace(0.33, 0.75, 9)  # 9^3 = 729 combos
    best = {"score": -1.0, "thr": {int(c): 0.0 for c in labels}}
    for t0 in coarse:
        for t1 in coarse:
            for t2 in coarse:
                thr = {int(labels[0]): float(t0), int(labels[1]): float(t1), int(labels[2]): float(t2)}
                y_hat = predict_with_thresholds_from_proba(proba, labels, thresholds=thr, default=default)
                sc = precision_score(y_val, y_hat, average="macro", zero_division=0)
                if sc > best["score"]:
                    best = {"score": sc, "thr": thr}

    # fine search around coarse best
    fine_center = np.array([best["thr"][int(labels[i])] for i in range(3)])
    fine = [np.clip(np.round(np.linspace(c - 0.08, c + 0.08, 9), 3), 0.0, 1.0) for c in fine_center]
    for t0 in fine[0]:
        for t1 in fine[1]:
            for t2 in fine[2]:
                thr = {int(labels[0]): float(t0), int(labels[1]): float(t1), int(labels[2]): float(t2)}
                y_hat = predict_with_thresholds_from_proba(proba, labels, thresholds=thr, default=default)
                sc = precision_score(y_val, y_hat, average="macro", zero_division=0)
                if sc > best["score"]:
                    best = {"score": sc, "thr": thr}
    return best


# =========================
# Fit / validate
# =========================
def fit_and_validate(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_estimators: int = 600,
    min_samples_split: int = 10,
    min_samples_leaf: int = 2,
    random_state: int = 42
) -> Tuple[RandomForestClassifier, pd.DataFrame, pd.Series]:
    # sanity: all predictors present
    missing = [c for c in PREDICTORS if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing predictor columns in TRAIN: {missing}")

    X_train, y_train = train_df[PREDICTORS], train_df["target"].astype(int)
    X_val, y_val = val_df[PREDICTORS], val_df["target"].astype(int)

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    val_pred = rf.predict(X_val)
    acc = accuracy_score(y_val, val_pred)
    print(f"\nValidation ({val_df['Season'].iloc[0] if len(val_df) else 'N/A'}) Accuracy: {acc:.3f}")
    print("\nValidation classification report:\n",
          classification_report(y_val, val_pred, digits=3, zero_division=0))

    return rf, X_val, y_val


def refit_on_train_plus_val(rf: RandomForestClassifier, train_df: pd.DataFrame, val_df: pd.DataFrame) -> RandomForestClassifier:
    X_tv = pd.concat([train_df[PREDICTORS], val_df[PREDICTORS]])
    y_tv = pd.concat([train_df["target"].astype(int), val_df["target"].astype(int)])
    rf.fit(X_tv, y_tv)
    return rf


def evaluate_on_test(rf: RandomForestClassifier, test_df: pd.DataFrame, thresholds=None, default=None):
    if len(test_df) == 0:
        print("\nNo played matches found in test season to evaluate.")
        return
    X_test, y_test = test_df[PREDICTORS], test_df["target"].astype(int)

    # precompute once
    proba = rf.predict_proba(X_test)
    if thresholds is None:
        test_pred = np.asarray([int(rf.classes_[int(p.argmax())]) for p in proba], dtype=int)
    else:
        test_pred = predict_with_thresholds_from_proba(proba, rf.classes_, thresholds=thresholds, default=default)

    acc = accuracy_score(y_test, test_pred)
    print(f"\nTest ({test_df['Season'].iloc[0]}) Accuracy: {acc:.3f}")
    print("\nTest classification report:\n",
          classification_report(y_test, test_pred, digits=3, zero_division=0))

def refit_on_all_played(rf: RandomForestClassifier, *dfs: pd.DataFrame) -> RandomForestClassifier:
    """Concatenate any number of feature DataFrames that contain 'target', and refit."""
    frames = [df for df in dfs if len(df)]
    if not frames:
        return rf
    combo = pd.concat(frames, ignore_index=True)
    X = combo[PREDICTORS]
    y = combo["target"].astype(int)
    rf.fit(X, y)
    return rf



def plot_importances(rf, save_path="feature_importances.png", top_n=5):
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-GUI backend
        import matplotlib.pyplot as plt
        import pandas as pd
    except ModuleNotFoundError:
        print("matplotlib not installed; skipping feature-importance plot.")
        return

    importances = pd.Series(rf.feature_importances_, index=PREDICTORS)
    # Show only top N most important features
    top_features = importances.nlargest(top_n)
    ax = top_features.sort_values().plot(kind="barh", figsize=(8, 4))
    ax.set_title(f"Top {top_n} Most Important Features")
    ax.set_xlabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved top {top_n} feature importances → {save_path}")


# =========================
# Upcoming: predictions + feature table (UPDATED)
# =========================
def _build_upcoming_feature_table(upcoming_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Create a readable table with IDs + all _l5 columns (and any predictors),
    representing exactly what the model had available for the upcoming games.
    """
    base = upcoming_feat.copy()
    id_cols = ["Season", "Date", "Home", "Away"]
    for col in id_cols:
        if col not in base.columns:
            base[col] = pd.NA

    # ensure all predictors exist
    for c in PREDICTORS:
        if c not in base.columns:
            base[c] = pd.NA

    l5_cols = [c for c in base.columns if c.endswith("_l5")]
    feat_cols = sorted(set(PREDICTORS + l5_cols))
    return base[id_cols + feat_cols]


def predict_upcoming_with_features(
    rf: RandomForestClassifier,
    df_all: pd.DataFrame,
    test_season: str,
    thresholds=None,
    default=None,
    dump_features_path: Optional[str] = None,
):
    """
    Build features on the FULL dataset (played + unplayed), then select upcoming rows.
    This ensures rolling _l5 features for upcoming matches are computed from history.
    Also returns a human-readable feature table with all *_l5 columns.
    """
    # Build features for ALL rows; keep unknown targets so upcoming stays in.
    feat_all = prepare_basic_features(df_all, drop_unknown_target=False)

    # Upcoming = rows in test season with unknown outcomes
    mask_upcoming = (
        (feat_all["Season"] == test_season)
        & (feat_all["HomeGoals"].isna() | feat_all["AwayGoals"].isna())
    )
    upcoming_feat = feat_all[mask_upcoming].copy()
    if upcoming_feat.empty:
        logging.info("No upcoming fixtures (unknown outcomes) found in test season.")
        return pd.DataFrame(), pd.DataFrame()

    # >>> NEW: compute true kickoff in UTC from Date+Time (UK)
    upcoming_feat["Kickoff_UTC"] = combine_date_time_to_utc(upcoming_feat)

    # >>> LIMIT TO NEXT 10 GAMES: Sort by kickoff time and take first 10
    upcoming_feat = upcoming_feat.sort_values("Kickoff_UTC").head(10)
    if upcoming_feat.empty:
        logging.info("No upcoming fixtures after filtering to next 10.")
        return pd.DataFrame(), pd.DataFrame()

    # Features used for prediction
    Xu = upcoming_feat[PREDICTORS]

    # Optional: dump the exact feature rows we used (nice for debugging/calibration)
    if dump_features_path is not None:
        dump_df = pd.concat(
            [upcoming_feat[["Season", "Date", "Home", "Away"]].reset_index(drop=True),
             Xu.reset_index(drop=True)],
            axis=1
        ).sort_values(["Date", "Home", "Away"])
        dump_df.to_csv(dump_features_path, index=False)
        print(f"Saved upcoming feature rows → {dump_features_path}")

    # Probabilities + predictions
    proba = rf.predict_proba(Xu)
    if thresholds is None:
        pred_cls = np.asarray([int(rf.classes_[int(p.argmax())]) for p in proba], dtype=int)
    else:
        pred_cls = predict_with_thresholds_from_proba(proba, rf.classes_, thresholds=thresholds, default=default)

    # Build outgoing predictions with ISO UTC kickoff in Date
    preds = upcoming_feat[["Season", "Home", "Away"]].copy()
    preds["Date"] = upcoming_feat["Kickoff_UTC"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # overwrite with real kickoff UTC
    preds["pred_class"] = pred_cls
    preds["pred_label"] = preds["pred_class"].map(LABEL_MAP)

    # Probability columns aligned to rf.classes_
    cls_to_name = {int(c): name for c, name in zip(rf.classes_, ["p_home_loss", "p_draw", "p_home_win"])}
    for i, cls in enumerate(rf.classes_):
        preds[cls_to_name[int(cls)]] = proba[:, i]

    # Build a readable feature table (IDs + *_l5 columns + predictors)
    feat_tbl = _build_upcoming_feature_table(upcoming_feat).copy()
    feat_tbl["pred_class"] = pred_cls
    feat_tbl["pred_label"] = preds["pred_label"].values
    feat_tbl["p_home_loss"] = preds["p_home_loss"].values
    feat_tbl["p_draw"]      = preds["p_draw"].values
    feat_tbl["p_home_win"]  = preds["p_home_win"].values

    return preds.sort_values("Date").reset_index(drop=True), feat_tbl.sort_values("Date").reset_index(drop=True)


# Backward-compatible wrapper
def predict_upcoming_in_test_season(
    rf: RandomForestClassifier,
    df_all: pd.DataFrame,
    test_season: str,
    thresholds=None,
    default=None,
    dump_features_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Build features on the FULL dataset (played + unplayed), then select upcoming rows.
    This ensures rolling _l5 features for upcoming matches are computed from history.
    """
    # Build features for ALL rows; keep unknown targets so upcoming stays in.
    feat_all = prepare_basic_features(df_all, drop_unknown_target=False)

    # Upcoming = rows in test season with unknown outcomes
    mask_upcoming = (
        (feat_all["Season"] == test_season)
        & (feat_all["HomeGoals"].isna() | feat_all["AwayGoals"].isna())
    )
    upcoming_feat = feat_all[mask_upcoming].copy()
    if upcoming_feat.empty:
        logging.info("No upcoming fixtures (unknown outcomes) found in test season.")
        return pd.DataFrame()

    # >>> NEW: compute true kickoff in UTC from Date+Time (UK)
    upcoming_feat["Kickoff_UTC"] = combine_date_time_to_utc(upcoming_feat)

    # >>> LIMIT TO NEXT 10 GAMES: Sort by kickoff time and take first 10
    upcoming_feat = upcoming_feat.sort_values("Kickoff_UTC").head(10)
    if upcoming_feat.empty:
        logging.info("No upcoming fixtures after filtering to next 10.")
        return pd.DataFrame()

    # The features used for prediction
    Xu = upcoming_feat[PREDICTORS]

    # Optional: dump the exact feature rows we used (nice for debugging/calibration)
    if dump_features_path is not None:
        dump_df = pd.concat(
            [upcoming_feat[["Season", "Date", "Home", "Away"]].reset_index(drop=True),
             Xu.reset_index(drop=True)],
            axis=1
        )
        dump_df.sort_values(["Date", "Home", "Away"]).to_csv(dump_features_path, index=False)
        print(f"Saved upcoming feature rows → {dump_features_path}")

    # Probabilities + class predictions (with optional thresholds)
    proba = rf.predict_proba(Xu)
    if thresholds is None:
        pred_cls = np.asarray([int(rf.classes_[int(p.argmax())]) for p in proba], dtype=int)
    else:
        pred_cls = predict_with_thresholds_from_proba(proba, rf.classes_, thresholds=thresholds, default=default)

    # Outgoing CSV with real kickoff in Date
    out = upcoming_feat[["Season", "Home", "Away"]].copy()
    out["Date"] = upcoming_feat["Kickoff_UTC"].dt.strftime("%Y-%m-%dT%H:%M:%SZ")  # overwrite with kickoff UTC
    out["pred_class"] = pred_cls
    out["pred_label"] = out["pred_class"].map(LABEL_MAP)

    # Probability columns aligned to rf.classes_
    cls_to_name = {int(c): name for c, name in zip(rf.classes_, ["p_home_loss", "p_draw", "p_home_win"])}
    for i, cls in enumerate(rf.classes_):
        out[cls_to_name[int(cls)]] = proba[:, i]

    return out.sort_values("Date").reset_index(drop=True)


# =========================
# Config file support
# =========================
def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    if not HAS_YAML:
        raise ImportError("pyyaml not installed. Install with: pip install pyyaml")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info(f"Loaded configuration from {config_path}")
    return config


def merge_config_with_args(config: Optional[Dict[str, Any]], args: argparse.Namespace) -> argparse.Namespace:
    """
    Merge config file with CLI arguments. CLI arguments take precedence.
    If an argument is None (not provided via CLI), use config value.
    """
    if config is None:
        return args
    
    # Data sources
    if args.glob is None and 'data' in config and 'glob' in config['data']:
        args.glob = config['data']['glob']
    if args.files is None and 'data' in config and 'files' in config['data']:
        args.files = config['data']['files']
    
    # Seasons
    if 'seasons' in config:
        if args.train_seasons is None and 'train' in config['seasons']:
            args.train_seasons = config['seasons']['train']
        if args.val_season is None and 'val' in config['seasons']:
            args.val_season = config['seasons']['val']
        if args.test_season is None and 'test' in config['seasons']:
            args.test_season = config['seasons']['test']
    
    # Model hyperparameters (only if not explicitly set via CLI)
    if 'model' in config:
        if args.n_estimators == 600 and 'n_estimators' in config['model']:  # default value
            args.n_estimators = config['model']['n_estimators']
        if args.min_samples_split == 14 and 'min_samples_split' in config['model']:
            args.min_samples_split = config['model']['min_samples_split']
        if args.min_samples_leaf == 3 and 'min_samples_leaf' in config['model']:
            args.min_samples_leaf = config['model']['min_samples_leaf']
    
    # Tuning
    if 'tuning' in config:
        # Check for explicit no_thresholds setting first
        if not args.no_thresholds and 'no_thresholds' in config['tuning']:
            args.no_thresholds = config['tuning']['no_thresholds']
        # Otherwise, fall back to use_thresholds (inverse logic)
        elif not args.no_thresholds and 'use_thresholds' in config['tuning']:
            args.no_thresholds = not config['tuning']['use_thresholds']
        if args.fallback_class is None and 'fallback_class' in config['tuning']:
            args.fallback_class = config['tuning']['fallback_class']
    
    # Output
    if 'output' in config:
        if args.save_upcoming is None and 'predictions' in config['output']:
            args.save_upcoming = config['output']['predictions']
        if args.dump_upcoming_features is None and 'features' in config['output']:
            args.dump_upcoming_features = config['output']['features']
        if not args.no_plot and 'feature_importance_plot' in config['output']:
            args.no_plot = not config['output']['feature_importance_plot']
    
    return args


# =========================
# CLI
# =========================
def main():
    parser = argparse.ArgumentParser(
        description="Train/Validate/Test Random Forest with season-based splits.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Using config file:
        python src/models/match_predictor.py --config configs/train_config.yaml
  
        # Using config with CLI overrides:
        python src/models/match_predictor.py --config configs/train_config.yaml --test-season 2026-2027
  
        # Using CLI only:
        python src/models/match_predictor.py --glob "data/processed/*.csv" --train-seasons 2021-2022 2022-2023 --val-season 2024-2025 --test-season 2025-2026
                """
    )
    
    # Config file option
    parser.add_argument("--config", type=str, help="Path to YAML config file. CLI args override config values.")
    
    # Data sources
    parser.add_argument("--glob", type=str, help="Glob for CSVs, e.g. 'data/processed/matches_*.csv'")
    parser.add_argument("--files", nargs="*", help="Explicit CSV paths")

    parser.add_argument("--train-seasons", nargs="+", default=None,
                        help="Seasons to TRAIN on, e.g. 2021-2022 2022-2023 2023-2024")
    parser.add_argument("--val-season", default=None, help="Season to VALIDATE on, e.g. 2024-2025")
    parser.add_argument("--test-season", default=None, help="Season to TEST on, e.g. 2025-2026")

    parser.add_argument("--n-estimators", type=int, default=600)
    parser.add_argument("--min-samples-split", type=int, default=14)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--save-upcoming", type=str, default=None,
                        help="If set, save upcoming fixture predictions (CSV) for the test season to this path.")
    parser.add_argument("--dump-upcoming-features", type=str, default=None,
                        help="Optional path to save the *feature rows* used for upcoming predictions.")
    parser.add_argument("--fallback-class", type=int, default=None,
                        help="Optional class to fall back to when no threshold is met; e.g., 2 for Home Win.")
    parser.add_argument("--no-thresholds", action="store_true",
                        help="Disable learned thresholds and use plain argmax (fallback ignored).")

    args = parser.parse_args()
    
    # Load config and merge with CLI args
    config = None
    if args.config:
        if not HAS_YAML:
            logging.error("Config file specified but pyyaml not installed. Install with: pip install pyyaml")
            return
        config = load_config(args.config)
        args = merge_config_with_args(config, args)
    
    # Validate required arguments
    if args.train_seasons is None:
        parser.error("--train-seasons is required (or provide via config file)")
    if args.val_season is None:
        parser.error("--val-season is required (or provide via config file)")
    if args.test_season is None:
        parser.error("--test-season is required (or provide via config file)")

    df_all = load_multiple_csvs(files=args.files, glob_pattern=args.glob)

    # Create splits
    train_df, val_df, test_df = season_split(
        df_all,
        train_seasons=args.train_seasons,
        val_season=args.val_season,
        test_season=args.test_season,
    )

    # Fit on train, validate
    rf, X_val, y_val = fit_and_validate(
        train_df, val_df,
        n_estimators=args.n_estimators,
        min_samples_split=args.min_samples_split,
        min_samples_leaf=args.min_samples_leaf,
    )

    # ---- Threshold handling (NEW) ----
    if args.no_thresholds:
        print("Thresholds disabled: using plain argmax (no fallback class).")
        thresholds_to_use = None
        default_to_use = None
    else:
        # Learn probability thresholds that maximize macro precision on VAL (FAST)
        val_proba = rf.predict_proba(X_val)  # compute ONCE
        labels = list(rf.classes_)
        best = search_thresholds_for_precision_fast(val_proba, y_val.to_numpy(), labels, default=args.fallback_class)
        y_val_thresh = predict_with_thresholds_from_proba(val_proba, labels, thresholds=best["thr"], default=args.fallback_class)
        print("Best macro precision on VAL:", precision_score(y_val, y_val_thresh, average="macro", zero_division=0))
        print("Chosen thresholds:", best["thr"])
        thresholds_to_use = best["thr"]
        default_to_use = args.fallback_class

    # Refit on train+val before final test
    rf = refit_on_train_plus_val(rf, train_df, val_df)

    # Final evaluation on played matches from test season (with thresholds or not)
    evaluate_on_test(rf, test_df, thresholds=thresholds_to_use, default=default_to_use)

    played_test = test_df.dropna(subset=["target"])
    if len(played_test):
        rf = refit_on_all_played(rf, train_df, val_df, played_test)
    else:
        rf = refit_on_all_played(rf, train_df, val_df)

        
    # Optional feature importance
    if not args.no_plot:
        plot_importances(rf)

    # Upcoming predictions + feature dump/print (built on FULL dataset)
    preds, up_feat = predict_upcoming_with_features(
        rf, df_all, args.test_season,
        thresholds=thresholds_to_use, default=default_to_use,
        dump_features_path=getattr(args, "dump_upcoming_features", None),
    )

    if not preds.empty:
        print("\nUpcoming prediction counts:")
        print(preds["pred_label"].value_counts())

        # Commented out to reduce output verbosity - uncomment to see feature details
        # print("\nSample of upcoming feature rows (first 10):")
        # try:
        #     print(up_feat.head(10).to_string(index=False))
        # except Exception:
        #     print(up_feat.head(10))

        if args.save_upcoming is not None:
            preds.to_csv(args.save_upcoming, index=False)
            print(f"\nSaved upcoming fixture predictions → {args.save_upcoming}")
    else:
        print("\nNo upcoming fixtures to save or display.")

if __name__ == "__main__":
    main()
