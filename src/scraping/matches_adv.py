"""
matches_adv.py — FBref Match Report scraper for a whole season with checkpointing

Parses:
  • DIV#team_stats (bars): Possession, Passing Accuracy, Shots on Target, Saves
  • DIV#team_stats_extra (grids): Corners, Fouls, Crosses, Interceptions, Offsides,
    Goal Kicks, Throw Ins, Long Balls, etc. (label-anchored nearest-number capture)

Features:
  • Checkpoint every N matches → data/tmp/match_reports_{SEASON}_partial.csv
  • Auto-resume: skips matches already in the partial
  • Final output → data/processed/matches_with_reports_{SEASON}.csv

Usage:
  python src/scraping/matches_adv.py --season 2024-2025 --checkpoint-every 10 \
    --delay-min 8 --delay-max 14
"""

import argparse
import os
import random
import re
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from requests.adapters import HTTPAdapter, Retry
from typing import Dict, List, Optional

_NUM = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?" # Matches numbers with optional commas and decimal points
PCT_RE = re.compile(rf"^\s*({_NUM})\s*%\s*$") # Turns percentages into objects with methods
NUM_RE = re.compile(rf"^\s*{_NUM}\s*$") # Turnes numbers into objects with methods
_MAIN_LABELS = ["Possession", "Passing Accuracy", "Shots on Target", "Saves"] # Main labels to extract

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/119.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://fbref.com/",
    "Connection": "keep-alive",
}
def make_session() -> requests.Session:
    """
    Creates a robust requests session with automatic retries and browser-like headers.
    
    The session is configured to:
    - Use realistic browser headers to avoid being blocked
    - Automatically retry failed requests (up to 4 times)
    - Use exponential backoff between retries
    - Handle common HTTP errors (429, 500, 502, 503, 504)
    
    Returns:
    - requests.Session: A configured session ready for web scraping
    """
    # Startes session with browser headers
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    
    # Configure automatic retry strategy object
    retries = Retry(
        total=4,  # Maximum number of retry attempts
        backoff_factor=1.5,  # Wait time multiplier: 1.5s, 2.25s, 3.375s, etc.
        status_forcelist=(429, 500, 502, 503, 504),  # HTTP errors that trigger retry
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    
    # Adapter customizes session behavior with retry logic and connection pooling (Reusing /SockeyTCP connections)
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)

    # Mount the adapter to handle both http and https requests
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    return s

def polite_sleep(min_s: float, max_s: float) -> None:
    """
    Sleeps the program for a random amount of time (Between min_s and max_s)

    Params:
    - min_s (float): minimum amount of seconds to sleep the program
    - max_s (float): the maximum amount of seconds to sleep the program
    Returns:
    - Nothing
    """
    if max_s <= 0:
        return
    time.sleep(random.uniform(min_s, max_s))


def _norm(s: str) -> str:
    """
    Normalizes text by standardizing dashes, removing non-breaking spaces, and trimming whitespace.
    
    Params:
    - s (str): Raw text string
    Returns:
    - (str): Normalized text
    """
    return (s or "").replace("–", "-").replace("—", "-").replace("\xa0", " ").strip()

def _canon_label(raw: str) -> str:
    """
    Converts a label to canonical form for use as a dictionary key.
    Examples: "Shots on Target" → "ShotsOnTarget", "passing accuracy:" → "PassingAccuracy"
    
    Params:
    - raw (str): Raw label text
    Returns:
    - (str): Canonical label in TitleCase with no spaces or punctuation
    """
    toks = re.findall(r"[A-Za-z0-9]+", _norm(raw).rstrip(":"))
    return "".join(t.title() for t in toks) if toks else ""

def _to_num(s: Optional[str]) -> Optional[float]:
    """
    Converts a string to a float, handling percentages and comma-separated numbers.
    Examples: "45.2%", "1,234", "67.5" → 45.2, 1234.0, 67.5
    
    Params:
    - s (Optional[str]): String representation of a number
    Returns:
    - (Optional[float]): Parsed float value, or None if parsing fails
    """
    if s is None:
        return None
    s = _norm(s)
    # Check if it's a percentage
    m = PCT_RE.match(s)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            return None
    # Check if it's a regular number
    if NUM_RE.match(s):
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None
    return None

def parse_div_team_stats(soup: BeautifulSoup) -> Dict[str, float]:
    """
    Parses the main team stats section (div#team_stats) which displays key match stats as bars.
    
    This extracts metrics like:
    - Possession: e.g., "55% vs 45%"
    - Passing Accuracy: e.g., "234 of 289 — 81% vs 198 of 251 — 79%"
    - Shots on Target: e.g., "5 of 12 vs 3 of 8"
    - Saves: e.g., "3 vs 5"
    
    For each stat, extracts up to three components:
    - Made: The first number (e.g., shots made, passes completed)
    - Att: Attempts (if present, e.g., "5 of 12" → made=5, att=12)
    - Pct: Percentage (if present)
    
    Params:
    - soup (BeautifulSoup): The parsed HTML of the match report page
    Returns:
    - Dict[str, float]: Dictionary with keys like "HomePossessionMade", "AwayPassingAccuracyPct", etc.
    """
    out: Dict[str, float] = {}
    cont = soup.select_one("div#team_stats")
    if not cont:
        return out

    # Get all text tokens from the container
    txts = [t.strip() for t in cont.stripped_strings if t.strip()]
    
    for label in _MAIN_LABELS:
        # Find the label in the text, then the next two tokens are home/away values
        # Give me index for every image and text elementif it mathces the label
        idx = next((i for i, t in enumerate(txts) if t.lower() == label.lower()), None)
        if idx is None:
            # Fallback: try matching just the first word (e.g., "Possession")
            token = label.split()[0].lower()
            idx = next((i for i, t in enumerate(txts) if token in t.lower()), None)
        if idx is None or idx + 2 >= len(txts):
            continue

        left, right = txts[idx + 1], txts[idx + 2]  # Home, Away values
        base = _canon_label(label)

        # Helper function to split compound stats like "234 of 289 — 81%"
        def split_combo(s: str):
            m_of = re.search(rf"({_NUM})\s*(?:of|/)\s*({_NUM})", s, re.I)
            p = re.search(rf"({_NUM})\s*%", s)
            made = float(m_of.group(1).replace(",", "")) if m_of else _to_num(s)
            att = float(m_of.group(2).replace(",", "")) if m_of else None
            pct = float(p.group(1).replace(",", "")) if p else None
            return made, att, pct

        h_made, h_att, h_pct = split_combo(left)
        a_made, a_att, a_pct = split_combo(right)

        # Store each component with descriptive keys
        if h_made is not None: out[f"Home{base}Made"] = h_made
        if h_att  is not None: out[f"Home{base}Att"]  = h_att
        if h_pct  is not None: out[f"Home{base}Pct"]  = h_pct
        if a_made is not None: out[f"Away{base}Made"] = a_made
        if a_att  is not None: out[f"Away{base}Att"]  = a_att
        if a_pct  is not None: out[f"Away{base}Pct"]  = a_pct

    return out

# ---------- grids: #team_stats_extra ----------
def find_team_stats_extra_node(soup: BeautifulSoup):
    """
    Locates the div#team_stats_extra section, which may be in the DOM or hidden in HTML comments.
    
    FBref sometimes wraps certain tables in HTML comments to prevent easy scraping.
    This function checks both locations.
    
    Params:
    - soup (BeautifulSoup): The parsed HTML of the match report page
    Returns:
    - BeautifulSoup node or None: The div#team_stats_extra element if found
    """
    # Try finding it directly in the DOM first
    node = soup.select_one("div#team_stats_extra")
    if node:
        return node
    
    # Search within HTML comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "team_stats_extra" in c:
            inner = BeautifulSoup(c, "lxml")
            node = inner.select_one("div#team_stats_extra")
            if node:
                return node
    return None

def parse_div_team_stats_extra(soup: BeautifulSoup, home_team: str, away_team: str) -> Dict[str, float]:
    """
    Parses the extra team stats section (div#team_stats_extra) which displays additional match stats in grid format.
    
    This extracts metrics like:
    - Corners, Fouls, Crosses, Interceptions, Offsides
    - Goal Kicks, Throw Ins, Long Balls, etc.
    
    The layout is typically:
        Home Value | Label | Away Value
        Example: "6 | Corners | 4"
    
    This function uses a label-anchored parsing strategy:
    1. Find each label (e.g., "Corners", "Fouls")
    2. Search left for the home team's value
    3. Search right for the away team's value
    
    Params:
    - soup (BeautifulSoup): The parsed HTML of the match report page
    - home_team (str): Home team name (used to skip header rows)
    - away_team (str): Away team name (used to skip header rows)
    Returns:
    - Dict[str, float]: Dictionary with keys like "HomeCornersMade", "AwayFoulsMade", etc.
    """
    out: Dict[str, float] = {}
    wrapper = find_team_stats_extra_node(soup)
    if not wrapper:
        return out

    def leaf_texts(grid):
        """Extract text from leaf-level div elements (those without child divs)."""
        leafs = []
        for d in grid.find_all("div"):
            if not d.find("div"):  # leaf node (no children)
                if "class" in d.attrs and "th" in d["class"]: # Table header row
                    continue  # skip team-name headers
                t = _norm(d.get_text(" ", strip=True))
                if t: # If text is not empty, add it to the list
                    leafs.append(t)
        return leafs

    # Find all grid containers (or use wrapper if no grids found)
    grids = wrapper.select("div.grid") or [wrapper]
    
    for grid in grids:
        tokens = leaf_texts(grid)

        # Remove first-row team name headers if present (e.g., "Arsenal | Stats | Chelsea")
        if len(tokens) >= 3 and (
            home_team.lower() in tokens[0].lower() or away_team.lower() in tokens[2].lower()
        ):
            tokens = tokens[3:]

        # Parse using label-anchored strategy
        for i, tok in enumerate(tokens):
            # Skip tokens that are purely numeric (not labels)
            if not re.search(r"[A-Za-z]", tok):
                continue
            base = _canon_label(tok)
            if not base:
                continue

            # Search left for home team's value
            j = i - 1
            left_num = None
            while j >= 0:
                left_num = _to_num(tokens[j])
                if left_num is not None:
                    break
                j -= 1

            # Search right for away team's value
            k = i + 1
            right_num = None
            while k < len(tokens):
                right_num = _to_num(tokens[k])
                if right_num is not None:
                    break
                k += 1

            # Store values (first occurrence of each label wins, to avoid duplicates)
            if left_num is not None and f"Home{base}Made" not in out:
                out[f"Home{base}Made"] = left_num
            if right_num is not None and f"Away{base}Made" not in out:
                out[f"Away{base}Made"] = right_num

    return out

# ---------- scrape one page ----------
def scrape_one_report(url: str, home: str, away: str, sess: requests.Session, timeout: int = 40) -> Dict[str, float]:
    """
    Scrapes advanced statistics from a single FBref match report page.
    
    Combines data from both the main stats bar (div#team_stats) and 
    the extra stats grid (div#team_stats_extra).
    
    Params:
    - url (str): The full URL of the match report page
    - home (str): Home team name
    - away (str): Away team name
    - sess (requests.Session): Session with retry logic and browser headers
    - timeout (int): Request timeout in seconds (default: 40)
    Returns:
    - Dict[str, float]: Combined statistics dictionary
    Raises:
    - RuntimeError: If HTTP request fails or page is blocked by Cloudflare
    """
    r = sess.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    
    # Check for Cloudflare or other blocking mechanisms
    low = r.text.lower()
    if any(s in low for s in ("just a moment", "attention required", "cf-challenge", "cf-error")):
        raise RuntimeError("Blocked by interstitial (Cloudflare).")
    
    soup = BeautifulSoup(r.text, "lxml")

    # Parse main stats (possession, passing accuracy, shots on target, saves)
    stats: Dict[str, float] = {}
    stats.update(parse_div_team_stats(soup))
    
    # Parse extra stats (corners, fouls, crosses, etc.)
    # Only add extra stats if they don't already exist in main stats
    extra = parse_div_team_stats_extra(soup, home, away)
    for k, v in extra.items():
        if k not in stats or stats[k] is None:
            stats[k] = v
    
    return stats

# ---------- season runner with checkpointing ----------
def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    """
    Writes a CSV file atomically using a temporary file to prevent corruption.
    
    If the script crashes mid-write, the original file (if any) remains intact.
    
    Params:
    - df (pd.DataFrame): DataFrame to save
    - path (str): Target file path
    """
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)  # Atomic operation on most filesystems

def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes the key columns (Date, Home, Away) for reliable merging.
    
    Ensures team names are stripped of whitespace and dates are in UTC format.
    
    Params:
    - df (pd.DataFrame): DataFrame with key columns
    Returns:
    - pd.DataFrame: DataFrame with normalized keys
    """
    df = df.copy()
    df["Home"] = df["Home"].astype(str).str.strip()
    df["Away"] = df["Away"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df

def _filter_stats_cols(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Filters the stats DataFrame to keep only relevant columns.
    
    Keeps:
    - Key columns: Date, Home, Away
    - MatchReportUrl (if present)
    - Stat columns starting with "Home_" or "Away_"
    
    Drops:
    - Home_xG and Away_xG (we use xG from the base schedule instead)
    
    Params:
    - stats (pd.DataFrame): Stats DataFrame from match reports
    Returns:
    - pd.DataFrame: Filtered DataFrame
    """
    keys = ["Date", "Home", "Away"]
    maybe_url = ["MatchReportUrl"] if "MatchReportUrl" in stats.columns else []
    # Match columns starting with "Home" or "Away" (without underscore requirement)
    # But exclude the key columns "Home" and "Away" themselves
    stat_cols = [c for c in stats.columns 
                 if (c.startswith("Home") or c.startswith("Away")) 
                 and c not in ("Home", "Away", "HomexG", "AwayxG")]
    keep = [c for c in (keys + maybe_url + stat_cols) if c in stats.columns]
    return stats[keep].copy()

def _coalesce_matchreport(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Consolidates MatchReport and MatchReportUrl columns into a single MatchReport column.
    
    Params:
    - merged (pd.DataFrame): Merged DataFrame that may have both columns
    Returns:
    - pd.DataFrame: DataFrame with single MatchReport column
    """
    if "MatchReport" in merged.columns and "MatchReportUrl" in merged.columns:
        merged["MatchReport"] = merged["MatchReport"].fillna(merged["MatchReportUrl"])
        merged = merged.drop(columns=["MatchReportUrl"])
    elif "MatchReportUrl" in merged.columns and "MatchReport" not in merged.columns:
        merged = merged.rename(columns={"MatchReportUrl": "MatchReport"})
    return merged

def _clean_suffixes(merged: pd.DataFrame) -> pd.DataFrame:
    """
    Removes pandas merge suffixes (_x, _y) from column names after merging.
    
    Strategy:
    - If both base and base_x exist, drop base_x
    - If only base_x exists, rename to base
    - Same logic for _y suffix
    
    Params:
    - merged (pd.DataFrame): DataFrame with potential _x/_y suffixes
    Returns:
    - pd.DataFrame: DataFrame with clean column names
    """
    cols = list(merged.columns)
    # Clean _x suffixes
    for c in cols:
        if c.endswith("_x"):
            base = c[:-2]
            if base in merged.columns:
                merged = merged.drop(columns=[c])
            else:
                merged = merged.rename(columns={c: base})
    
    # Clean _y suffixes
    cols = list(merged.columns)
    for c in cols:
        if c.endswith("_y"):
            base = c[:-2]
            if base in merged.columns or (base + "_x") in merged.columns:
                merged = merged.drop(columns=[c])
            else:
                merged = merged.rename(columns={c: base})
    return merged

def run_season(season: str, delay_min: float, delay_max: float, checkpoint_every: int) -> None:
    """
    Main orchestrator: scrapes all match reports for a season with automatic checkpointing.
    
    Process:
    1. Load base schedule from data/raw/match/match_stats_{season}.csv
    2. Check for existing partial file (resume capability)
    3. Scrape each match report URL with polite delays
    4. Save checkpoints periodically to data/tmp/
    5. Merge scraped stats with base schedule
    6. Save final output to data/processed/matches_with_reports_{season}.csv
    
    Features:
    - Auto-resume: Skips already-scraped matches if partial file exists
    - Checkpointing: Saves progress every N matches to prevent data loss
    - Safe merge: Handles column conflicts and deduplication
    
    Params:
    - season (str): Season label, e.g., "2024-2025"
    - delay_min (float): Minimum seconds to wait between requests
    - delay_max (float): Maximum seconds to wait between requests
    - checkpoint_every (int): Save partial file after this many successful scrapes (0 to disable)
    """
    in_path = f"data/raw/match/match_stats_{season}.csv"
    out_dir = "data/processed"
    tmp_dir = "data/tmp"
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    out_path = os.path.join(out_dir, f"matches_with_reports_{season}.csv")
    partial_path = os.path.join(tmp_dir, f"match_reports_{season}_partial.csv")

    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Schedule file not found: {in_path}")

    sch = pd.read_csv(in_path, parse_dates=["Date"])
    sch = _normalize_keys(sch)

    url_col = next((c for c in ("MatchReportUrl", "MatchReport") if c in sch.columns), None)
    if url_col is None:
        raise ValueError("Schedule missing MatchReport/MatchReportUrl column")

    # Load partial checkpoint file (if any) to enable resume capability
    partial_df = pd.DataFrame()
    done_urls = set()
    if os.path.exists(partial_path):
        try:
            partial_df = pd.read_csv(partial_path, low_memory=False, parse_dates=["Date"])
            if "MatchReportUrl" in partial_df.columns:
                done_urls = set(partial_df["MatchReportUrl"].dropna().astype(str).tolist())
            print(f"[{season}] Resuming with {len(done_urls)} already scraped from partial.")
        except Exception as e:
            print(f"[{season}] Warning: failed to read partial ({e}). Starting fresh.")

    # Prepare loop - count only completed matches (with scores and match report URLs)
    completed_mask = (
        sch[url_col].notna() & 
        sch[url_col].str.contains("/matches/", na=False) &
        sch["Score"].notna() &
        (sch["Score"].astype(str).str.strip() != "")
    )
    total = int(completed_mask.sum())
    print(f"[{season}] Will process {total} completed matches (skipping {int(sch[url_col].notna().sum()) - total} future matches).")
    sess = make_session()
    rows: List[Dict] = []

    processed = 0
    successes = 0
    failures = 0

    # Main scraping loop
    for _, r in sch.iterrows():
        url = r.get(url_col)
        if not isinstance(url, str) or not url:
            continue  # Skip matches without report URLs

        # Skip uncompleted matches (they have history URLs instead of match report URLs)
        if "/stathead/matchup/" in url or "/matches/" not in url:
            continue  # Skip future/uncompleted matches
        
        # Additional check: skip if no score (match not played yet)
        score = r.get("Score")
        if pd.isna(score) or str(score).strip() == "":
            continue  # Skip matches without scores

        if url in done_urls:
            processed += 1
            continue  # Skip already-scraped matches (resume functionality)

        home, away = str(r["Home"]), str(r["Away"])
        print(f"[{season}] Fetching {home} vs {away} ({processed + 1}/{total})")
        polite_sleep(delay_min, delay_max)  # Be polite to avoid being blocked

        try:
            stats = scrape_one_report(url, home, away, sess)
        except Exception as e:
            failures += 1
            print(f"[{season}] Failed: {e}")
            processed += 1
            continue

        # Build row with match metadata + scraped stats
        row = {
            "Date": pd.to_datetime(r["Date"], utc=True, errors="coerce"),
            "Home": home,
            "Away": away,
            "MatchReportUrl": url,
            **stats
        }
        rows.append(row)
        successes += 1
        processed += 1
        done_urls.add(url)

        print(f"[{season}] Parsed ✓ (ok={successes}, fail={failures})")

        # Checkpoint: Save progress periodically
        if checkpoint_every > 0 and (successes % checkpoint_every == 0):
            ck_df = pd.concat([partial_df, pd.DataFrame(rows)], ignore_index=True) if not partial_df.empty else pd.DataFrame(rows)
            atomic_write_csv(ck_df, partial_path)
            print(f"[{season}] Checkpoint saved → {partial_path}  (rows={len(ck_df)})")

        # Progress update
        if processed % 5 == 0:
            print(f"[{season}] Progress: {processed}/{total} | ok={successes} fail={failures}")

    # Finalize partial (in case last chunk < checkpoint_every)
    if rows:
        ck_df = pd.concat([partial_df, pd.DataFrame(rows)], ignore_index=True) if not partial_df.empty else pd.DataFrame(rows)
        atomic_write_csv(ck_df, partial_path)
        print(f"[{season}] Final checkpoint saved → {partial_path}  (rows={len(ck_df)})")

    # ---- FINAL MERGE: Combine base schedule with scraped advanced stats ----
    if os.path.exists(partial_path):
        extra = pd.read_csv(partial_path, parse_dates=["Date"])
        extra = _normalize_keys(extra)
    else:
        extra = pd.DataFrame(columns=["Date", "Home", "Away", "MatchReportUrl"])

    if extra.empty:
        print(f"[{season}] No advanced stats parsed; saving schedule as-is.")
        features = sch.copy()
    else:
        # Filter: Keep only keys + URL + stat columns (drop xG to avoid conflicts)
        extra_clean = _filter_stats_cols(extra)
        
        # Deduplicate: Keep last occurrence if same match was scraped multiple times
        before = len(extra_clean)
        extra_clean = extra_clean.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
        removed = before - len(extra_clean)
        if removed:
            print(f"[{season}] Deduped extra: removed {removed} duplicate rows on (Date,Home,Away)")

        # Merge schedule with advanced stats (left join keeps all schedule rows)
        features = sch.merge(extra_clean, on=["Date", "Home", "Away"], how="left")
        
        # Clean up: Consolidate MatchReport columns
        features = _coalesce_matchreport(features)
        
        # Clean up: Remove pandas merge suffixes (_x, _y)
        features = _clean_suffixes(features)

    atomic_write_csv(features, out_path)
    print(f"[{season}] Done. ok={successes} fail={failures} → {out_path}")
    print(f"[{season}] Partial (resume file) remains at → {partial_path}")

# ---------- CLI ----------
def main():
    """
    Command-line interface for running the match report scraper.
    
    Example usage:
        python src/scraping/matches_adv.py --season 2024-2025 --checkpoint-every 10 --delay-min 8 --delay-max 14
    """
    ap = argparse.ArgumentParser(
        description="Scrape advanced match statistics from FBref match report pages with checkpointing"
    )
    ap.add_argument("--season", required=True, help="Season label, e.g. 2024-2025")
    ap.add_argument("--delay-min", type=float, default=8.0, help="Minimum delay between requests (seconds)")
    ap.add_argument("--delay-max", type=float, default=14.0, help="Maximum delay between requests (seconds)")
    ap.add_argument("--checkpoint-every", type=int, default=10, 
                   help="Save partial after this many successes (0 to disable)")
    args = ap.parse_args()

    run_season(args.season, args.delay_min, args.delay_max, args.checkpoint_every)

if __name__ == "__main__":
    main()
