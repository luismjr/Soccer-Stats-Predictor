"""
Fetches basic Premier League match-level data from FBref.
Data is used to build features for the model.
"""

import re
import time
import datetime as dt
import pandas as pd
import requests
from urllib.parse import urljoin
from bs4 import BeautifulSoup, Comment
from pathlib import Path
from io import StringIO
from typing import Iterable, Optional

_SCORE_RE = re.compile(r"^\s*(\d+)\s*–\s*(\d+)\s*$") 
FBREF_BASE = "https://fbref.com" # Base URL for FBref
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

def season_label(end_year: int) -> str:
    """
    Return the label of the season for a given end year.
    """
    return f"{end_year-1}-{end_year}"


def pl_schedule_url(end_year: int) -> str:
    """
    Return the URL of the Premier League schedule page for a given season.

    Params:
    - end_year (int): The end year of the season (e.g., 2025) to collect data for
    Returns:
    - url (str): The URL of the Premier League schedule page for FBref
    """
    s = season_label(end_year)
    return f"https://fbref.com/en/comps/9/{s}/schedule/{s}-Premier-League-Scores-and-Fixtures"


def _extract_schedule_table_node(soup: BeautifulSoup):
    """
    This function finds the schedule table in the DOM of the page.

    Params:
    - soup (BeautifulSoup): The BeautifulSoup object of the page
    Returns:
    - table (BeautifulSoup): The BeautifulSoup object of the schedule table
    """

    # Try to find the schedule table with class=stats_table
    table = soup.find("table", {"class": "stats_table"})

    # If the table is found and contains "Home" and "Away" in the text, return the table
    if table and "Home" in table.text and "Away" in table.text:
        print("Found schedule table in DOM with class=stats_table")
        return table

def _extract_schedule_links(soup: BeautifulSoup) -> pd.DataFrame:
    """
    Parse the schedule table and pull out:
      - Date text
      - Home team name
      - Away team name
      - Match Report URL (absolute)
    Return a small DataFrame you can merge back on (Date, Home, Away).
    """
    # Try to find the schedule table, if not return empty dataframe
    table = _extract_schedule_table_node(soup)
    if not table:
        print("No schedule table found for link extraction")
        return pd.DataFrame(columns=["Date", "Home", "Away", "MatchReportUrl"])

    rows = []
    # Iterate through the table rows, skipping headers rows to be collected
    tbody = table.find("tbody") or table
    for table_row in tbody.find_all("tr"):
        # Skip repeat header rows
        if "class" in table_row.attrs and any(cls in ("thead", "over_header") for cls in table_row.attrs["class"]):
            continue

        # Skips empty separator rows
        tds = table_row.find_all(["td", "th"])
        if not tds:
            continue

        # Retrieves whole tag and then uses .get_text(strip=True) to get the text if applicable, otherwise returns empty string
        date = (table_row.find("td", {"data-stat": "date"}) or {}).get_text(strip=True) if table_row.find("td", {"data-stat": "date"}) else ""
        home = (table_row.find("td", {"data-stat": "home_team"}) or {}).get_text(strip=True) if table_row.find("td", {"data-stat": "home_team"}) else ""
        away = (table_row.find("td", {"data-stat": "away_team"}) or {}).get_text(strip=True) if table_row.find("td", {"data-stat": "away_team"}) else ""

        # Find the "Match Report" table data cell then find the anchor tag within it
        match_report_cell = table_row.find("td", {"data-stat": "match_report"})
        href = None
        if match_report_cell:
            a = match_report_cell.find("a")
            if a and a.get("href"):
                href = urljoin(FBREF_BASE, a["href"])

        if date and home and away:
            rows.append({
                "Date_raw": date,  
                "Home": home,
                "Away": away,
                "MatchReportUrl": href
            })

    print(f"Extracted {len(rows)} match rows, {sum(1 for r in rows if r['MatchReportUrl'])} with URLs")
    return pd.DataFrame(rows)

# ---------- HTML extraction ----------
def _extract_schedule_table_html(soup: BeautifulSoup) -> Optional[str]:
    """
    FBref often wraps tables in HTML comments. Try visible first, then comments.
    The schedule table usually has id starting with 'sched_' but id can vary;
    safest is to search by the table caption text or by columns typical to schedule.

    Params:
    - soup (BeautifulSoup): The BeautifulSoup object of the page
    Returns:
    - (str): The HTML of the table
    """
    # 1) Try to find a <table> that contains typical schedule headers
    candidates = soup.find_all("table", class_="stats_table")
    for t in candidates:
        if t.find("thead") and ("Home" in t.text and "Away" in t.text and "Score" in t.text):
            print(f"Found schedule table in DOM: {t}")
            return str(t)

    # 2) Search inside HTML comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if ("Home" in c and "Away" in c and "Score" in c) or ("Scores and Fixtures" in c):
            print(f"Found schedule table in comments: {c}")
            return str(c)

    print("No schedule table found in DOM or comments")
    return None

def _split_score(val: str) -> tuple[Optional[int], Optional[int]]:
    """
    This function splits the score into home and away goals for easier calculation and analysis.

    Params:
    - val (str): The score string (e.g., "1-0")
    Returns:
    - tuple[Optional[int], Optional[int]]: A tuple of two integers
    """
    # Normalizes the score
    parsed_score = _SCORE_RE.match(val.strip().replace("-", "–"))  # normalize

    # If the score is not valid, returns None
    if not parsed_score:
        return None, None

    # Returns the home and away goals as integers
    return int(parsed_score.group(1)), int(parsed_score.group(2))

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function normalizes the columns of the dataframe to make them more readable and consistent.

    Params:
    - df (pd.DataFrame): The dataframe with the columns to be normalized
    Returns:
    - df (pd.DataFrame): The dataframe with normalized columns
    """
    # Flatten MultiIndex if present
    if isinstance(df.columns, pd.MultiIndex): # Are the entire set of columns represented as a MultiIndex?
        new_columns = []
        for tup in df.columns:   # each header is a tuple, e.g. ("Playing Time", "MP")
            # convert each piece to a string, ignore "nan"
            parts = [str(x) for x in tup if str(x) != "nan"]
            # join pieces with spaces, e.g. "Playing Time MP"
            col_name = " ".join(parts).strip()
            new_columns.append(col_name)
        df.columns = new_columns

    # Standardize common column names
    rename_dict = {
        "Wk": "Week",
    }

    # Find and rename xG columns to be more descriptive
    xg_cols = [c for c in df.columns if c.startswith("xG")]
    if len(xg_cols) == 2:
        rename_dict[xg_cols[0]] = "HomexG"  # First xG column (home team)
        rename_dict[xg_cols[1]] = "AwayxG"   # Second xG column (away team)

    # Apply the renames
    df = df.rename(columns=rename_dict)

    # Create standardized goal columns from Score column
    if "Score" in df.columns:
        hgs, ags = [], []
        for score in df["Score"].astype(str).tolist():
            h, a = _split_score(score)  # split score to home and away using helper function
            hgs.append(h)
            ags.append(a)
        df["HomeGoals"] = hgs
        df["AwayGoals"] = ags

    # Drop the original 'Match Report' column. We don't need it.
    if "Match Report" in df.columns:
        df = df.drop(columns=["Match Report"])

    return df #df with new column names

def _postprocess_schedule(df: pd.DataFrame, end_year: int) -> pd.DataFrame:
    """
    This fucntion converts and drops data for easier pandas manipulation
    """
    # Drop header repeats if collected extra header rows
    if "Home" in df.columns:
        df = df[df["Home"].notna()]
    if "Away" in df.columns:
        df = df[df["Away"].notna()]

    # Convert date to datetime object for easier manipulation
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", utc=True) # replace invalid dates with NaT

    # Add season column for tracking
    df["Season"] = season_label(end_year)

    # Reset index after dropping header repeats
    return df.reset_index(drop=True)

# ---------- Main fetchers ----------
def fetch_pl_matches_one_season(end_year: int, timeout: int = 30) -> pd.DataFrame:
    """
    This function fetches the schedule for a given season and returns a pandas dataframe

    Params:
    - end_year (int): The end year of the season (e.g., 2025)
    - timeout (int): The timeout for the request in seconds (default: 30)
    Returns:
    - df (pd.DataFrame): The standardized dataframe with the schedule for the given season
    """

    # Get the url of the schedule page and make the request
    url = pl_schedule_url(end_year)
    print(f"Fetching schedule from {url}")
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.HTTPError as e:
        if resp.status_code == 404:
            print(f"Season {season_label(end_year)} not found (404) - likely doesn't exist yet")
            return pd.DataFrame()  # Return empty DataFrame
        elif resp.status_code == 403:
            print(f"Access forbidden (403) for {season_label(end_year)} - trying with delay...")
            time.sleep(5)  # Wait longer
            try:
                resp = requests.get(url, headers=HEADERS, timeout=timeout)
                resp.raise_for_status()
            except:
                print(f"Still blocked after retry - skipping {season_label(end_year)}")
                return pd.DataFrame()  # Return empty DataFrame
        else:
            raise e

    # Parse the response and extract the schedule table
    soup = BeautifulSoup(resp.text, "lxml")
    html = _extract_schedule_table_html(soup)
    if not html:
        print(f"Could not find schedule table on: {url}")
        raise RuntimeError(f"Could not find schedule table on: {url}")

    # Convert the html to a pandas dataframe
    sio = StringIO(html) # creates a file-like object in memory
    dfs = pd.read_html(sio)
    if not dfs:
        print("pandas.read_html found no tables in schedule HTML.")
        raise RuntimeError("pandas.read_html found no tables in schedule HTML.")

    # Heuristic: FBref schedules usually have the first table as the main schedule
    df = dfs[0]
    df = _standardize_columns(df)
    df = _postprocess_schedule(df, end_year)

    # 2. Get the links separately
    links_df = _extract_schedule_links(soup)
    # Parse Date_raw into datetime to match df["Date"]
    if not links_df.empty:
        links_df["Date"] = pd.to_datetime(links_df["Date_raw"], errors="coerce", utc=True)
        links_df = links_df.drop(columns=["Date_raw"])

        # Debug: Inspect links_df to ensure URLs are present
        print(f"Links DataFrame for {season_label(end_year)}:\n{links_df.head()}")

        # Merge into main schedule DataFrame
        df = df.merge(links_df, on=["Date", "Home", "Away"], how="left")

        # Debug: Check if MatchReportUrl is in the merged DataFrame
        print(f"Merged DataFrame columns: {df.columns.tolist()}")
        if "MatchReportUrl" in df.columns:
            # Rename MatchReportUrl to MatchReport for final output
            df = df.rename(columns={"MatchReportUrl": "MatchReport"})
            urls_found = df["MatchReport"].notna().sum()
            print(f"Found {urls_found} match report URLs out of {len(df)} matches")
        else:
            print(f"Warning: MatchReportUrl not found in merged DataFrame for {season_label(end_year)}")

    return df

def fetch_pl_matches_many(end_years: Iterable[int], delay_sec: float = 2.5) -> pd.DataFrame:
    """
    This function fetches the macthes from the previous 5 seasons

    Params:
    - end_years (Iterable[int]): The end years of the seasons (e.g., [2025, 2024, 2023])
    - delay_sec (float): The delay in seconds between requests (default: 2.5)
    Returns:
    - df (pd.DataFrame): The concatenated dataframe of matches from the previous 5 seasons
    """
    # Fetch the matches for each season and append to the frames list
    frames = []
    for y in end_years:
        print(f"Fetching schedule {season_label(y)} …")
        df = fetch_pl_matches_one_season(y)
        if not df.empty:
            frames.append(df)
        time.sleep(delay_sec)  # be polite

    if frames:
        return pd.concat(frames, ignore_index=True) # concatenated df has new indexes
    else:
        print("No data was successfully fetched!")
        return pd.DataFrame()
    

def get_available_seasons() -> list[int]:
    """
    Get list of available seasons based on current date.
    Avoids trying to fetch seasons that don't exist yet.
    """
    current_year = dt.datetime.now(dt.timezone.utc).year
    current_month = dt.datetime.now(dt.timezone.utc).month
    
    # Premier League seasons start in August
    if current_month >= 8:
        # Current season is ongoing, include it
        latest_season = current_year + 1
    else:
        # Current season hasn't started yet
        latest_season = current_year
    
    # Get last 5 completed/ongoing seasons
    years = list(range(latest_season, latest_season - 5, -1))
    print(f"Will attempt to fetch seasons: {[season_label(y) for y in years]}")
    return years

# ---------- CLI ----------
if __name__ == "__main__":
    # Get the latest seasons to fetch
    years = get_available_seasons()

    # Fetch the matches for the previous 5 seasons
    all_matches = fetch_pl_matches_many(years, delay_sec=2.5)

    # Create the output directory if it doesn't exist
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save per-season and combined
    for y in years:
        sl = season_label(y)
        all_matches[all_matches["Season"] == sl].to_csv(out_dir / f"matches_{sl}.csv", index=False)

    all_matches.to_csv(out_dir / "matches_last5.csv", index=False)
    print("Saved:", out_dir.resolve())