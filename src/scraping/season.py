"""
Gets season-wide stats of each premier league team of that season
"""


import time
import datetime as dt
from typing import Iterable, Optional
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment


# Simple headers for few requests
HEADERS = {"User-Agent": "Mozilla/5.0 (educational project)"}


def season_label(end_year: int) -> str:
    """
    Computes the date label of a season in the format 'YYYY-YYYY' that FBref uses (e.g., 2024-2025)
    Used for properly formatting the URL of the Premier League season page for season-wide stats

    Params:
    - end_year (int): The end year of the season (e.g., 2025)
    Returns:
    - (str): The label in the format 'YYYY-YYYY'
    """
    return f"{end_year-1}-{end_year}"


def pl_season_url(end_year: int) -> str:
    """
    Computes the entire URL of the Premier League season page for season-wide stats

    Params:
    - end_year (int): The end year of the season (e.g., 2025)
    Returns:
    - (str): The URL of the Premier League season page
    """
    s = season_label(end_year)
    return f"https://fbref.com/en/comps/9/{s}/{s}-Premier-League-Stats"


def _extract_squad_standard_table_html(soup: BeautifulSoup) -> Optional[str]:
    """
    Extract the Squad Standard Stats table from the fbref web page.

    FBref may render tables directly in the DOM or hide them in HTML comments.
    This function checks both locations.

    Params:
    - soup (BeautifulSoup): The BeautifulSoup object of the page
    Returns:
    - (str | None): The HTML string of the table, or None if not found
    """
    # 1) Try to find the table directly in the DOM (table exists as a normal <table> element)
    table_id = "stats_squads_standard_for"  # Squad Standard Stats (team-level)
    # Directly present in DOM?
    direct = soup.find("table", id=table_id)
    if direct:
        return str(direct) # Return the HTML string of the table if found

    # 2) Search inside HTML comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in c:
            return str(c)

    return None # Return None if the table is not found


def fetch_pl_squad_standard(end_year: int, timeout=30) -> pd.DataFrame:
    """
    Fetch one season's 'Squad Standard Stats' (one row per team).
    Returns a tidy DataFrame with a 'Season' column added.

    Params:
    - end_year (int): The end year of the season (e.g., 2025)
    - timeout (int): The timeout for the request in seconds (default: 30)
    Returns:
    - df (pd.DataFrame): The dataframe of squad standard stats for the given season
    """
    # Gets the URL of the specific Premier League season 
    url = pl_season_url(end_year)

    # Makes the request to the URL
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()  # Raises an exception for HTTP errors

    # Parses the HTML of the page using lxml, and builds a searchable BeautifulSoup object (Parse tree)
    soup = BeautifulSoup(resp.text, "lxml")
    html = _extract_squad_standard_table_html(soup)
    if not html:
        raise RuntimeError(f"Could not find Squad Standard table on page: {url}")

    # Finds the tables with the title "Squad Standard Stats"
    df_list = pd.read_html(html, match="Squad Standard Stats")
    if not df_list:
        # Some seasons don't include the title text in the parsed fragment; fallback to first table.
        df_list = pd.read_html(html)

    # Gets the firstdf from the table
    df = df_list[0]

    # print("Raw DataFrame:")
    # print(df.head(15))   # show first 15 rows to can spot headers/NaNs
    # print(df.info())     # shows row counts, dtypes, NaNs

    # Drop rows that are header repeats or aggregates (where 'Squad' is NaN or 'Squad' equals 'Squad')
    if "Squad" in df.columns:
        df = df[df["Squad"].notna()]
        df = df[df["Squad"].str.lower() != "squad"]

    # Add season label
    df["Season"] = season_label(end_year)

    # Clean multi-index columns (Multiople header rows) if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns]

    # Reset indexes after dropping header repeats and return the dataframe
    return df.reset_index(drop=True)


def fetch_many_seasons(end_years: Iterable[int], delay_sec: float = 3.0) -> pd.DataFrame:
    """
    Loop over seasons and concatenate results.
    
    Params:
    - end_years (Iterable[int]): The end years of each season to be fetches data from (e.g., [2025, 2024, 2023])
    - delay_sec (float): The delay in seconds between requests (default: 3.0)
    Returns:
    - combo (pd.DataFrame): The concatenated dataframe of squad standard stats from the previous 5 seasons
    """
    # Start a list of pandas dataframes
    frames = []

    for y in end_years:
        print(f"Fetching {season_label(y)} …")
        df = fetch_pl_squad_standard(y)
        frames.append(df)
        time.sleep(delay_sec)
    combo = pd.concat(frames, ignore_index=True)
    return combo


if __name__ == "__main__":
    # Gets the current year to determine which date to use for the current season
    current_year = dt.datetime.now(dt.timezone.utc).year

    # If date is in august, season has started, so this season includes next calendar year 
    # Otherwise use the current year
    if dt.datetime.now(dt.timezone.utc).month >= 8:
        last5_end_years = list(range(current_year + 1, current_year - 4, -1))  # e.g., 2025..2021
    else:
        last5_end_years = list(range(current_year, current_year - 4, -1))  # e.g., 2025..2021

    all_df = fetch_many_seasons(last5_end_years, delay_sec=3)

    # Save per-season and combined
    for y in last5_end_years:
        sl = season_label(y)
        out = all_df[all_df["Season"] == sl].reset_index(drop=True)
        out.to_csv(f"data/raw/squad_standard_{sl}.csv", index=False)

    all_df.to_csv("data/raw/squad_standard_last5.csv", index=False)
    print("Done ✓")
