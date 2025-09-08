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

HEADERS = {"User-Agent": "Mozilla/5.0 (educational project)"}


def season_label(end_year: int) -> str:
    """
    This function returns the label of the season in the format 'YYYY-YYYY' that FBref uses (e.g., 2024-2025)

    Params:
    - end_year (int): The end year of the season (e.g., 2025)
    Returns:
    - (str): The label in the format 'YYYY-YYYY'
    """
    return f"{end_year-1}-{end_year}"


def pl_season_url(end_year: int) -> str:
    """
    This function returns the URL of the Premier League season page

    Params:
    - end_year (int): The end year of the season (e.g., 2025)
    Returns:
    - (str): The URL of the Premier League season page
    """
    s = season_label(end_year)
    return f"https://fbref.com/en/comps/9/{s}/{s}-Premier-League-Stats"


def _extract_squad_standard_table_html(soup: BeautifulSoup) -> Optional[str]:
    """
    Find html of matches data by id directly or in comments to be converted into a dataframe

    Params:
    - soup (BeautifulSoup): The BeautifulSoup object of the page
    Returns:
    - (str): The HTML of the table
    """
    table_id = "stats_squads_standard_for"  # Squad Standard Stats (team-level)
    # Directly present in DOM?
    direct = soup.find("table", id=table_id)
    if direct:
        return str(direct)

    # Otherwise inside comments
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if table_id in c:
            return str(c)
    return None


def fetch_pl_squad_standard(end_year: int, timeout=30) -> pd.DataFrame:
    """
    Fetch one season's 'Squad Standard Stats' (one row per team).
    Returns a tidy DataFrame with a 'Season' column added.
    """
    # Sets and GETS the URL of the Premier League season page
    url = pl_season_url(end_year)
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()  # Raises an exception for HTTP errors

    # Parses the HTML of the page into a BeautifulSoup object then extracts the table as a string
    soup = BeautifulSoup(resp.text, "lxml")
    html = _extract_squad_standard_table_html(soup)
    if not html:
        raise RuntimeError(f"Could not find Squad Standard table on page: {url}")

    # Gets the df from the table. FBref sometimes includes repeated header rows; tidy them.
    sio = StringIO(html)
    df_list = pd.read_html(html, match="Squad Standard Stats")
    if not df_list:
        # Some seasons don't include the title text in the parsed fragment; fallback to first table.
        df_list = pd.read_html(html)

    # Gets the df from the table
    df = df_list[0]

    print("Raw DataFrame:")
    print(df.head(15))   # show first 15 rows so you can spot headers/NaNs
    print(df.info())     # shows row counts, dtypes, NaNs

    # Drop rows that are header repeats or aggregates (where 'Squad' is NaN or 'Squad' equals 'Squad')
    if "Squad" in df.columns:
        df = df[df["Squad"].notna()]
        df = df[df["Squad"].str.lower() != "squad"]

    # Add season label
    df["Season"] = season_label(end_year)

    # Optional: clean multi-index columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in tup if str(x) != "nan"]).strip() for tup in df.columns]

    return df.reset_index(drop=True)


def fetch_many_seasons(end_years: Iterable[int], delay_sec: float = 3.0) -> pd.DataFrame:
    """
    Loop over seasons and concatenate results.
    delay_sec: be polite between requests.
    """
    frames = []
    for y in end_years:
        print(f"Fetching {season_label(y)} …")
        df = fetch_pl_squad_standard(y)
        frames.append(df)
        time.sleep(delay_sec)
    combo = pd.concat(frames, ignore_index=True)
    return combo


if __name__ == "__main__":
    # Example: last 5 completed seasons ending up to the current year
    current_year = dt.datetime.now(dt.timezone.utc).year
    # If season has started this season includes next calendar year, otherwise use the current year and seasons
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
