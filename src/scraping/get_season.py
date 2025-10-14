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
    - timeout (int): The timeout (wait time) for the request in seconds (default: 30)
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
    # Use StringIO to avoid FutureWarning about passing literal HTML
    sio = StringIO(html)
    df_list = pd.read_html(sio, match="Squad Standard Stats")
    if not df_list:
        # Some seasons don't include the title text in the parsed fragment; fallback to first table.
        sio = StringIO(html)  # Reset StringIO
        df_list = pd.read_html(sio)

    # Gets the firstdf from the web page
    df = df_list[0]

    # Reset indexes after dropping header repeats and return the dataframe
    return df



if __name__ == "__main__":
    # Gets the current year to determine which date to use for the current season
    current_year = dt.datetime.now(dt.timezone.utc).year

    # If date is in august, season has started, so this season includes next calendar year 
    # Otherwise use the current year
    if dt.datetime.now(dt.timezone.utc).month >= 8:
        last5_end_years = list(range(current_year + 1, current_year - 4, -1))  # e.g., 2025..2021
    else:
        last5_end_years = list(range(current_year, current_year - 4, -1))  # e.g., 2025..2021

    # Ensure output directory exists
    Path("data/raw/season").mkdir(parents=True, exist_ok=True)

    # Get season-wide stats for each season
    for y in last5_end_years: # last 5 seasons
        sl = season_label(y) # season label
        print(f"Fetching {sl} …")
        
        # Try to fetch the season-wide stats
        try:
            df = fetch_pl_squad_standard(y)
            # Save the dataframe to a csv file
            df.to_csv(f"data/raw/season/team_stats_{sl}.csv", index=False)
            print(f"Saved to data/raw/season/team_stats_{sl}.csv")
            time.sleep(3)  # Be polite between requests

        # If the season-wide stats are not found, print an error message and continue
        except Exception as e:
            print(f"  Failed to fetch {sl}: {e}")
            print(f"  Skipping this season and continuing...")
            continue

    print("Done ✓")
