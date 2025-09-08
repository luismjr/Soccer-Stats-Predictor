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

_NUM = r"-?\d{1,3}(?:,\d{3})*(?:\.\d+)?"
PCT_RE = re.compile(rf"^\s*({_NUM})\s*%\s*$")
NUM_RE = re.compile(rf"^\s*{_NUM}\s*$")
_MAIN_LABELS = ["Possession", "Passing Accuracy", "Shots on Target", "Saves"]

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
    Make a requests session with the browser headers.
    """
    s = requests.Session()
    s.headers.update(BROWSER_HEADERS)
    retries = Retry(
        total=4, 
        backoff_factor=1.5,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET", "HEAD"),
        raise_on_status=False,
    )
    # Adapter is used to customize the behavior of the session
    adapter = HTTPAdapter(max_retries=retries, pool_connections=8, pool_maxsize=8)

    # Mount the adapter to the session
    s.mount("https://", adapter)
    s.mount("http://", adapter)

    # Return the session
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
    return (s or "").replace("–", "-").replace("—", "-").replace("\xa0", " ").strip()

def _canon_label(raw: str) -> str:
    toks = re.findall(r"[A-Za-z0-9]+", _norm(raw).rstrip(":"))
    return "".join(t.title() for t in toks) if toks else ""

def _to_num(s: Optional[str]) -> Optional[float]:
    if s is None:
        return None
    s = _norm(s)
    m = PCT_RE.match(s)
    if m:
        try:
            return float(m.group(1).replace(",", ""))
        except Exception:
            return None
    if NUM_RE.match(s):
        try:
            return float(s.replace(",", ""))
        except Exception:
            return None
    return None

def parse_div_team_stats(soup: BeautifulSoup) -> Dict[str, float]:
    out: Dict[str, float] = {}
    cont = soup.select_one("div#team_stats")
    if not cont:
        return out

    txts = [t.strip() for t in cont.stripped_strings if t.strip()]
    for label in _MAIN_LABELS:
        # find label then take next two tokens as home/away
        idx = next((i for i, t in enumerate(txts) if t.lower() == label.lower()), None)
        if idx is None:
            token = label.split()[0].lower()
            idx = next((i for i, t in enumerate(txts) if token in t.lower()), None)
        if idx is None or idx + 2 >= len(txts):
            continue

        left, right = txts[idx + 1], txts[idx + 2]
        base = _canon_label(label)

        # split "x of y — z%" or "x/y z%" if present
        def split_combo(s: str):
            m_of = re.search(rf"({_NUM})\s*(?:of|/)\s*({_NUM})", s, re.I)
            p = re.search(rf"({_NUM})\s*%", s)
            made = float(m_of.group(1).replace(",", "")) if m_of else _to_num(s)
            att = float(m_of.group(2).replace(",", "")) if m_of else None
            pct = float(p.group(1).replace(",", "")) if p else None
            return made, att, pct

        h_made, h_att, h_pct = split_combo(left)
        a_made, a_att, a_pct = split_combo(right)

        if h_made is not None: out[f"Home{base}Made"] = h_made
        if h_att  is not None: out[f"Home{base}Att"]  = h_att
        if h_pct  is not None: out[f"Home{base}Pct"]  = h_pct
        if a_made is not None: out[f"Away{base}Made"] = a_made
        if a_att  is not None: out[f"Away{base}Att"]  = a_att
        if a_pct  is not None: out[f"Away{base}Pct"]  = a_pct

    return out

# ---------- grids: #team_stats_extra ----------
def find_team_stats_extra_node(soup: BeautifulSoup):
    node = soup.select_one("div#team_stats_extra")
    if node:
        return node
    for c in soup.find_all(string=lambda t: isinstance(t, Comment)):
        if "team_stats_extra" in c:
            inner = BeautifulSoup(c, "lxml")
            node = inner.select_one("div#team_stats_extra")
            if node:
                return node
    return None

def parse_div_team_stats_extra(soup: BeautifulSoup, home_team: str, away_team: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    wrapper = find_team_stats_extra_node(soup)
    if not wrapper:
        return out

    def leaf_texts(grid):
        leafs = []
        for d in grid.find_all("div"):
            if not d.find("div"):  # leaf
                if "class" in d.attrs and "th" in d["class"]:
                    continue  # skip team-name headers
                t = _norm(d.get_text(" ", strip=True))
                if t:
                    leafs.append(t)
        return leafs

    grids = wrapper.select("div.grid") or [wrapper]
    for grid in grids:
        tokens = leaf_texts(grid)

        # remove first-row team names triplet if present
        if len(tokens) >= 3 and (
            home_team.lower() in tokens[0].lower() or away_team.lower() in tokens[2].lower()
        ):
            tokens = tokens[3:]

        # Anchor on each label token; assign nearest numeric left/right
        for i, tok in enumerate(tokens):
            if not re.search(r"[A-Za-z]", tok):
                continue
            base = _canon_label(tok)
            if not base:
                continue

            # left numeric
            j = i - 1
            left_num = None
            while j >= 0:
                left_num = _to_num(tokens[j])
                if left_num is not None:
                    break
                j -= 1

            # right numeric
            k = i + 1
            right_num = None
            while k < len(tokens):
                right_num = _to_num(tokens[k])
                if right_num is not None:
                    break
                k += 1

            # write once per label (first seen wins)
            if left_num is not None and f"Home{base}Made" not in out:
                out[f"Home{base}Made"] = left_num
            if right_num is not None and f"Away{base}Made" not in out:
                out[f"Away{base}Made"] = right_num

    return out

# ---------- scrape one page ----------
def scrape_one_report(url: str, home: str, away: str, sess: requests.Session, timeout: int = 40) -> Dict[str, float]:
    r = sess.get(url, timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError(f"HTTP {r.status_code} for {url}")
    low = r.text.lower()
    if any(s in low for s in ("just a moment", "attention required", "cf-challenge", "cf-error")):
        raise RuntimeError("Blocked by interstitial (Cloudflare).")
    soup = BeautifulSoup(r.text, "lxml")

    stats: Dict[str, float] = {}
    stats.update(parse_div_team_stats(soup))
    extra = parse_div_team_stats_extra(soup, home, away)
    for k, v in extra.items():
        if k not in stats or stats[k] is None:
            stats[k] = v
    return stats

# ---------- season runner with checkpointing ----------
def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp = f"{path}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)

def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Home"] = df["Home"].astype(str).str.strip()
    df["Away"] = df["Away"].astype(str).str.strip()
    df["Date"] = pd.to_datetime(df["Date"], utc=True, errors="coerce")
    return df

def _filter_stats_cols(stats: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only keys + optional MatchReportUrl + stat columns (Home_*/Away_*),
    but drop Home_xG/Away_xG from stats so schedule's xG remains source of truth.
    """
    keys = ["Date", "Home", "Away"]
    maybe_url = ["MatchReportUrl"] if "MatchReportUrl" in stats.columns else []
    stat_cols = [c for c in stats.columns if c.startswith("Home_") or c.startswith("Away_")]
    stat_cols = [c for c in stat_cols if c not in ("Home_xG", "Away_xG")]
    keep = [c for c in (keys + maybe_url + stat_cols) if c in stats.columns]
    return stats[keep].copy()

def _coalesce_matchreport(merged: pd.DataFrame) -> pd.DataFrame:
    if "MatchReport" in merged.columns and "MatchReportUrl" in merged.columns:
        merged["MatchReport"] = merged["MatchReport"].fillna(merged["MatchReportUrl"])
        merged = merged.drop(columns=["MatchReportUrl"])
    elif "MatchReportUrl" in merged.columns and "MatchReport" not in merged.columns:
        merged = merged.rename(columns={"MatchReportUrl": "MatchReport"})
    return merged

def _clean_suffixes(merged: pd.DataFrame) -> pd.DataFrame:
    cols = list(merged.columns)
    for c in cols:
        if c.endswith("_x"):
            base = c[:-2]
            if base in merged.columns:
                merged = merged.drop(columns=[c])
            else:
                merged = merged.rename(columns={c: base})
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
    in_path = f"data/raw/matches_{season}.csv"
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

    # Load partial (if any) and build resume set by URL
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

    # Prepare loop
    total = int(sch[url_col].notna().sum())
    print(f"[{season}] Will process {total} matches with report URLs.")
    sess = make_session()
    rows: List[Dict] = []

    processed = 0
    successes = 0
    failures = 0

    for _, r in sch.iterrows():
        url = r.get(url_col)
        if not isinstance(url, str) or not url:
            continue

        if url in done_urls:
            processed += 1
            continue  # skip already scraped

        home, away = str(r["Home"]), str(r["Away"])
        print(f"[{season}] Fetching {home} vs {away} ({processed + 1}/{total})")
        polite_sleep(delay_min, delay_max)

        try:
            stats = scrape_one_report(url, home, away, sess)
        except Exception as e:
            failures += 1
            print(f"[{season}] Failed: {e}")
            processed += 1
            continue

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

        # Checkpoint every N successes
        if checkpoint_every > 0 and (successes % checkpoint_every == 0):
            ck_df = pd.concat([partial_df, pd.DataFrame(rows)], ignore_index=True) if not partial_df.empty else pd.DataFrame(rows)
            atomic_write_csv(ck_df, partial_path)
            print(f"[{season}] Checkpoint saved → {partial_path}  (rows={len(ck_df)})")

        if processed % 5 == 0:
            print(f"[{season}] Progress: {processed}/{total} | ok={successes} fail={failures}")

    # Finalize partial (in case last chunk < checkpoint_every)
    if rows:
        ck_df = pd.concat([partial_df, pd.DataFrame(rows)], ignore_index=True) if not partial_df.empty else pd.DataFrame(rows)
        atomic_write_csv(ck_df, partial_path)
        print(f"[{season}] Final checkpoint saved → {partial_path}  (rows={len(ck_df)})")

    # ---- SAFE MERGE STRATEGY (same as your merge_matches.py) ----
    if os.path.exists(partial_path):
        extra = pd.read_csv(partial_path, parse_dates=["Date"])
        extra = _normalize_keys(extra)
    else:
        extra = pd.DataFrame(columns=["Date", "Home", "Away", "MatchReportUrl"])

    if extra.empty:
        print(f"[{season}] No advanced stats parsed; saving schedule as-is.")
        features = sch.copy()
    else:
        # keep only keys + URL + Home_/Away_ stats (drop xG from stats), dedupe
        extra_clean = _filter_stats_cols(extra)
        before = len(extra_clean)
        extra_clean = extra_clean.drop_duplicates(subset=["Date", "Home", "Away"], keep="last")
        removed = before - len(extra_clean)
        if removed:
            print(f"[{season}] Deduped extra: removed {removed} duplicate rows on (Date,Home,Away)")

        # merge
        features = sch.merge(extra_clean, on=["Date", "Home", "Away"], how="left")
        # coalesce MatchReport with URL, drop URL
        features = _coalesce_matchreport(features)
        # scrub any lingering _x/_y
        features = _clean_suffixes(features)

    atomic_write_csv(features, out_path)
    print(f"[{season}] Done. ok={successes} fail={failures} → {out_path}")
    print(f"[{season}] Partial (resume file) remains at → {partial_path}")

# ---------- CLI ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", required=True, help="Season label, e.g. 2024-2025")
    ap.add_argument("--delay-min", type=float, default=8.0)
    ap.add_argument("--delay-max", type=float, default=14.0)
    ap.add_argument("--checkpoint-every", type=int, default=10, help="Save partial after this many successes (0 to disable)")
    args = ap.parse_args()

    run_season(args.season, args.delay_min, args.delay_max, args.checkpoint_every)

if __name__ == "__main__":
    main()
