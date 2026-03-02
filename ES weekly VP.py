"""
Weekly Volume Profile Calculator (ES Futures)
----------------------------------------------
Calculates the previous week's POC, VAH, VAL using
1-minute close prices from Databento (CME Globex).

Session: Monday 18:00 ET -> Friday 15:59 ET
Data:    ohlcv-1m via Databento Historical API

Usage:
    python weekly_vp.py              (default 70% VA)
    python weekly_vp.py --va 68      (custom VA%)
    python weekly_vp.py --va 80
"""

import argparse
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

# Import local telegram utility
from telegram_notifier import send_telegram_message, format_vp_message

import databento as db
import numpy as np
import pandas as pd

# -- CONFIG --------------------------------------------------
SYMBOL      = "ES.c.0"           # continuous front-month ES
DATASET     = "GLBX.MDP3"       # CME Globex
SCHEMA      = "ohlcv-1m"        # 1-minute OHLCV
TICK_SIZE   = 0.25               # ES tick size
TZ_ET       = ZoneInfo("US/Eastern")
TZ_UTC      = ZoneInfo("UTC")

# Data cache directory (same folder as the script)
DATA_DIR    = Path(__file__).parent / "data"


def get_previous_week_dates(today=None):
    """Return (Monday, Friday) of the *previous* trading week."""
    if today is None:
        today = date.today()

    days_since_monday = today.weekday()  # 0=Mon
    if days_since_monday == 0:
        prev_monday = today - timedelta(days=7)
    else:
        prev_monday = today - timedelta(days=days_since_monday + 7)

    prev_friday = prev_monday + timedelta(days=4)
    return prev_monday, prev_friday


def cache_path(monday, friday):
    """Return the CSV cache file path for a given week."""
    return DATA_DIR / f"ES_1m_{monday.isoformat()}_{friday.isoformat()}.csv"


def load_cached(filepath):
    """Load cached 1-min data from CSV."""
    df = pd.read_csv(filepath, parse_dates=["ts_event"], index_col="ts_event")
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def save_cache(df, filepath):
    """Save filtered session data to CSV for reuse."""
    DATA_DIR.mkdir(exist_ok=True)
    save_df = df.copy()
    # Convert index back to UTC for consistent storage
    save_df.index = save_df.index.tz_convert(TZ_UTC)
    save_df.index.name = "ts_event"
    save_df.to_csv(filepath)
    print(f"  Cached to: {filepath.name}")


def fetch_from_api(api_key, start, end):
    """Fetch 1-minute OHLCV from Databento for the given date range."""
    client = db.Historical(key=api_key)

    data = client.timeseries.get_range(
        dataset=DATASET,
        symbols=[SYMBOL],
        stype_in="continuous",
        schema=SCHEMA,
        start=str(start),
        end=str(end + timedelta(days=2)),
    )

    df = data.to_df()
    return df


def filter_session(df, monday, friday):
    """Filter DataFrame to Monday 18:00 ET -> Friday 16:00 ET (exclusive)."""
    if df.index.tz is None:
        df.index = df.index.tz_localize(TZ_UTC)
    df.index = df.index.tz_convert(TZ_ET)

    sess_start = datetime(monday.year, monday.month, monday.day, 18, 0, tzinfo=TZ_ET)
    sess_end   = datetime(friday.year, friday.month, friday.day, 15, 59, tzinfo=TZ_ET)

    mask = (df.index >= sess_start) & (df.index < sess_end)
    filtered = df.loc[mask].copy()

    if filtered.empty:
        print(f"WARNING: No data found between {sess_start} and {sess_end}")
        sys.exit(1)

    # Sanity check: expect ~5,700 bars for a full week (5 days × 22h × 60min)
    expected_bars = 5 * 22 * 60  # approximate
    bar_count = len(filtered)
    if abs(bar_count - expected_bars) > 120:  # tolerance of ~2 hours
        print(f"  ⚠ Bar count ({bar_count}) deviates from expected (~{expected_bars}).")
        print(f"    Possible DST shift or partial data — please verify.")

    print(f"  Session: {filtered.index[0]} -> {filtered.index[-1]}")
    print(f"  Bars:    {len(filtered)}")
    return filtered


def get_data(api_key, monday, friday):
    """Load from cache or fetch from API."""
    cp = cache_path(monday, friday)

    if cp.exists():
        print(f"\n[*] Found cached data: {cp.name}")
        df = load_cached(cp)
        df.index = df.index.tz_convert(TZ_ET)
        print(f"  Bars: {len(df)}")
        return df

    # No cache -- fetch from API
    if not api_key:
        print("ERROR: No cached data and no API key set.")
        print("  Set DATABENTO_API_KEY or run once with a key to cache data.")
        sys.exit(1)

    print("\n[*] No cached data found. Fetching from Databento...")
    raw_df = fetch_from_api(api_key, monday, friday)
    print(f"  Raw bars: {len(raw_df)}")

    print("\n[*] Filtering session (Mon 18:00 -> Fri 15:59 ET)...")
    filtered = filter_session(raw_df, monday, friday)

    # Save to cache
    save_cache(filtered, cp)
    return filtered


def build_volume_profile(df):
    """Distribute each bar's volume across its high-low range (matches TradingView).

    For each 1-min bar, volume is split evenly among all tick levels
    from round(low/tick)*tick to round(high/tick)*tick.

    Uses vectorized NumPy operations for speed.
    """
    from collections import defaultdict
    grid = defaultdict(float)

    highs  = df["high"].values
    lows   = df["low"].values
    vols   = df["volume"].values

    tick_his = np.rint(highs / TICK_SIZE).astype(np.int64)
    tick_los = np.rint(lows  / TICK_SIZE).astype(np.int64)
    n_ticks  = np.maximum(tick_his - tick_los + 1, 1)
    shares   = vols / n_ticks

    for i in range(len(df)):
        for t in range(tick_los[i], tick_his[i] + 1):
            grid[t * TICK_SIZE] += shares[i]

    profile = pd.Series(grid).sort_index()
    profile.index.name = "price_bin"
    return profile


def calc_value_area(profile, va_pct):
    """Calculate POC, VAH, VAL from a volume profile Series."""
    poc = profile.idxmax()

    total_vol = profile.sum()
    target = total_vol * va_pct

    prices = profile.index.tolist()
    poc_idx = prices.index(poc)

    lo = poc_idx
    hi = poc_idx
    running_vol = profile.iloc[poc_idx]
    last_idx = len(prices) - 1

    while running_vol < target:
        # Guard: stop if both boundaries are fully exhausted
        can_go_up   = hi < last_idx
        can_go_down = lo > 0

        if not can_go_up and not can_go_down:
            break

        vol_above = profile.iloc[hi + 1] if can_go_up   else 0.0
        vol_below = profile.iloc[lo - 1] if can_go_down else 0.0

        if vol_above >= vol_below:
            hi += 1
            running_vol += vol_above
        else:
            lo -= 1
            running_vol += vol_below

    vah = prices[hi]
    val = prices[lo]
    return poc, vah, val


def main():
    # -- Args ------------------------------------------------
    parser = argparse.ArgumentParser(description="Weekly Volume Profile (ES)")
    parser.add_argument("--va", type=float, nargs="+", default=[70.0, 68.0],
                        help="Value Area percentage(s) (default: 70 68)")
    parser.add_argument("--telegram", action="store_true",
                        help="Send results to Telegram")
    args = parser.parse_args()

    # -- API Key ---------------------------------------------
    api_key = os.environ.get("DATABENTO_API_KEY")

    # -- Date Range ------------------------------------------
    monday, friday = get_previous_week_dates()
    va_label = "  ".join(f"{v:.0f}%" for v in args.va)
    print(f"\n{'=' * 50}")
    print(f"  Weekly Volume Profile - ES")
    print(f"  Week: {monday.strftime('%b %d')} -> {friday.strftime('%b %d, %Y')}")
    print(f"  Value Area: {va_label}")
    print(f"{'=' * 50}")

    # -- Get Data (cache or API) -----------------------------
    df = get_data(api_key, monday, friday)

    # -- Profile ---------------------------------------------
    print(f"\n[*] Building volume profile...")
    profile = build_volume_profile(df)
    print(f"  Price levels: {len(profile)}")

    # -- Results Storage -------------------------------------
    va_results = {}

    # -- Value Area (for each VA%) ---------------------------
    for va in args.va:
        va_pct = va / 100.0
        poc, vah, val = calc_value_area(profile, va_pct)
        va_results[va] = (poc, vah, val)

        print(f"\n{'-' * 50}")
        print(f"  RESULTS  (VA = {va:.0f}%)")
        print(f"{'-' * 50}")
        print(f"  VAH:  {vah:.2f}")
        print(f"  POC:  {poc:.2f}")
        print(f"  VAL:  {val:.2f}")
        print(f"{'-' * 50}")

    # -- Telegram Notification -------------------------------
    if args.telegram:
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        chat_id = os.environ.get("TELEGRAM_CHAT_ID")

        if not bot_token or not chat_id:
            print("\n[!] ERROR: TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID not set.")
            return

        print("\n[*] Sending results to Telegram...")
        message = format_vp_message(monday, friday, SYMBOL, va_results)
        success = send_telegram_message(bot_token, chat_id, message)
        
        if success:
            print("  Telegram message sent successfully!")
        else:
            print("  Failed to send Telegram message.")


if __name__ == "__main__":
    main()
