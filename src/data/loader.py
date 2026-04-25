"""
loader.py — Load the M5 Walmart dataset (sales, calendar, sell_prices).

Responsibility: multi-file CSV ingestion with basic diagnostics.
Adapted from DTNet industrial loader for M5 retail data.
"""

import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

np.random.seed(42)

# Resolve from __file__ so path is correct regardless of CWD.
_PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR: Path = _PROJECT_ROOT / "data" / "raw"

# M5 file names
SALES_FILE: str = "sales_train_evaluation.csv"
CALENDAR_FILE: str = "calendar.csv"
PRICES_FILE: str = "sell_prices.csv"


def load_m5() -> Dict[str, pd.DataFrame]:
    """Load all three M5 CSV files and return as a dict of DataFrames.

    Returns
    -------
    dict
        Keys: 'sales', 'calendar', 'prices'.
        - sales: (30490, 1947) — item metadata + 1941 daily sales columns
        - calendar: (1969, 14) — date, events, SNAP flags
        - prices: (~6.8M, 4) — weekly sell prices per store-item
    """
    dfs: Dict[str, pd.DataFrame] = {}

    for key, filename in [
        ("sales", SALES_FILE),
        ("calendar", CALENDAR_FILE),
        ("prices", PRICES_FILE),
    ]:
        filepath: Path = RAW_DATA_DIR / filename
        if not filepath.exists():
            raise FileNotFoundError(f"[loader] File not found: {filepath}")
        print(f"[loader] Loading '{filename}'...")
        dfs[key] = pd.read_csv(filepath)
        print(f"[loader]   -> shape {dfs[key].shape}")

    return dfs


def print_basic_info(dfs: Dict[str, pd.DataFrame]) -> None:
    """Print a structured diagnostic summary of the M5 dataset.

    Parameters
    ----------
    dfs : dict
        Dict of DataFrames as returned by ``load_m5()``.
    """
    print(f"\n{'='*60}")
    print(f"  M5 DATASET SUMMARY")
    print(f"{'='*60}")

    sales = dfs["sales"]
    cal = dfs["calendar"]
    prices = dfs["prices"]

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    meta_cols = [c for c in sales.columns if not c.startswith("d_")]

    print(f"\n  Sales:")
    print(f"    Shape        : {sales.shape[0]:,} rows x {sales.shape[1]:,} columns")
    print(f"    Metadata cols: {meta_cols}")
    print(f"    Day columns  : {len(day_cols)} ({day_cols[0]} to {day_cols[-1]})")
    for col in meta_cols:
        print(f"    {col:<15}: {sales[col].nunique():>6,} unique")

    print(f"\n  Calendar:")
    print(f"    Shape     : {cal.shape}")
    print(f"    Date range: {cal['date'].min()} to {cal['date'].max()}")
    print(f"    Events    : {cal['event_name_1'].dropna().nunique()} unique events")
    print(f"    SNAP days : CA={cal['snap_CA'].sum()}, TX={cal['snap_TX'].sum()}, WI={cal['snap_WI'].sum()}")

    print(f"\n  Prices:")
    print(f"    Shape     : {prices.shape[0]:,} rows")
    print(f"    Price range: ${prices['sell_price'].min():.2f} - ${prices['sell_price'].max():.2f}")
    print(f"    Mean price : ${prices['sell_price'].mean():.2f}")

    print(f"{'='*60}\n")


def main() -> None:
    """Entry point for standalone inspection."""
    dfs = load_m5()
    print_basic_info(dfs)
    print("[loader] Done.")


if __name__ == "__main__":
    main()
