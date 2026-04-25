"""
preprocess.py — Feature engineering for the M5 Walmart dataset.

Responsibility: Transform raw sales, calendar, and price data into
per-item-store feature matrices suitable for GNN training.

Feature groups
--------------
- Demand features: rolling mean, std, CV over 7/14/28 day windows
- Stockout features: zero_rate, max consecutive zeros, zero transitions
- Price features: current price, price change count, price volatility
- Calendar features: weekend_ratio, SNAP sensitivity, event sensitivity
- Trend features: linear trend slope over last 90 days
- Hierarchy encoding: one-hot for state/dept/cat
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

np.random.seed(42)

PROCESSED_DATA_DIR: Path = Path("data/processed")

# M5 hierarchy columns
META_COLUMNS: List[str] = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]

# Feature window sizes
ROLLING_WINDOWS: List[int] = [7, 14, 28]

# M5 evaluation: predict last 28 days
FORECAST_HORIZON: int = 28


def _compute_rolling_features(
    sales_row: np.ndarray,
    windows: List[int],
) -> Dict[str, float]:
    """Compute rolling demand statistics for a single item-store time series.

    Parameters
    ----------
    sales_row : np.ndarray
        1D array of daily sales (length = number of days).
    windows : list[int]
        Rolling window sizes.

    Returns
    -------
    dict
        Feature name -> value. Uses the last available window.
    """
    features: Dict[str, float] = {}
    series = pd.Series(sales_row, dtype=float)

    for w in windows:
        roll = series.iloc[-(w + FORECAST_HORIZON):-FORECAST_HORIZON] if len(series) > w + FORECAST_HORIZON else series.iloc[-w:]
        mean_val = float(roll.mean())
        std_val = float(roll.std()) if len(roll) > 1 else 0.0
        cv = std_val / (mean_val + 1e-8)

        features[f"demand_mean_{w}d"] = mean_val
        features[f"demand_std_{w}d"] = std_val
        features[f"demand_cv_{w}d"] = cv

    return features


def _compute_stockout_features(sales_row: np.ndarray) -> Dict[str, float]:
    """Compute stockout-related features for a single item-store time series.

    Parameters
    ----------
    sales_row : np.ndarray
        1D array of daily sales.

    Returns
    -------
    dict
        zero_rate, max_zero_streak, zero_transitions.
    """
    n = len(sales_row)
    is_zero = (sales_row == 0).astype(int)
    zero_rate = float(is_zero.mean())

    # Max consecutive zeros
    max_streak = 0
    current_streak = 0
    transitions = 0
    prev_zero = False

    for i in range(n):
        if is_zero[i]:
            current_streak += 1
            max_streak = max(max_streak, current_streak)
            if not prev_zero and i > 0:
                transitions += 1
            prev_zero = True
        else:
            current_streak = 0
            prev_zero = False

    return {
        "zero_rate": zero_rate,
        "max_zero_streak": float(max_streak),
        "zero_transitions": float(transitions),
    }


def _compute_price_features(
    prices_df: pd.DataFrame,
    store_id: str,
    item_id: str,
) -> Dict[str, float]:
    """Compute price features for a single item-store pair.

    Parameters
    ----------
    prices_df : pd.DataFrame
        Full sell_prices DataFrame.
    store_id : str
        Store identifier.
    item_id : str
        Item identifier.

    Returns
    -------
    dict
        price_current, price_mean, price_changes, price_cv.
    """
    mask = (prices_df["store_id"] == store_id) & (prices_df["item_id"] == item_id)
    item_prices = prices_df.loc[mask, "sell_price"]

    if len(item_prices) == 0:
        return {
            "price_current": 0.0,
            "price_mean": 0.0,
            "price_changes": 0.0,
            "price_cv": 0.0,
        }

    current = float(item_prices.iloc[-1])
    mean_p = float(item_prices.mean())
    std_p = float(item_prices.std()) if len(item_prices) > 1 else 0.0
    changes = float((item_prices.diff().abs() > 0).sum())

    return {
        "price_current": current,
        "price_mean": mean_p,
        "price_changes": changes,
        "price_cv": std_p / (mean_p + 1e-8),
    }


def _compute_trend_slope(sales_row: np.ndarray, window: int = 90) -> float:
    """Compute linear trend slope over the last `window` days.

    Parameters
    ----------
    sales_row : np.ndarray
        1D array of daily sales.
    window : int
        Number of days to compute trend over.

    Returns
    -------
    float
        Slope of the linear fit (positive = growing demand).
    """
    end_idx = len(sales_row) - FORECAST_HORIZON
    start_idx = max(0, end_idx - window)
    segment = sales_row[start_idx:end_idx].astype(float)

    if len(segment) < 2:
        return 0.0

    x = np.arange(len(segment), dtype=float)
    # Simple least-squares slope
    x_mean = x.mean()
    y_mean = segment.mean()
    denom = float(np.sum((x - x_mean) ** 2))
    if denom < 1e-8:
        return 0.0
    slope = float(np.sum((x - x_mean) * (segment - y_mean)) / denom)
    return slope


def _compute_calendar_features(
    sales_row: np.ndarray,
    calendar_df: pd.DataFrame,
    state_id: str,
) -> Dict[str, float]:
    """Compute calendar-derived features for a single item-store.

    Parameters
    ----------
    sales_row : np.ndarray
        1D array of daily sales aligned with calendar days d_1..d_N.
    calendar_df : pd.DataFrame
        Calendar DataFrame.
    state_id : str
        State ID for SNAP column selection.

    Returns
    -------
    dict
        weekend_ratio, snap_sensitivity, event_sensitivity.
    """
    n = min(len(sales_row), len(calendar_df))

    # Weekend ratio
    wday = calendar_df["wday"].values[:n]
    is_weekend = (wday == 1) | (wday == 2)  # Saturday=1, Sunday=2
    weekend_sales = float(sales_row[:n][is_weekend].mean()) if is_weekend.any() else 0.0
    weekday_sales = float(sales_row[:n][~is_weekend].mean()) if (~is_weekend).any() else 0.0
    weekend_ratio = weekend_sales / (weekday_sales + 1e-8)

    # SNAP sensitivity
    snap_col = f"snap_{state_id}"
    if snap_col in calendar_df.columns:
        is_snap = calendar_df[snap_col].values[:n] == 1
        snap_sales = float(sales_row[:n][is_snap].mean()) if is_snap.any() else 0.0
        non_snap_sales = float(sales_row[:n][~is_snap].mean()) if (~is_snap).any() else 0.0
        snap_sensitivity = (snap_sales - non_snap_sales) / (non_snap_sales + 1e-8)
    else:
        snap_sensitivity = 0.0

    # Event sensitivity
    has_event = calendar_df["event_name_1"].notna().values[:n]
    event_sales = float(sales_row[:n][has_event].mean()) if has_event.any() else 0.0
    no_event_sales = float(sales_row[:n][~has_event].mean()) if (~has_event).any() else 0.0
    event_sensitivity = (event_sales - no_event_sales) / (no_event_sales + 1e-8)

    return {
        "weekend_ratio": weekend_ratio,
        "snap_sensitivity": snap_sensitivity,
        "event_sensitivity": event_sensitivity,
    }


def preprocess(
    dfs: Dict[str, pd.DataFrame],
    output_filename: str = "m5_features.csv",
    max_items: int = 0,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Run the full M5 preprocessing pipeline.

    Steps (in order):
    1. Extract metadata and daily sales matrix.
    2. Compute per-item-store demand features (rolling stats).
    3. Compute stockout features (zero rate, streaks).
    4. Compute price features.
    5. Compute calendar features (weekend, SNAP, events).
    6. Compute trend slope.
    7. Normalize all numeric features with min-max scaling.
    8. Save result.

    Parameters
    ----------
    dfs : dict
        Dict with 'sales', 'calendar', 'prices' DataFrames.
    output_filename : str
        Name for the saved CSV file.
    max_items : int
        If > 0, limit processing to first N rows (for testing).

    Returns
    -------
    df_features : pd.DataFrame
        Feature matrix with one row per item-store (30,490 rows).
    scalers : dict
        Min/max scaler parameters for each feature column.
    """
    sales = dfs["sales"]
    calendar = dfs["calendar"]
    prices = dfs["prices"]

    if max_items > 0:
        sales = sales.head(max_items)

    day_cols = [c for c in sales.columns if c.startswith("d_")]
    sales_matrix = sales[day_cols].values
    n_rows = len(sales)

    print(f"[preprocess] Starting pipeline - {n_rows:,} item-store pairs, {len(day_cols)} days")

    # Pre-index prices for fast lookup
    print("[preprocess] Indexing prices...")
    prices_indexed = prices.set_index(["store_id", "item_id"])

    all_features: List[Dict] = []

    for idx in range(n_rows):
        if idx % 5000 == 0:
            print(f"[preprocess]   Processing row {idx:,}/{n_rows:,}...")

        row_sales = sales_matrix[idx]
        item_id = str(sales.iloc[idx]["item_id"])
        store_id = str(sales.iloc[idx]["store_id"])
        dept_id = str(sales.iloc[idx]["dept_id"])
        cat_id = str(sales.iloc[idx]["cat_id"])
        state_id = str(sales.iloc[idx]["state_id"])

        feats: Dict[str, float] = {}

        # Metadata (kept as strings for mapping, not features)
        feats["item_id"] = item_id
        feats["store_id"] = store_id
        feats["dept_id"] = dept_id
        feats["cat_id"] = cat_id
        feats["state_id"] = state_id

        # 1. Rolling demand features
        feats.update(_compute_rolling_features(row_sales, ROLLING_WINDOWS))

        # 2. Stockout features
        feats.update(_compute_stockout_features(row_sales))

        # 3. Price features (fast lookup)
        try:
            item_prices = prices_indexed.loc[(store_id, item_id), "sell_price"]
            if isinstance(item_prices, pd.Series):
                current = float(item_prices.iloc[-1])
                mean_p = float(item_prices.mean())
                std_p = float(item_prices.std()) if len(item_prices) > 1 else 0.0
                changes = float((item_prices.diff().abs() > 0).sum())
            else:
                current = float(item_prices)
                mean_p = current
                std_p = 0.0
                changes = 0.0
            feats["price_current"] = current
            feats["price_mean"] = mean_p
            feats["price_changes"] = changes
            feats["price_cv"] = std_p / (mean_p + 1e-8)
        except KeyError:
            feats["price_current"] = 0.0
            feats["price_mean"] = 0.0
            feats["price_changes"] = 0.0
            feats["price_cv"] = 0.0

        # 4. Calendar features
        feats.update(_compute_calendar_features(row_sales, calendar, state_id))

        # 5. Trend slope
        feats["trend_slope"] = _compute_trend_slope(row_sales)

        # 6. Overall demand level
        feats["total_sales"] = float(row_sales.sum())
        feats["demand_mean_all"] = float(row_sales.mean())

        all_features.append(feats)

    df_features = pd.DataFrame(all_features)

    # Separate metadata from numeric features
    meta_cols = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
    numeric_cols = [c for c in df_features.columns if c not in meta_cols]

    # Min-max normalize numeric features
    print("[preprocess] Normalizing features...")
    scalers: Dict[str, Dict[str, float]] = {}
    for col in numeric_cols:
        col_min = float(df_features[col].min())
        col_max = float(df_features[col].max())
        scalers[col] = {"min": col_min, "max": col_max}
        col_range = col_max - col_min
        if col_range == 0.0:
            df_features[col] = 0.0
        else:
            df_features[col] = (df_features[col] - col_min) / col_range

    # Save
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DATA_DIR / output_filename
    df_features.to_csv(out_path, index=False)
    print(f"[preprocess] Saved {out_path} (shape {df_features.shape})")
    print(f"[preprocess] Feature columns ({len(numeric_cols)}): {numeric_cols}")

    return df_features, scalers


def main() -> None:
    """Entry point for standalone preprocessing."""
    from src.data.loader import load_m5
    dfs = load_m5()
    df_features, scalers = preprocess(dfs)
    print(f"\n[preprocess] Done. Feature matrix shape: {df_features.shape}")


if __name__ == "__main__":
    main()
