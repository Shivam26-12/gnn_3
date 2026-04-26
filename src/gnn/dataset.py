"""dataset.py — Item-store level sliding-window DataLoaders for M5.

Each node = one item-store pair (30,490 nodes).
Features: 25 per node (lags, rolling stats, price, calendar, encodings).
Target: 28-day sales forecast per node.
Optimized with vectorized numpy for speed.
"""
from __future__ import annotations
import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data

try:
    from torch_geometric.loader import DataLoader
except ImportError:
    from torch_geometric.data import DataLoader

np.random.seed(42)
torch.manual_seed(42)

FORECAST_HORIZON: int = 28
FEATURE_WINDOW: int = 28
STRIDE: int = 7
BATCH_SIZE: int = 4
TRAIN_RATIO: float = 0.70
VAL_RATIO: float = 0.15


def _build_edge_index_and_attrs(
    sales_df: pd.DataFrame,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build edges: same-item-cross-store (same state) + same-store-same-dept."""
    n = len(sales_df)
    item_ids = sales_df["item_id"].values
    store_ids = sales_df["store_id"].values
    dept_ids = sales_df["dept_id"].values
    state_ids = sales_df["state_id"].values

    # Index maps
    item_to_rows: Dict[str, List[int]] = {}
    store_dept_to_rows: Dict[str, List[int]] = {}
    for i in range(n):
        item_to_rows.setdefault(str(item_ids[i]), []).append(i)
        key = f"{store_ids[i]}_{dept_ids[i]}"
        store_dept_to_rows.setdefault(key, []).append(i)

    src_list, dst_list, attr_list = [], [], []

    # Cross-store & inter-state edges
    print("[dataset] Building cross-store & inter-state edges...")
    for item_id, rows in item_to_rows.items():
        if len(rows) <= 1:
            continue
        states_for_rows = [str(state_ids[r]) for r in rows]
        stores_for_rows = [str(store_ids[r]) for r in rows]
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                src_list.extend([rows[i], rows[j]])
                dst_list.extend([rows[j], rows[i]])
                if states_for_rows[i] == states_for_rows[j]:
                    # Same state: geographic substitution (criticality 0.7)
                    attr_list.extend([[0.7, 1.0, 1.0], [0.7, 1.0, 1.0]])
                else:
                    # Different state: logistics balancing (criticality 0.2)
                    attr_list.extend([[0.2, 1.0, 1.0], [0.2, 1.0, 1.0]])

    n_cross = len(src_list)
    print(f"[dataset]   Cross-store + Inter-state edges: {n_cross}")

    # Same-store-same-dept: connect each item to 5 random neighbors in same store-dept
    # (full connectivity would be too dense)
    print("[dataset] Building intra-dept edges...")
    rng = np.random.default_rng(42)
    K_NEIGHBORS = 5
    for key, rows in store_dept_to_rows.items():
        if len(rows) <= 1:
            continue
        for r in rows:
            neighbors = rng.choice(
                [x for x in rows if x != r],
                size=min(K_NEIGHBORS, len(rows) - 1),
                replace=False,
            )
            for nb in neighbors:
                src_list.append(r)
                dst_list.append(nb)
                attr_list.append([0.4, 0.5, 0.5])

    print(f"[dataset]   Total edges: {len(src_list)}")
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(attr_list, dtype=torch.float32)
    return edge_index, edge_attr


def _precompute_features(
    sales_matrix: np.ndarray,
    prices_df: pd.DataFrame,
    calendar_df: pd.DataFrame,
    sales_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized feature computation for all item-store pairs.

    Returns:
        rolling_feats: (N, T, 15) time-varying features
        static_feats: (N, 10) static features
        snap_matrix: (N, T) SNAP flags per item-store per day
    """
    N, T = sales_matrix.shape
    print(f"[dataset] Pre-computing features for {N:,} series x {T:,} days...")

    # --- Static features ---
    store_ids = sales_df["store_id"].values
    dept_ids = sales_df["dept_id"].values
    cat_ids = sales_df["cat_id"].values
    state_ids = sales_df["state_id"].values

    unique_stores = sorted(set(store_ids))
    unique_depts = sorted(set(dept_ids))
    store_map = {s: i / max(len(unique_stores) - 1, 1) for i, s in enumerate(unique_stores)}
    dept_map = {d: i / max(len(unique_depts) - 1, 1) for i, d in enumerate(unique_depts)}

    overall_mean = sales_matrix.mean(axis=1)
    overall_std = sales_matrix.std(axis=1)
    overall_cv = overall_std / (overall_mean + 1e-8)
    overall_zero_rate = (sales_matrix == 0).mean(axis=1)

    static_feats = np.zeros((N, 10), dtype=np.float32)
    for i in range(N):
        static_feats[i, 0] = overall_mean[i]
        static_feats[i, 1] = overall_cv[i]
        static_feats[i, 2] = overall_zero_rate[i]
        static_feats[i, 3] = store_map.get(str(store_ids[i]), 0.0)
        static_feats[i, 4] = dept_map.get(str(dept_ids[i]), 0.0)
        static_feats[i, 5] = 1.0 if str(cat_ids[i]) == "FOODS" else 0.0
        static_feats[i, 6] = 1.0 if str(cat_ids[i]) == "HOBBIES" else 0.0
        static_feats[i, 7] = 1.0 if str(cat_ids[i]) == "HOUSEHOLD" else 0.0
        static_feats[i, 8] = 1.0 if str(state_ids[i]) in ("CA",) else 0.0
        static_feats[i, 9] = 1.0 if str(state_ids[i]) in ("TX",) else 0.0

    # --- Price lookup & Discount Depth ---
    print("[dataset]   Computing price matrix & discount depth...")
    max_price_map = prices_df.groupby(["store_id", "item_id"])["sell_price"].max().to_dict()
    max_price_arr = np.zeros(N, dtype=np.float32)
    item_ids = sales_df["item_id"].values
    for i in range(N):
        p = max_price_map.get((str(store_ids[i]), str(item_ids[i])), 0.0)
        max_price_arr[i] = float(p) + 1e-8

    discount_matrix = np.ones((N, T), dtype=np.float32)
    series_to_idx = { (str(store_ids[i]), str(item_ids[i])): i for i in range(N) }
    
    wk_array = calendar_df['wm_yr_wk'].values[:T]
    wk_to_t = {}
    for t_idx, wk in enumerate(wk_array):
        wk_to_t.setdefault(wk, []).append(t_idx)
        
    for row in prices_df.itertuples(index=False):
        idx = series_to_idx.get((str(row.store_id), str(row.item_id)))
        if idx is not None and row.wm_yr_wk in wk_to_t:
            ts = wk_to_t[row.wm_yr_wk]
            discount_matrix[idx, ts] = float(row.sell_price) / max_price_arr[idx]

    # --- SNAP matrix ---
    snap_cols = {"CA": "snap_CA", "TX": "snap_TX", "WI": "snap_WI"}
    snap_matrix = np.zeros((N, T), dtype=np.float32)
    for i in range(N):
        st = str(state_ids[i])
        col = snap_cols.get(st)
        if col and col in calendar_df.columns:
            snap_vals = calendar_df[col].values[:T]
            snap_matrix[i, :len(snap_vals)] = snap_vals

    # --- Consecutive Zeros ---
    print("[dataset]   Computing consecutive zero sales...")
    sf = sales_matrix.astype(np.float32)
    is_zero = (sf == 0).astype(np.float32)
    consec_zeros = np.zeros((N, T), dtype=np.float32)
    for t_idx in range(1, T):
        consec_zeros[:, t_idx] = (consec_zeros[:, t_idx-1] + 1) * is_zero[:, t_idx]

    # --- Event Countdowns & Encodings ---
    print("[dataset]   Computing event countdowns and categorical flags...")
    days_until_event = np.zeros(T, dtype=np.float32)
    is_sporting = np.zeros(T, dtype=np.float32)
    is_cultural = np.zeros(T, dtype=np.float32)
    is_national = np.zeros(T, dtype=np.float32)
    is_religious = np.zeros(T, dtype=np.float32)
    
    last_event_t = 9999
    for t_idx in range(T-1, -1, -1):
        evt_type = calendar_df.iloc[t_idx].get("event_type_1")
        if pd.notna(evt_type):
            last_event_t = t_idx
            if evt_type == "Sporting": is_sporting[t_idx] = 1.0
            elif evt_type == "Cultural": is_cultural[t_idx] = 1.0
            elif evt_type == "National": is_national[t_idx] = 1.0
            elif evt_type == "Religious": is_religious[t_idx] = 1.0
        days_until_event[t_idx] = max(0, last_event_t - t_idx)
        
    days_until_event = days_until_event / (days_until_event.max() + 1e-8)
    day_of_month = pd.to_datetime(calendar_df["date"]).dt.day.values[:T] / 31.0

    # --- Rolling features (vectorized) ---
    print("[dataset]   Computing rolling features...")
    cumsum = np.cumsum(np.pad(sf, ((0,0),(1,0))), axis=1)
    cumsq = np.cumsum(np.pad(sf**2, ((0,0),(1,0))), axis=1)

    # Expanded to 21 channels for advanced STGNN features
    rolling_feats = np.zeros((N, T, 21), dtype=np.float32)

    for t in range(FEATURE_WINDOW, T):
        # Lags
        rolling_feats[:, t, 0] = sf[:, t-1]       # lag1
        rolling_feats[:, t, 1] = sf[:, t-7] if t >= 7 else 0     # lag7
        rolling_feats[:, t, 2] = sf[:, t-14] if t >= 14 else 0   # lag14
        rolling_feats[:, t, 3] = sf[:, t-28] if t >= 28 else 0   # lag28

        # Rolling mean/std for 7, 14, 28 days
        for wi, w in enumerate([7, 14, 28]):
            if t >= w:
                s = cumsum[:, t] - cumsum[:, t-w]
                sq = cumsq[:, t] - cumsq[:, t-w]
                mean = s / w
                var = sq / w - mean**2
                var = np.maximum(var, 0)
                rolling_feats[:, t, 4 + wi] = mean
                rolling_feats[:, t, 7 + wi] = np.sqrt(var)

        # Zero rate last 28 days
        if t >= 28:
            rolling_feats[:, t, 10] = (sf[:, t-28:t] == 0).mean(axis=1)

        # 11: Discount Depth
        rolling_feats[:, t, 11] = discount_matrix[:, t]

        # 12: SNAP
        rolling_feats[:, t, 12] = snap_matrix[:, t]

        # 13: Days Until Event
        rolling_feats[:, t, 13] = days_until_event[t]

        # 14: Day of Month
        rolling_feats[:, t, 14] = day_of_month[t]

        # 15: Consecutive Zeros
        rolling_feats[:, t, 15] = consec_zeros[:, t]

        # 16-19: Event Types
        rolling_feats[:, t, 16] = is_sporting[t]
        rolling_feats[:, t, 17] = is_cultural[t]
        rolling_feats[:, t, 18] = is_national[t]
        rolling_feats[:, t, 19] = is_religious[t]

        # 20: Day of week (normalized)
        if t < len(calendar_df):
            rolling_feats[:, t, 20] = calendar_df.iloc[t]["wday"] / 7.0

    print("[dataset]   Feature pre-computation complete.")
    return rolling_feats, static_feats, snap_matrix


def build_dataloaders(
    sales_df: pd.DataFrame,
    G: nx.DiGraph,
    calendar_df: pd.DataFrame = None,
    prices_df: pd.DataFrame = None,
    batch_size: int = BATCH_SIZE,
    forecast_horizon: int = FORECAST_HORIZON,
    feature_window: int = FEATURE_WINDOW,
    stride: int = STRIDE,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train/val/test loaders with item-store nodes and rich features."""
    day_cols = [c for c in sales_df.columns if c.startswith("d_")]
    sales_matrix = sales_df[day_cols].values.astype(np.float32)
    N, T = sales_matrix.shape

    # Build edges
    edge_index, edge_attr = _build_edge_index_and_attrs(sales_df)

    # Pre-compute features
    if calendar_df is None or prices_df is None:
        from src.data.loader import load_m5
        dfs = load_m5()
        if calendar_df is None:
            calendar_df = dfs["calendar"]
        if prices_df is None:
            prices_df = dfs["prices"]

    rolling_feats, static_feats, _ = _precompute_features(
        sales_matrix, prices_df, calendar_df, sales_df
    )

    # Generate exact M5 splits based on the competition timeline
    # M5 Evaluation Test set (d_1914 to d_1941): Target starts at T - 28
    test_pos = T - forecast_horizon
    # Validation set (d_1886 to d_1913): Target starts at T - 56
    val_pos = T - 2 * forecast_horizon
    
    train_windows = []
    pos = feature_window
    # Train windows end strictly before the validation target starts
    while pos + forecast_horizon <= val_pos:
        train_windows.append(pos)
        pos += stride

    val_windows = [val_pos]
    test_windows = [test_pos]
    windows = train_windows + val_windows + test_windows

    n_windows = len(windows)
    n_features = rolling_feats.shape[2] + static_feats.shape[1]  # 15 + 10 = 25
    print(f"[dataset] Windows: {n_windows}, Features: {n_features}")

    # Build Data objects
    all_data: List[Data] = []
    for wi, t in enumerate(windows):
        if wi % 50 == 0:
            print(f"[dataset]   Building window {wi}/{n_windows}...")
        # Extract sequences for the STGNN
        x_seq = rolling_feats[:, t - feature_window:t, :]  # (N, seq_len, 15)
        x_static = static_feats                            # (N, 10)
        y = sales_matrix[:, t:t + forecast_horizon]        # (N, 28)
        y_hist = sales_matrix[:, t - feature_window:t]     # (N, seq_len)

        all_data.append(Data(
            x_seq=torch.from_numpy(x_seq.copy()),
            x_static=torch.from_numpy(x_static.copy()),
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.from_numpy(y.copy()),
            y_hist=torch.from_numpy(y_hist.copy()),
        ))

    # Exact M5 split
    n_train = len(train_windows)
    n_val = len(val_windows)
    train_data = all_data[:n_train]
    val_data = all_data[n_train:n_train + n_val]
    test_data = all_data[n_train + n_val:]

    print(f"[dataset] Split: train={len(train_data)} windows, val={len(val_data)} (d_{val_pos+1}-d_{test_pos}), test={len(test_data)} (d_{test_pos+1}-d_{T})")

    # Normalize features (fit on train only)
    # We fit separate scalers for sequence and static features
    train_x_seq = torch.cat([d.x_seq for d in train_data], dim=0).numpy() # (N_train_windows * N, seq_len, 15)
    train_x_static = torch.cat([d.x_static for d in train_data], dim=0).numpy() # (N_train_windows * N, 10)
    
    # Flatten sequence for scaler
    flat_train_x_seq = train_x_seq.reshape(-1, train_x_seq.shape[-1])
    
    scaler_seq = StandardScaler()
    scaler_seq.fit(flat_train_x_seq)
    scaler_seq.scale_ = np.where(scaler_seq.scale_ == 0, 1.0, scaler_seq.scale_)
    
    scaler_static = StandardScaler()
    scaler_static.fit(train_x_static)
    scaler_static.scale_ = np.where(scaler_static.scale_ == 0, 1.0, scaler_static.scale_)

    for d in all_data:
        # Scale sequence
        shape = d.x_seq.shape
        flat_seq = d.x_seq.numpy().reshape(-1, shape[-1])
        scaled_seq = scaler_seq.transform(flat_seq).reshape(shape)
        d.x_seq = torch.from_numpy(scaled_seq.astype(np.float32))
        
        # Scale static
        scaled_static = scaler_static.transform(d.x_static.numpy())
        d.x_static = torch.from_numpy(scaled_static.astype(np.float32))

    # Save scalers
    scalers = {'seq': scaler_seq, 'static': scaler_static}
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    with open("data/processed/scaler.pkl", "wb") as f:
        pickle.dump(scalers, f)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
