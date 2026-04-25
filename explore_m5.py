"""
explore_m5.py — Deep exploration of the M5 Walmart dataset
to understand its structure for DTNet adaptation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter

np.random.seed(42)

# ── Paths ──────────────────────────────────────────────────────────────────
M5_DIR = Path(r"C:\Users\SHIVAM BHATT\Downloads\m5-forecasting-accuracy")

# ── 1. Calendar ────────────────────────────────────────────────────────────
print("=" * 70)
print("  1. CALENDAR DATA")
print("=" * 70)

cal = pd.read_csv(M5_DIR / "calendar.csv")
print(f"\n  Shape: {cal.shape}")
print(f"  Columns: {list(cal.columns)}")
print(f"\n  Date range: {cal['date'].min()} → {cal['date'].max()}")
print(f"  Unique weekdays: {cal['weekday'].nunique()} → {sorted(cal['weekday'].unique())}")

# Events
event_cols = [c for c in cal.columns if 'event' in c]
print(f"\n  Event columns: {event_cols}")
for col in event_cols:
    non_null = cal[col].dropna()
    print(f"    {col}: {len(non_null)} non-null entries, {non_null.nunique()} unique values")
    if non_null.nunique() <= 15:
        print(f"      Values: {sorted(non_null.unique())}")

# SNAP days
snap_cols = [c for c in cal.columns if 'snap' in c.lower()]
print(f"\n  SNAP columns: {snap_cols}")
for col in snap_cols:
    print(f"    {col}: {cal[col].sum()} SNAP days out of {len(cal)}")

print(f"\n  First 3 rows:")
print(cal.head(3).to_string(index=False))

# ── 2. Sales ───────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  2. SALES DATA")
print("=" * 70)

# Read just a chunk first to understand structure
sales = pd.read_csv(M5_DIR / "sales_train_evaluation.csv")
print(f"\n  Shape: {sales.shape[0]:,} rows × {sales.shape[1]:,} columns")

# Identify metadata vs day columns
meta_cols = [c for c in sales.columns if not c.startswith("d_")]
day_cols = [c for c in sales.columns if c.startswith("d_")]
print(f"  Metadata columns ({len(meta_cols)}): {meta_cols}")
print(f"  Day columns: {len(day_cols)} (from {day_cols[0]} to {day_cols[-1]})")

# ── Hierarchy analysis ─────────────────────────────────────────────────────
print(f"\n  ── Hierarchy ──")
for col in meta_cols:
    nuniq = sales[col].nunique()
    print(f"    {col:<15}: {nuniq:>6,} unique values")
    if nuniq <= 15:
        print(f"      → {sorted(sales[col].unique())}")

# ── Cross-tabulations ──────────────────────────────────────────────────────
print(f"\n  ── Cross-tabs ──")

# Items per store
items_per_store = sales.groupby("store_id")["item_id"].nunique()
print(f"\n  Items per store:")
for store, n_items in items_per_store.items():
    print(f"    {store}: {n_items:,} items")

# Items per department
items_per_dept = sales.groupby("dept_id")["item_id"].nunique()
print(f"\n  Items per department:")
for dept, n_items in items_per_dept.items():
    print(f"    {dept}: {n_items:,} items")

# Items per category
items_per_cat = sales.groupby("cat_id")["item_id"].nunique()
print(f"\n  Items per category:")
for cat, n_items in items_per_cat.items():
    print(f"    {cat}: {n_items:,} items")

# Stores per state
stores_per_state = sales.groupby("state_id")["store_id"].nunique()
print(f"\n  Stores per state:")
for state, n_stores in stores_per_state.items():
    print(f"    {state}: {n_stores} stores → {sorted(sales[sales['state_id']==state]['store_id'].unique())}")

# ── Sales statistics ───────────────────────────────────────────────────────
print(f"\n  ── Sales Statistics ──")

sales_matrix = sales[day_cols].values  # (items, days)
print(f"  Sales matrix shape: {sales_matrix.shape}")
print(f"  Total sales: {sales_matrix.sum():,.0f}")
print(f"  Mean daily sales per item: {sales_matrix.mean():.3f}")
print(f"  Median daily sales per item: {np.median(sales_matrix):.1f}")
print(f"  Max single-day sale: {sales_matrix.max()}")
print(f"  % zeros (potential stockouts): {(sales_matrix == 0).mean() * 100:.1f}%")

# Per-store aggregate
print(f"\n  Mean daily sales per store:")
for store in sorted(sales["store_id"].unique()):
    store_sales = sales[sales["store_id"] == store][day_cols].values
    print(f"    {store}: mean={store_sales.mean():.3f}  total={store_sales.sum():,.0f}  zero%={((store_sales==0).mean()*100):.1f}%")

# ── Zero-sales analysis (proxy for stockouts) ─────────────────────────────
print(f"\n  ── Zero-Sales / Stockout Analysis ──")

# Items with very high zero-rate
zero_rates = (sales_matrix == 0).mean(axis=1)  # per item
print(f"  Distribution of item-level zero-sales rate:")
for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
    n = (zero_rates > threshold).sum()
    print(f"    >{threshold*100:.0f}% zeros: {n:,} items ({n/len(zero_rates)*100:.1f}%)")

# Consecutive zeros (potential stockout runs)
print(f"\n  Consecutive zero-sales streaks (sampled 500 items):")
sample_idx = np.random.choice(len(sales_matrix), min(500, len(sales_matrix)), replace=False)
max_streaks = []
for idx in sample_idx:
    row = sales_matrix[idx]
    streak = 0
    max_s = 0
    for val in row:
        if val == 0:
            streak += 1
            max_s = max(max_s, streak)
        else:
            streak = 0
    max_streaks.append(max_s)

max_streaks = np.array(max_streaks)
print(f"    Mean max consecutive zeros: {max_streaks.mean():.1f} days")
print(f"    Median max consecutive zeros: {np.median(max_streaks):.0f} days")
print(f"    95th percentile: {np.percentile(max_streaks, 95):.0f} days")
print(f"    Max: {max_streaks.max()} days")

# ── 3. Sell Prices ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("  3. SELL PRICES")
print("=" * 70)

prices = pd.read_csv(M5_DIR / "sell_prices.csv")
print(f"\n  Shape: {prices.shape[0]:,} rows × {prices.shape[1]} columns")
print(f"  Columns: {list(prices.columns)}")
print(f"  Store-item combinations: {prices.groupby(['store_id','item_id']).ngroups:,}")
print(f"  Week IDs: {prices['wm_yr_wk'].min()} → {prices['wm_yr_wk'].max()} ({prices['wm_yr_wk'].nunique()} weeks)")
print(f"  Price range: ${prices['sell_price'].min():.2f} → ${prices['sell_price'].max():.2f}")
print(f"  Mean price: ${prices['sell_price'].mean():.2f}")
print(f"  Median price: ${prices['sell_price'].median():.2f}")

# Price changes per item
price_changes = prices.groupby(["store_id", "item_id"])["sell_price"].apply(
    lambda x: (x.diff().abs() > 0).sum()
)
print(f"\n  Price changes per store-item:")
print(f"    Mean: {price_changes.mean():.1f} changes")
print(f"    Median: {price_changes.median():.0f}")
print(f"    Max: {price_changes.max()}")
print(f"    Items with 0 price changes: {(price_changes == 0).sum():,} ({(price_changes==0).mean()*100:.1f}%)")

# ── 4. DTNet Graph Mapping Preview ─────────────────────────────────────────
print("\n" + "=" * 70)
print("  4. POTENTIAL DTNET GRAPH STRUCTURE")
print("=" * 70)

n_states = sales["state_id"].nunique()
n_stores = sales["store_id"].nunique()
n_depts = sales["dept_id"].nunique()
n_cats = sales["cat_id"].nunique()
n_items = sales["item_id"].nunique()

total_nodes = n_states + n_stores + n_depts + n_cats + n_items
print(f"\n  Potential node layers:")
print(f"    State (top-level)  : {n_states:>6,} nodes")
print(f"    Store              : {n_stores:>6,} nodes")
print(f"    Department         : {n_depts:>6,} nodes")
print(f"    Category           : {n_cats:>6,} nodes")
print(f"    Item (leaf-level)  : {n_items:>6,} nodes")
print(f"    ─────────────────────────────")
print(f"    TOTAL              : {total_nodes:>6,} nodes")

# Edge counts
state_store = sales[["state_id", "store_id"]].drop_duplicates()
store_dept = sales[["store_id", "dept_id"]].drop_duplicates()
dept_cat = sales[["dept_id", "cat_id"]].drop_duplicates()
cat_item = sales[["cat_id", "item_id"]].drop_duplicates()

# Cross-store edges: same item sold in multiple stores
item_stores = sales.groupby("item_id")["store_id"].apply(set)
items_in_multiple_stores = item_stores[item_stores.apply(len) > 1]
cross_store_edges = sum(len(stores) * (len(stores) - 1) // 2 for stores in items_in_multiple_stores)

print(f"\n  Potential edges:")
print(f"    State → Store      : {len(state_store):>6,} edges")
print(f"    Store → Department : {len(store_dept):>6,} edges")
print(f"    Department → Cat   : {len(dept_cat):>6,} edges")
print(f"    Category → Item    : {len(cat_item):>6,} edges")
print(f"    Cross-store (shared items): {cross_store_edges:>6,} edges")

# Items appearing across multiple stores
print(f"\n  Cross-store item sharing:")
store_counts = item_stores.apply(len)
for n in range(1, 11):
    cnt = (store_counts == n).sum()
    if cnt > 0:
        print(f"    Items in exactly {n} store(s): {cnt:,}")

print(f"\n  NOTE: Every item appears in {store_counts.min()}-{store_counts.max()} stores")

print("\n" + "=" * 70)
print("  EXPLORATION COMPLETE")
print("=" * 70)
