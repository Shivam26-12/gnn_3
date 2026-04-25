"""
explore_m5_deep.py - Deep EDA on M5 dataset focused on:
  1. Temporal patterns (weekly, monthly, yearly seasonality)
  2. Demand shock detection (sudden spikes/drops)
  3. Stockout clustering across stores
  4. Cross-store demand correlation
  5. Event & SNAP day impact
  6. Department-level dynamics
  7. Potential node feature engineering
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore")

np.random.seed(42)

M5_DIR = Path(r"C:\Users\SHIVAM BHATT\Downloads\m5-forecasting-accuracy")

print("Loading data...")
cal = pd.read_csv(M5_DIR / "calendar.csv")
sales = pd.read_csv(M5_DIR / "sales_train_evaluation.csv")
prices = pd.read_csv(M5_DIR / "sell_prices.csv")

day_cols = [c for c in sales.columns if c.startswith("d_")]
sales_matrix = sales[day_cols].values  # (30490, 1941)

# Map d_1..d_1941 to dates
d_to_date = dict(zip(cal["d"], pd.to_datetime(cal["date"])))
dates = pd.to_datetime([d_to_date[d] for d in day_cols])

# =========================================================================
print("\n" + "=" * 70)
print("  1. TEMPORAL PATTERNS")
print("=" * 70)

# Aggregate daily sales across all items
daily_total = sales_matrix.sum(axis=0)  # (1941,)
daily_mean = sales_matrix.mean(axis=0)

# Weekly pattern
day_of_week = dates.dayofweek  # 0=Mon, 6=Sun
weekly_avg = pd.Series(daily_total, index=dates).groupby(dates.dayofweek).mean()
dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
print("\n  Average total daily sales by day-of-week:")
for dow, val in weekly_avg.items():
    bar = "#" * int(val / weekly_avg.max() * 40)
    print(f"    {dow_names[dow]}: {val:>10,.0f}  {bar}")

# Monthly pattern
monthly_avg = pd.Series(daily_total, index=dates).groupby(dates.month).mean()
month_names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
print("\n  Average total daily sales by month:")
for m, val in monthly_avg.items():
    bar = "#" * int(val / monthly_avg.max() * 40)
    print(f"    {month_names[m-1]}: {val:>10,.0f}  {bar}")

# Yearly trend
yearly_avg = pd.Series(daily_total, index=dates).groupby(dates.year).mean()
print("\n  Average total daily sales by year:")
for y, val in yearly_avg.items():
    print(f"    {y}: {val:>10,.0f}")

# =========================================================================
print("\n" + "=" * 70)
print("  2. DEMAND SHOCK DETECTION")
print("=" * 70)

# For each store, compute rolling stats and detect shocks
print("\n  Detecting demand shocks (daily store sales deviating >3 sigma from 28-day rolling mean)...")

stores = sorted(sales["store_id"].unique())
store_shocks = {}

for store in stores:
    store_mask = sales["store_id"] == store
    store_daily = sales[store_mask][day_cols].sum(axis=0).values.astype(float)
    
    # 28-day rolling mean and std
    series = pd.Series(store_daily)
    roll_mean = series.rolling(28, min_periods=7).mean()
    roll_std = series.rolling(28, min_periods=7).std()
    
    # Z-score
    z = (series - roll_mean) / (roll_std + 1e-8)
    
    n_spikes = (z > 3).sum()
    n_drops = (z < -3).sum()
    store_shocks[store] = {"spikes": int(n_spikes), "drops": int(n_drops), 
                           "total": int(n_spikes + n_drops)}
    
print(f"\n  {'Store':<10} {'Spikes(>3sig)':>15} {'Drops(<-3sig)':>15} {'Total Shocks':>15}")
print(f"  {'-'*10} {'-'*15} {'-'*15} {'-'*15}")
for store in stores:
    s = store_shocks[store]
    print(f"  {store:<10} {s['spikes']:>15} {s['drops']:>15} {s['total']:>15}")

# =========================================================================
print("\n" + "=" * 70)
print("  3. STOCKOUT CLUSTERING ACROSS STORES")
print("=" * 70)

# For a sample of items, check: when item goes to 0 in one store,
# does it go to 0 in other stores around the same time?

print("\n  Analyzing simultaneous zero-sales across stores for sampled items...")

# Pick 200 medium-frequency items (not too sparse, not too dense)
item_means = sales.groupby("item_id")[day_cols].apply(lambda x: x.values.mean())
medium_items = item_means[(item_means > 0.5) & (item_means < 3.0)].index.tolist()
sample_items = np.random.choice(medium_items, min(200, len(medium_items)), replace=False)

sync_scores = []
for item_id in sample_items:
    item_data = sales[sales["item_id"] == item_id][day_cols].values  # (10, 1941) one row per store
    # For each day, count how many stores have zero sales
    zeros_per_day = (item_data == 0).sum(axis=0)  # (1941,)
    # Fraction of days where >50% of stores simultaneously have zero
    frac_sync_zero = (zeros_per_day >= 5).mean()
    sync_scores.append(frac_sync_zero)

sync_scores = np.array(sync_scores)
print(f"\n  Sampled {len(sample_items)} medium-frequency items")
print(f"  Fraction of days with synchronized zeros (>=5/10 stores zero):")
print(f"    Mean:   {sync_scores.mean():.3f}")
print(f"    Median: {np.median(sync_scores):.3f}")
print(f"    Min:    {sync_scores.min():.3f}")
print(f"    Max:    {sync_scores.max():.3f}")

# Distribution
for thresh in [0.1, 0.3, 0.5, 0.7]:
    n = (sync_scores > thresh).sum()
    print(f"    Items with >{thresh*100:.0f}% sync-zero days: {n}/{len(sync_scores)} ({n/len(sync_scores)*100:.1f}%)")

# =========================================================================
print("\n" + "=" * 70)
print("  4. CROSS-STORE DEMAND CORRELATION")
print("=" * 70)

print("\n  Computing pairwise store-level daily sales correlation...")

# Build store-level daily totals
store_daily = {}
for store in stores:
    store_mask = sales["store_id"] == store
    store_daily[store] = sales[store_mask][day_cols].sum(axis=0).values.astype(float)

store_df = pd.DataFrame(store_daily)
corr_matrix = store_df.corr()

print(f"\n  Store-to-store correlation matrix:")
print(f"  {'':>8}", end="")
for s in stores:
    print(f" {s:>7}", end="")
print()

for s1 in stores:
    print(f"  {s1:>8}", end="")
    for s2 in stores:
        val = corr_matrix.loc[s1, s2]
        print(f" {val:>7.3f}", end="")
    print()

# Same-state vs cross-state
state_map = {s: s.split("_")[0] for s in stores}
same_state_corrs = []
cross_state_corrs = []
for i, s1 in enumerate(stores):
    for j, s2 in enumerate(stores):
        if i >= j:
            continue
        val = corr_matrix.loc[s1, s2]
        if state_map[s1] == state_map[s2]:
            same_state_corrs.append(val)
        else:
            cross_state_corrs.append(val)

print(f"\n  Same-state avg correlation:  {np.mean(same_state_corrs):.4f}  (n={len(same_state_corrs)})")
print(f"  Cross-state avg correlation: {np.mean(cross_state_corrs):.4f}  (n={len(cross_state_corrs)})")

# =========================================================================
print("\n" + "=" * 70)
print("  5. EVENT & SNAP DAY IMPACT")
print("=" * 70)

# Merge calendar with daily totals
cal_slim = cal[["d", "date", "event_name_1", "event_type_1", "snap_CA", "snap_TX", "snap_WI"]].copy()
cal_slim["daily_total"] = [daily_total[i] if i < len(daily_total) else np.nan 
                           for i in range(len(cal_slim))]
cal_slim = cal_slim.dropna(subset=["daily_total"])

baseline_mean = cal_slim["daily_total"].mean()

# SNAP impact per state
print("\n  SNAP day impact on sales:")
for state, snap_col in [("CA", "snap_CA"), ("TX", "snap_TX"), ("WI", "snap_WI")]:
    # Get store-level daily totals
    state_stores = [s for s in stores if s.startswith(state)]
    state_daily = np.zeros(len(day_cols))
    for store in state_stores:
        store_mask = sales["store_id"] == store
        state_daily += sales[store_mask][day_cols].sum(axis=0).values
    
    snap_days = cal_slim[cal_slim[snap_col] == 1].index.tolist()
    non_snap = cal_slim[cal_slim[snap_col] == 0].index.tolist()
    
    snap_sales = np.mean([state_daily[i] for i in snap_days if i < len(state_daily)])
    non_snap_sales = np.mean([state_daily[i] for i in non_snap if i < len(state_daily)])
    lift = (snap_sales - non_snap_sales) / non_snap_sales * 100
    
    print(f"    {state}: SNAP avg={snap_sales:,.0f}  non-SNAP avg={non_snap_sales:,.0f}  lift={lift:+.1f}%")

# Event impact
print("\n  Event type impact on total sales:")
for etype in sorted(cal_slim["event_type_1"].dropna().unique()):
    event_days = cal_slim[cal_slim["event_type_1"] == etype]["daily_total"]
    non_event = cal_slim[cal_slim["event_type_1"].isna()]["daily_total"]
    lift = (event_days.mean() - non_event.mean()) / non_event.mean() * 100
    print(f"    {etype:<15}: avg={event_days.mean():,.0f}  vs baseline={non_event.mean():,.0f}  lift={lift:+.1f}%")

# Top individual events
print("\n  Top 10 events by sales impact:")
event_impact = cal_slim.groupby("event_name_1")["daily_total"].mean().sort_values(ascending=False)
non_event_baseline = cal_slim[cal_slim["event_name_1"].isna()]["daily_total"].mean()
for i, (event, avg) in enumerate(event_impact.head(10).items()):
    lift = (avg - non_event_baseline) / non_event_baseline * 100
    print(f"    {i+1:>2}. {event:<25} avg_sales={avg:>10,.0f}  lift={lift:+.1f}%")

# =========================================================================
print("\n" + "=" * 70)
print("  6. DEPARTMENT-LEVEL DYNAMICS")
print("=" * 70)

depts = sorted(sales["dept_id"].unique())
print(f"\n  {'Dept':<15} {'Mean/item':>10} {'Zero%':>8} {'Max sale':>10} {'Items':>8} {'Volatility':>12}")
print(f"  {'-'*15} {'-'*10} {'-'*8} {'-'*10} {'-'*8} {'-'*12}")

for dept in depts:
    dept_mask = sales["dept_id"] == dept
    dept_data = sales[dept_mask][day_cols].values
    n_items = dept_mask.sum()
    mean_sale = dept_data.mean()
    zero_pct = (dept_data == 0).mean() * 100
    max_sale = dept_data.max()
    # Volatility = std of daily dept totals / mean
    dept_daily = dept_data.sum(axis=0)
    volatility = dept_daily.std() / (dept_daily.mean() + 1e-8)
    print(f"  {dept:<15} {mean_sale:>10.3f} {zero_pct:>7.1f}% {max_sale:>10} {n_items:>8,} {volatility:>12.3f}")

# =========================================================================
print("\n" + "=" * 70)
print("  7. POTENTIAL NODE FEATURES FOR DTNET")
print("=" * 70)

print("""
  Based on the EDA, here are the features we could engineer per node:

  STORE-LEVEL NODE FEATURES:
    - mean_daily_sales        : average daily total sales
    - sales_volatility        : std/mean of daily sales (CV)
    - zero_rate               : fraction of item-days with zero sales
    - trend_slope             : linear regression slope over last 90 days
    - snap_sensitivity        : sales lift % on SNAP days
    - event_sensitivity       : sales lift % on event days
    - weekend_ratio           : weekend vs weekday sales ratio
    - price_level             : mean sell price across items
    - n_active_items          : items with >0 sales in last 30 days

  ITEM-LEVEL NODE FEATURES:
    - mean_sales              : average daily sales
    - sales_cv                : coefficient of variation
    - zero_streak_max         : longest consecutive zero-sales run
    - zero_rate               : fraction of days with zero sales
    - price                   : current sell price
    - price_change_freq       : how often price changes
    - cross_store_sync        : correlation of sales across stores
    - seasonality_strength    : amplitude of monthly pattern
    - dept_encoded            : one-hot department
    - cat_encoded             : one-hot category

  EDGE FEATURES (store-item or cross-store):
    - demand_correlation      : pairwise Pearson corr between connected nodes
    - shared_item_count       : number of shared items (cross-store edges)
    - geographic_proximity    : same-state flag
    - substitution_score      : when one store stocks out, does the other spike?
""")

# =========================================================================
print("\n" + "=" * 70)
print("  8. DISRUPTION DEFINITION CANDIDATES")
print("=" * 70)

# Compute some concrete disruption statistics
print("\n  A. Zero-to-nonzero transitions (potential stockout recovery):")
transitions = 0
stockout_events = 0
for i in range(min(3000, len(sales_matrix))):  # sample
    row = sales_matrix[i]
    was_zero = False
    zero_run = 0
    for j in range(len(row)):
        if row[j] == 0:
            zero_run += 1
            was_zero = True
        else:
            if was_zero and zero_run >= 7:
                stockout_events += 1
            transitions += 1 if was_zero else 0
            was_zero = False
            zero_run = 0

print(f"    Sampled 3000 item-store series:")
print(f"    Zero-to-nonzero transitions: {transitions:,}")
print(f"    Stockout events (>=7 consecutive zeros): {stockout_events:,}")

# B. Demand spikes
print(f"\n  B. Demand spikes (>3x rolling 28-day mean) per department:")
for dept in depts:
    dept_mask = sales["dept_id"] == dept
    dept_daily = sales[dept_mask][day_cols].sum(axis=0).values.astype(float)
    series = pd.Series(dept_daily)
    roll_mean = series.rolling(28, min_periods=7).mean()
    spikes = (series > roll_mean * 3).sum()
    drops = (series < roll_mean * 0.33).sum()
    print(f"    {dept:<15}: {spikes:>3} spikes, {drops:>3} drops")

print("\n" + "=" * 70)
print("  DEEP EDA COMPLETE")
print("=" * 70)
