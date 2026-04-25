import pandas as pd
import numpy as np
import time
from pathlib import Path

def deep_eda():
    print("Loading M5 Data...")
    sales = pd.read_csv("c:/Users/SHIVAM BHATT/Downloads/m5-forecasting-accuracy/sales_train_evaluation.csv")
    calendar = pd.read_csv("c:/Users/SHIVAM BHATT/Downloads/m5-forecasting-accuracy/calendar.csv")
    
    day_cols = [c for c in sales.columns if c.startswith('d_')]
    sales_mat = sales[day_cols].values
    
    print("\n--- 1. SPARSITY & ZERO-INFLATION ---")
    total_cells = sales_mat.size
    zero_cells = np.sum(sales_mat == 0)
    print(f"Total Sales Matrix: 30,490 x 1,941 = {total_cells:,} days")
    print(f"Exact Zero Sales: {zero_cells:,} ({zero_cells/total_cells*100:.2f}%)")
    
    zero_rates_per_item = np.mean(sales_mat == 0, axis=1)
    print(f"Median zero-rate per item-store: {np.median(zero_rates_per_item)*100:.2f}%")
    print(f"Items that are >90% zeros: {np.sum(zero_rates_per_item > 0.90):,}")
    print(f"Items that are <10% zeros (Fast movers): {np.sum(zero_rates_per_item < 0.10):,}")
    
    print("\n--- 2. DEMAND VOLATILITY (Lumpiness) ---")
    # CV2 = (Variance / Mean^2). CV2 > 0.5 is usually considered lumpy/erratic
    means = np.mean(sales_mat, axis=1) + 1e-8
    variances = np.var(sales_mat, axis=1)
    cv2 = variances / (means ** 2)
    print(f"Median Coefficient of Variation Squared (CV2): {np.median(cv2):.2f}")
    print(f"Highly erratic items (CV2 > 1.0): {np.sum(cv2 > 1.0):,} ({np.sum(cv2 > 1.0)/30490*100:.1f}%)")
    
    print("\n--- 3. CROSS-STORE CORRELATION (Validating our Edges) ---")
    # Pick a random sample of items to check cross-store correlations
    np.random.seed(42)
    sample_items = np.random.choice(sales['item_id'].unique(), size=50, replace=False)
    cross_store_corrs = []
    
    for item in sample_items:
        # Get all stores in CA for this item
        item_data = sales[(sales['item_id'] == item) & (sales['state_id'] == 'CA')]
        if len(item_data) > 1:
            mat = item_data[day_cols].values
            # Compute correlation matrix
            if np.std(mat[0]) > 0 and np.std(mat[1]) > 0:
                corr = np.corrcoef(mat[0], mat[1])[0, 1]
                if not np.isnan(corr):
                    cross_store_corrs.append(corr)
    
    print(f"Average Pearson Correlation of identical items across CA stores: {np.mean(cross_store_corrs):.3f}")
    
    print("\n--- 4. EVENT SHOCKS ---")
    # Compute average sales on event days vs non-event days
    sales_per_day = np.sum(sales_mat, axis=0)
    has_event = calendar['event_name_1'].notna().values[:1941]
    
    avg_normal = np.mean(sales_per_day[~has_event])
    avg_event = np.mean(sales_per_day[has_event])
    print(f"Average total daily volume (Normal Day): {avg_normal:,.0f}")
    print(f"Average total daily volume (Event Day):  {avg_event:,.0f}")
    print(f"Event Day Impact: {((avg_event / avg_normal) - 1) * 100:.1f}%")
    
    # SNAP Impact
    snap_ca = calendar['snap_CA'].values[:1941] == 1
    ca_sales = sales[sales['state_id'] == 'CA'][day_cols].values
    ca_daily = np.sum(ca_sales, axis=0)
    print(f"CA Volume (Non-SNAP): {np.mean(ca_daily[~snap_ca]):,.0f}")
    print(f"CA Volume (SNAP Day): {np.mean(ca_daily[snap_ca]):,.0f}")
    print(f"SNAP Day Lift in CA: {((np.mean(ca_daily[snap_ca]) / np.mean(ca_daily[~snap_ca])) - 1) * 100:.1f}%")

if __name__ == '__main__':
    deep_eda()
