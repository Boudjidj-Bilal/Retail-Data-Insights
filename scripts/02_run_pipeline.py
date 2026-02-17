
import pandas as pd
import os
import json
from datetime import datetime

print("="*60)
print("SIMPLE DATA PIPELINE")
print("="*60)

DATA_DIR = './data'
OUTPUT_DIR = './data/processed'

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\n[1/5] Loading data...")

orders = pd.read_csv(f'{DATA_DIR}/orders.csv')
print(f"  ✓ orders: {len(orders):,} rows")

orders_prior = pd.read_csv(f'{DATA_DIR}/order_products__prior.csv')
print(f"  ✓ orders_prior: {len(orders_prior):,} rows")

products = pd.read_csv(f'{DATA_DIR}/products.csv')
print(f"  ✓ products: {len(products):,} rows")

aisles = pd.read_csv(f'{DATA_DIR}/aisles.csv')  # 134 aisles
print(f"  ✓ aisles: {len(aisles)} rows (134 aisles)")

departments = pd.read_csv(f'{DATA_DIR}/departments.csv')  # 21 departments
print(f"  ✓ departments: {len(departments)} rows (21 departments)")

# Load priced products if available
if os.path.exists(f'{DATA_DIR}/products_priced_eur.csv'):
    products_priced = pd.read_csv(f'{DATA_DIR}/products_priced_eur.csv')
    print(f"  ✓ products_priced_eur: {len(products_priced):,} rows")
else:
    products_priced = None
    print("  No priced products (run scripts/01_convert_prices.py first)")

# ============================================================================
# STEP 2: CREATE BASKETS
# ============================================================================

print("\n[2/5] Creating baskets for association rules...")

# Group products by order
baskets = orders_prior.groupby('order_id')['product_id'].apply(list)
baskets = baskets.reset_index()
baskets.columns = ['order_id', 'products']
baskets.to_csv(f'{OUTPUT_DIR}/baskets.csv', index=False)
print(f"  ✓ baskets.csv: {len(baskets):,} orders")

# ============================================================================
# STEP 3: CREATE RFM CUSTOMER SEGMENTS 
# ============================================================================

print("\n[3/5] Creating RFM customer segments...")

# Count products per order
order_counts = orders_prior.groupby('order_id').size().reset_index()
order_counts.columns = ['order_id', 'num_items']

# Merge with orders
order_data = pd.merge(orders, order_counts, on='order_id', how='left')

# Add prices if available
if products_priced is not None:
    order_prices = pd.merge(
        orders_prior[['order_id', 'product_id']],
        products_priced[['product_id', 'price_eur']],
        on='product_id',
        how='left'
    )
    order_prices['price_eur'] = order_prices['price_eur'].fillna(0)
    order_value = order_prices.groupby('order_id')['price_eur'].sum().reset_index()
    order_value.columns = ['order_id', 'order_value_eur']
    order_data = pd.merge(order_data, order_value, on='order_id', how='left')
else:
    order_data['order_value_eur'] = 0

# Aggregate to customer level (206K customers - NOT 32M rows!)
rfm = order_data.groupby('user_id').agg({
    'order_id': 'nunique',           # Frequency
    'order_value_eur': 'sum',         # Monetary
    'num_items': 'sum',
    'days_since_prior_order': 'mean'  # Recency proxy
}).reset_index()

rfm.columns = ['user_id', 'num_orders', 'total_spent_eur', 
               'total_items', 'avg_days_between_orders']

# Simple RFM scoring (1-5)
rfm['R_score'] = pd.qcut(rfm['avg_days_between_orders'].fillna(0), 
                         q=5, labels=[5,4,3,2,1]).astype(int)
rfm['F_score'] = pd.qcut(rfm['num_orders'], q=5, labels=[1,2,3,4,5]).astype(int)
rfm['M_score'] = pd.qcut(rfm['total_spent_eur'], q=5, labels=[1,2,3,4,5]).astype(int)
rfm['RFM_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']

# Customer segments (Section 5: budget vs. premium)
def get_segment(row):
    if row['F_score'] >= 4 and row['M_score'] >= 4:
        return 'Champions'
    elif row['F_score'] >= 3 and row['M_score'] >= 3:
        return 'Loyal'
    elif row['M_score'] >= 4:
        return 'Big Spenders'
    elif row['F_score'] >= 4:
        return 'Budget Shoppers'
    else:
        return 'Regular'

rfm['segment'] = rfm.apply(get_segment, axis=1)
rfm.to_csv(f'{OUTPUT_DIR}/rfm_customer_segments.csv', index=False)
print(f"  ✓ rfm_customer_segments.csv: {len(rfm):,} customers")

# ============================================================================
# STEP 4: ADD CATEGORIES (134 Aisles, 21 Departments)
# ============================================================================

print("\n[4/5] Adding category information...")

# Merge orders_prior with products (for aisle/dept)
product_data = pd.merge(
    orders_prior,
    products[['product_id', 'aisle_id', 'department_id']],
    on='product_id',
    how='left'
)

# Add aisle names (134 aisles from aisles.csv)
product_data = pd.merge(
    product_data,
    aisles.rename(columns={'aisle': 'aisle_name'}),
    on='aisle_id',
    how='left'
)

# Add department names (21 departments from departments.csv)
product_data = pd.merge(
    product_data,
    departments.rename(columns={'department': 'department_name'}),
    on='department_id',
    how='left'
)

# Aisle performance (134 aisles)
aisle_perf = product_data.groupby('aisle_name').agg({
    'order_id': 'nunique',
    'product_id': 'count'
}).reset_index()
aisle_perf.columns = ['aisle', 'num_orders', 'items_sold']
aisle_perf.to_csv(f'{OUTPUT_DIR}/aisle_performance.csv', index=False)
print(f"  ✓ aisle_performance.csv: {len(aisle_perf)} aisles (134 total)")

# Department performance (21 departments)
dept_perf = product_data.groupby('department_name').agg({
    'order_id': 'nunique',
    'product_id': 'count'
}).reset_index()
dept_perf.columns = ['department', 'num_orders', 'items_sold']
dept_perf.to_csv(f'{OUTPUT_DIR}/department_performance.csv', index=False)
print(f"  ✓ department_performance.csv: {len(dept_perf)} departments (21 total)")

# ============================================================================
# STEP 5: SAVE METADATA (For PDF Report Section 7)
# ============================================================================

print("\n[5/5] Saving metadata...")

metadata = {
    'date': datetime.now().isoformat(),
    'project': 'Data-Driven Retail Insights',
    'instructor': 'Assan Sanogo',
    'data_sources': {
        'orders': len(orders),
        'orders_prior': len(orders_prior),
        'aisles': 134,
        'departments': 21
    },
    'output_files': {
        'baskets.csv': len(baskets),
        'rfm_customer_segments.csv': len(rfm),
        'aisle_performance.csv': len(aisle_perf),
        'department_performance.csv': len(dept_perf)
    },
    'price_coverage': 'Available' if products_priced is not None else 'Not available'
}

with open(f'{OUTPUT_DIR}/metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"  ✓ metadata.json saved")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*60)
print("PIPELINE COMPLETE")
print("="*60)
print(f"\nFiles created in {OUTPUT_DIR}/:")
for file in os.listdir(OUTPUT_DIR):
    size_mb = os.path.getsize(f'{OUTPUT_DIR}/{file}') / 1024 / 1024
    print(f"  ✓ {file}: {size_mb:.2f} MB")

print(f"\nReference files used:")
print(f"  - aisles.csv: 134 aisles")
print(f"  - departments.csv: 21 departments")

print(f"\nReady for:")
print(f"  - Association rules (baskets.csv)")
print(f"  - Customer segmentation (rfm_customer_segments.csv)")
print(f"  - Streamlit app")
print(f"  - PDF Report (Section 7)")