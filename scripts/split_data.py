import pandas as pd
import numpy as np

# Create a function to perform a temporal split of the Instacart dataset based on order_number
def temporal_split_instacart(
    order_products_prior,
    order_products_train,
    orders,
    products,
    departments,
    train_ratio=0.7,
    save_path='data/processed/',
    visualize=True
):

# 1. Merge prior + train datasets
 
    print("\n[1/6] Merging prior + train datasets...")
    all_order_products = pd.concat([
        order_products_prior,
        order_products_train
    ], ignore_index=True)
    
    print(f"  Total rows: {len(all_order_products):,}")
    print(f"  Unique orders: {all_order_products['order_id'].nunique():,}")
    print(f"  Unique products: {all_order_products['product_id'].nunique():,}")
    
# 2. Temporal split based on order_number
    
    print("\n[2/6] Temporal split based on order_number...")
    
    orders_sorted = orders[['order_id', 'user_id', 'order_number']].sort_values('order_number')
    unique_orders = orders_sorted['order_id'].unique()
    
    n_orders = len(unique_orders)
    
    train_end = int(n_orders * train_ratio)
    
    train_order_ids = set(unique_orders[:train_end])
    test_order_ids = set(unique_orders[train_end:])
    
    print(f"  Total orders: {n_orders:,}")
    print(f"  Train orders: {len(train_order_ids):,} ({len(train_order_ids)/n_orders*100:.1f}%)")
    print(f"  Test orders: {len(test_order_ids):,} ({len(test_order_ids)/n_orders*100:.1f}%)")
    
    train_data = all_order_products[all_order_products['order_id'].isin(train_order_ids)]
    test_data = all_order_products[all_order_products['order_id'].isin(test_order_ids)]
    
    print(f"\n  Train rows: {len(train_data):,}")
    print(f"  Test rows: {len(test_data):,}")
    
# 3. Check distribution of basket sizes to ensure splits are homogeneous 
    print("\n[3/6] Checking basket size distributions...")
    
    train_basket_sizes = train_data.groupby('order_id').size()
    test_basket_sizes = test_data.groupby('order_id').size()
    
    train_bs_stats = {
        'mean': train_basket_sizes.mean(),
        'median': train_basket_sizes.median(),
        'std': train_basket_sizes.std(),
        'min': train_basket_sizes.min(),
        'max': train_basket_sizes.max()
    }
    
    test_bs_stats = {
        'mean': test_basket_sizes.mean(),
        'median': test_basket_sizes.median(),
        'std': test_basket_sizes.std(),
        'min': test_basket_sizes.min(),
        'max': test_basket_sizes.max()
    }
    
    print(f"\n  TRAIN - Basket Size:")
    print(f"    Mean: {train_bs_stats['mean']:.2f}, Median: {train_bs_stats['median']:.0f}, Std: {train_bs_stats['std']:.2f}")
    print(f"    Range: [{train_bs_stats['min']}, {train_bs_stats['max']}]")
    
    print(f"\n  TEST - Basket Size:")
    print(f"    Mean: {test_bs_stats['mean']:.2f}, Median: {test_bs_stats['median']:.0f}, Std: {test_bs_stats['std']:.2f}")
    print(f"    Range: [{test_bs_stats['min']}, {test_bs_stats['max']}]")
    
    print(f"\n  📊 Distribution Comparison:")
    train_test_diff = abs(train_bs_stats['mean'] - test_bs_stats['mean'])
    print(f"    Train vs Test: Δ mean = {train_test_diff:.2f}", end="")
    
    if train_test_diff > 1.0:
        print(f" WARNING (>1.0)")
    else:
        print(f" ✅")
    
# 4. Check department diversity distributions to ensure splits are homogeneous

    print("\n[4/6] Checking department diversity distributions...")
        
    products_full = products.merge(
        departments[['department_id', 'department']], 
        on='department_id'
    )
    
    train_with_dept = train_data.merge(
        products_full[['product_id', 'department']], 
        on='product_id', 
        how='left'
    )
    test_with_dept = test_data.merge(
        products_full[['product_id', 'department']], 
        on='product_id', 
        how='left'
    )
    
    train_diversity = train_with_dept.groupby('order_id')['department'].nunique()
    test_diversity = test_with_dept.groupby('order_id')['department'].nunique()
    
    train_div_stats = {
        'mean': train_diversity.mean(),
        'median': train_diversity.median(),
        'std': train_diversity.std()
    }
    
    test_div_stats = {
        'mean': test_diversity.mean(),
        'median': test_diversity.median(),
        'std': test_diversity.std()
    }
    
    print(f"\n  TRAIN - Department Diversity:")
    print(f"    Mean: {train_div_stats['mean']:.2f}, Median: {train_div_stats['median']:.0f}, Std: {train_div_stats['std']:.2f}")
    
    print(f"\n  TEST - Department Diversity:")
    print(f"    Mean: {test_div_stats['mean']:.2f}, Median: {test_div_stats['median']:.0f}, Std: {test_div_stats['std']:.2f}")
    
    print(f"\n  📊 Distribution Comparison:")
    train_test_div_diff = abs(train_div_stats['mean'] - test_div_stats['mean'])
    print(f"    Train vs Test: Δ mean = {train_test_div_diff:.2f}", end="")
    
    if train_test_div_diff > 0.5:
        print(f" WARNING (>0.5)")
    else:
        print(f" ✅")
    
# 5. Saving splits
    
    print("\n[6/6] Saving splits...")
    
    import os
    os.makedirs(save_path, exist_ok=True)
    
    train_data.to_csv(f'{save_path}train.csv', index=False)
    print(f"  ✅ Train saved: {save_path}train.csv ({len(train_data):,} rows)")
    
    test_data.to_csv(f'{save_path}test.csv', index=False)
    print(f"  ✅ Test saved: {save_path}test.csv ({len(test_data):,} rows)")
    
    # Return results
    return {'train': train_data, 'test': test_data}

# Usage example (to be run in a separate script)

if __name__ == "__main__":
    import sys
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    sys.path.insert(0, project_root)
    
    from scripts.load_data import load_instacart_data
    
    data = load_instacart_data()
    
    splits = temporal_split_instacart(
        order_products_prior=data['order_products_prior'],
        order_products_train=data['order_products_train'],
        orders=data['orders'],
        products=data['products'],
        departments=data['departments'],
        train_ratio=0.7,
        save_path='data/processed/',
        visualize=False
    )
    
    print("\nTemporal split completed!")
    for split_name, split_data in splits.items():
        print(f"  - {split_name.capitalize()}: {len(split_data):,} rows")