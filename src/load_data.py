import pandas as pd
def load_instacart_data(data_path='../data/raw/'):

    dtypes_optimized = {
        'orders': {'order_id': 'int32', 'user_id': 'int32', 'eval_set': 'category',
                   'order_number': 'int8', 'order_dow': 'int8',
                   'order_hour_of_day': 'int8', 'days_since_prior_order': 'float32'},
        'order_products': {'order_id': 'int32', 'product_id': 'int32',
                           'add_to_cart_order': 'uint8', 'reordered': 'int8'},
        'products': {'product_id': 'int32', 'aisle_id': 'uint8',
                     'department_id': 'int8', 'product_name': 'category'},
        'aisles': {'aisle_id': 'uint8', 'aisle': 'category'},
        'departments': {'department_id': 'int8', 'department': 'category'}
    }
    
    data = {
        'orders': pd.read_csv(f'{data_path}orders.csv', dtype=dtypes_optimized['orders']),
        'products': pd.read_csv(f'{data_path}products.csv', dtype=dtypes_optimized['products']),
        'order_products_prior': pd.read_csv(f'{data_path}order_products__prior.csv', dtype=dtypes_optimized['order_products']),
        'order_products_train': pd.read_csv(f'{data_path}order_products__train.csv', dtype=dtypes_optimized['order_products']),
        'aisles': pd.read_csv(f'{data_path}aisles.csv', dtype=dtypes_optimized['aisles']),
        'departments': pd.read_csv(f'{data_path}departments.csv', dtype=dtypes_optimized['departments'])
    }
    
    return data

def quick_info(df, name="Dataset"):

    # name for each dataset
    print(f"\n{'='*60}")
    print(f"{name.upper()}")
    print(f"{'='*60}")
    
    # Shape
    print(f"\n SHAPE:")
    print(f"  - Rows: {df.shape[0]:,}")
    print(f"  - Columns: {df.shape[1]}")
    
    # Missing values
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\n⚠️  MISSING VALUES:")
        for col, count in missing[missing > 0].items():
            print(f"  - {col}: {count:,}")
    else:
        print(f"\n✅ NO MISSING VALUES")
    
    # Duplicates
    dup = df.duplicated().sum()
    if dup > 0:
        print(f"\n⚠️  DUPLICATES: {dup:,} rows")
    else:
        print(f"\n✅ NO DUPLICATES")

def create_products_enriched(order_products_prior, products, aisles, departments):
# compute product sales with information from products, aisles and departments

    product_sales = order_products_prior.groupby('product_id').agg(nb_sales=('order_id', 'count'),reorder_rate=('reordered', 'mean')).reset_index()
    
    products_enriched = product_sales.merge(products[['product_id', 'product_name', 'aisle_id', 'department_id']], on='product_id',how='left')

    products_enriched = products_enriched.merge(departments[['department_id', 'department']], on='department_id',how='left')
    
    products_enriched = products_enriched.merge(aisles[['aisle_id', 'aisle']], on='aisle_id',how='left')

# add frequency of purchase
    total_orders = order_products_prior['order_id'].nunique()
    
    products_enriched['purchase_frequency'] = (products_enriched['nb_sales'] / total_orders * 100)

    return products_enriched