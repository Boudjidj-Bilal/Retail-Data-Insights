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

