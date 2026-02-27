"""
Association Rules Generation and Evaluation Functions

This module provides utilities for:
1. Generating association rules using FP-Growth algorithm
2. Preparing transaction data from DataFrames
3. Evaluating rules using offline hold-out validation
4. Comparing multiple recommendation approaches

References:
- Han et al. (2000): Mining frequent patterns without candidate generation
- Shani & Gunawardana (2011): Evaluating Recommendation Systems
"""

import pandas as pd
import numpy as np
import random
from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import gc


# ════════════════════════════════════════════════════════════════════════
# GENERATORS - Read CSV by chunks without loading everything in memory
# ════════════════════════════════════════════════════════════════════════

def csv_chunk_generator(filepath, chunksize=100_000, filter_column=None, filter_values=None):
    """
    Generator that reads a CSV file chunk by chunk and applies optional filtering.
    
    Arguments:
        filepath: Path to the CSV file
        chunksize: Number of rows per chunk (default: 100_000)
        filter_column: Optional column to filter on ('department', 'segment')
        filter_values: Value(s) to keep (string or list)
    
    Yields:
        Filtered DataFrame chunks
    """
    if isinstance(filter_values, str):
        filter_values = [filter_values]

    for chunk in pd.read_csv(filepath, chunksize=chunksize):
        if filter_column and filter_values:
            chunk = chunk[chunk[filter_column].isin(filter_values)]
        if not chunk.empty:
            yield chunk


def get_top_products_from_csv(filepath, top_n, filter_column=None, filter_values=None, chunksize=100_000):
    """
    Identify top N products by frequency from a CSV file using chunks.
    
    Makes a first pass through the CSV to count product frequencies,
    without loading the entire file into memory.
    
    Arguments:
        filepath: Path to the CSV file
        top_n: Number of top products to keep
        filter_column: Optional column to filter on
        filter_values: Value(s) to keep
        chunksize: Number of rows per chunk
    
    Returns:
        Set of top N product names
    """
    print(f"    Computing top {top_n} products (pass 1/2)...")
    product_counts = {}

    for chunk in csv_chunk_generator(filepath, chunksize, filter_column, filter_values):
        for product, count in chunk['product_name'].value_counts().items():
            product_counts[product] = product_counts.get(product, 0) + count

    top_products = set(
        sorted(product_counts, key=product_counts.get, reverse=True)[:top_n]
    )
    return top_products


def prepare_transactions_from_csv(filepath, filter_column=None, filter_values=None,
                                   top_n_products=None, chunksize=100_000):
    """
    Prepare transaction list from a CSV file using a generator (chunk by chunk).
    
    Memory-efficient alternative to prepare_transactions() which requires
    the full DataFrame to be loaded in memory.
    
    Makes 2 passes through the CSV:
    - Pass 1 (if top_n_products): count product frequencies to identify top N
    - Pass 2: build the transaction dict {order_id -> [products]}
    
    Arguments:
        filepath: Path to the CSV file (must have 'order_id' and 'product_name' columns)
        filter_column: Optional column to filter ('department', 'segment')
        filter_values: Value(s) to keep (string or list)
        top_n_products: Keep only top N most frequent products (int)
        chunksize: Number of rows per chunk (default: 100_000)
    
    Returns:
        List of transactions
    """
    # Pass 1: identify top N products if needed
    top_products = None
    if top_n_products:
        top_products = get_top_products_from_csv(
            filepath, top_n_products, filter_column, filter_values, chunksize
        )

    # Pass 2: build transactions dict
    print(f"    Building transactions (pass 2/2)...")
    transactions_dict = {}

    for chunk in csv_chunk_generator(filepath, chunksize, filter_column, filter_values):
        # Keep only top N products
        if top_products:
            chunk = chunk[chunk['product_name'].isin(top_products)]

        # Group by order
        for order_id, group in chunk.groupby('order_id'):
            if order_id not in transactions_dict:
                transactions_dict[order_id] = []
            transactions_dict[order_id].extend(group['product_name'].tolist())

    return list(transactions_dict.values())


# Rules generation function using FP-Growth algorithm

def generate_association_rules(transactions, min_support=0.005, min_confidence=0.15, 
                               min_lift=1.3, max_transactions=None):
    """
    Generate association rules from transaction list using FP-Growth algorithm.
    
    Process:
    1. Encode transactions to binary matrix
    2. Find frequent itemsets using FP-Growth (items appearing together often)
    3. Generate association rules from frequent itemsets
    4. Filter rules by confidence and lift thresholds
    
    Arguments:
        transactions: List of lists, each containing product names
                     Example: [['Banana', 'Milk'], ['Banana', 'Bread', 'Eggs']]
        
        min_support: Minimum support threshold - Support = P(A inter B) = proportion of transactions containing the itemset
        
        min_confidence: Minimum confidence thresholdd - Confidence = P(B|A) = probability of B given A => Measures how often the rule is correct
        
        min_lift: Minimum lift threshold - Lift = P(A inter B) / (P(A) × P(B))
                 Lift = 1.3 means 30% more likely than random
        
        max_transactions: Maximum number of transactions to use - Used to limit computation time and memory usage
    
    Returns:
        List of association rules with columns: antecedent, consequent, support, confidence, lift

    """
    
    # Sample transactions if dataset too large
    if max_transactions and len(transactions) > max_transactions:
        transactions = random.sample(transactions, max_transactions)
    
    # Step 1: Encode transactions to binary matrix
    # Each row = transaction, each column = product (1 if present, 0 otherwise)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)
    
    # Step 2: Find frequent itemsets using FP-Growth
    # Returns all combinations of products appearing together frequently
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)
    
    if len(frequent_itemsets) == 0:
        return None
    
    # Step 3: Generate association rules from frequent itemsets
    rules = association_rules(frequent_itemsets,metric="confidence",min_threshold=min_confidence)
    
    if len(rules) == 0:
        return None
    
    # Step 4: Clean and filter rules
    # Convert frozensets to readable strings
    rules['antecedent'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['consequent'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))
    
    # Filter by lift
    rules_filtered = rules[rules['lift'] > min_lift].copy()
    
    # Keep only essential columns
    rules_filtered = rules_filtered[['antecedent', 'consequent', 'support', 'confidence', 'lift']]
    
    # Free memory
    del df_encoded, frequent_itemsets, rules
    gc.collect()
    
    return rules_filtered


def prepare_transactions(data, filter_column=None, filter_values=None, 
                        top_n_products=None):
    """
    Prepare transaction list from DataFrame for association rule mining.
    
    Converts long-format DataFrame into list of transactions.
    Use prepare_transactions_from_csv() instead if working with large CSV files.
    
    Arguments:
        data: DataFrame with columns 'product_name' and 'order_id'
        filter_column: Optional column to filter ('department', 'segment')
        filter_values: Value(s) to keep (string or list)
        top_n_products: Keep only top N products (int)
    
    Returns:
        List of transactions
    """
    
    # Filter by column if specified
    if filter_column and filter_values:
        if isinstance(filter_values, str):
            filter_values = [filter_values]
        data = data[data[filter_column].isin(filter_values)]
    
    # Keep only top N products
    if top_n_products:
        top_products = data['product_name'].value_counts().head(top_n_products).index
        data = data[data['product_name'].isin(top_products)]
    
    # Group by order
    transactions = data.groupby('order_id')['product_name'].apply(list).tolist()
    
    return transactions


# ════════════════════════════════════════════════════════════════════════
# RULE EVALUATION
# ════════════════════════════════════════════════════════════════════════

def evaluate_rules_from_csv(rules, filepath, groupby_column=None, k=10,
                             sample_size=10_000, min_basket_size=4, chunksize=100_000):
    """
    Evaluate association rules from a CSV file using chunks (memory-efficient).
    
    Same logic as evaluate_rules() but reads test data chunk by chunk
    instead of requiring a full DataFrame in memory.
    
    Arguments:
        rules: DataFrame with columns [antecedent, consequent] and optional groupby_column
        filepath: Path to the test CSV file
        groupby_column: Optional column for grouped evaluation ('department', 'segment')
        k: Number of recommendations to generate
        sample_size: Number of test baskets to sample (default: 10000)
        min_basket_size: Minimum basket size for evaluation (default: 4)
        chunksize: Number of rows per chunk
    
    Returns:
        Dictionary with metrics or None if no recommendations generated
    """
    print("    Building test baskets from CSV...")

    # Build baskets dict from CSV chunks
    baskets_dict = {}      # order_id -> list of products
    group_dict = {}        # order_id -> group value (if groupby_column)

    for chunk in csv_chunk_generator(filepath, chunksize):
        for order_id, group in chunk.groupby('order_id'):
            if order_id not in baskets_dict:
                baskets_dict[order_id] = []
            baskets_dict[order_id].extend(group['product_name'].tolist())

            # Keep track of group value per basket
            if groupby_column and order_id not in group_dict:
                group_dict[order_id] = group[groupby_column].iloc[0]

    # Convert to DataFrame
    if groupby_column:
        test_baskets = pd.DataFrame([
            {'order_id': oid, 'product_name': prods, groupby_column: group_dict.get(oid)}
            for oid, prods in baskets_dict.items()
        ])
    else:
        test_baskets = pd.DataFrame([
            {'order_id': oid, 'product_name': prods}
            for oid, prods in baskets_dict.items()
        ])

    del baskets_dict, group_dict
    gc.collect()

    # Delegate to shared evaluation logic
    return _evaluate_baskets(rules, test_baskets, groupby_column, k, sample_size, min_basket_size)


def evaluate_rules(rules, test_data, groupby_column=None, k=10, 
                   sample_size=10000, min_basket_size=4):
    """
    Evaluate association rules using offline hold-out validation methodology.
    
    Source : Shani & Gunawardana, 2011
    
    This function implements the evaluation for recommendation
    
    1. BASKET SPLIT (50/50)
       Each test basket is split into:
       - Known items (first 50%): Input to the recommendation system
       - Target items (last 50%): Actual purchases used for evaluation
    
    2. RULE APPLICATION
       Apply association rules to known items:
       - For each rule: IF rule_antecedent is in known_items THEN recommend rule_consequent
       - Aggregate all recommendations (up to K items)
    
    3. METRICS CALCULATION
       Compare recommendations against target items using confusion matrix:
       
                          | Recommended | Not Recommended |
       --------------------------------------------------------
       Purchased          |     tp      |       fn        |
       Not Purchased      |     fp      |       tn        |
       
       where:
       - tp (true positives)  = products recommended AND purchased
       - fp (false positives) = products recommended but NOT purchased
       - fn (false negatives) = products purchased but NOT recommended
       - tn (true negatives)  = products neither recommended nor purchased
       
       In our implementation:
       tp = Intersection recommendations + target items
       fp = Recommended but wrong = recommendations - tp
       fn = Missed recommendations = target items - tp
       tn = NOT calculated
    
    METRICS COMPUTED
    ================
    K = numlber of recommendations generated (here K=10)

    Precision@K = tp / K = |recommended & purchased| / K
    - Measures: What proportion of recommendations are correct?
    - Range: [0, 1], higher is better

    Recall@K = tp / |target| = |recommended & purchased| / |target|
    - Measures: What proportion of purchases were recommended?
    - Range: [0, 1], higher is better
    
    Coverage = baskets_with_recs / total_baskets
    - Measures: What proportion of baskets receive recommendations?
    - Range: [0, 1], higher is better

    Args:
        rules: DataFrame with columns [antecedent, consequent] and optional groupby_column
               Each row represents one rule: IF antecedent THEN consequent
        
        test_data: DataFrame with columns [order_id, product_name] and optional groupby_column
                  Each row represents one product in one order
        
        groupby_column: Optional column for grouped evaluation
                       Example: 'department' (apply only department-specific rules)
                       Example: 'segment' (apply only segment-specific rules)
                       If None, all rules are applied to all baskets
        
        k: Number of recommendations to generate 
        
        sample_size: Number of test baskets to sample (default: 10000)
                    Balances evaluation speed vs statistical significance
        
        min_basket_size: Minimum basket size for evaluation (default: 4). Baskets with <4 products cannot be split 50/50 meaningfully
    
    Returns:
        Dictionary with metrics:
        {
            'precision@K': float,      # Average precision across all baskets
            'recall@K': float,         # Average recall across all baskets
            'coverage': float,         # Proportion receiving recommendations
            'avg_hits': float,         # Average correct recommendations per basket
            'n_baskets': int,          # Total baskets evaluated
            'n_baskets_with_recs': int # Baskets with ≥1 recommendation
        }
        
        Returns None if no valid baskets found or no recommendations generated.
    """
    
    # Step 1: Group test data by order to create baskets
    if groupby_column:
        # With grouping: keep track of department/segment per basket
        test_baskets = test_data.groupby('order_id').agg({
            'product_name': list,
            groupby_column: 'first'  # simplify to one value per basket => can be problematic 
        }).reset_index()
    else:
        # Without grouping: just aggregate products per order
        test_baskets = test_data.groupby('order_id')['product_name'].apply(list).reset_index()
        test_baskets.columns = ['order_id', 'product_name']

    # Delegate to shared evaluation logic
    return _evaluate_baskets(rules, test_baskets, groupby_column, k, sample_size, min_basket_size)


def _evaluate_baskets(rules, test_baskets, groupby_column, k, sample_size, min_basket_size):
    """
    Core evaluation logic shared by evaluate_rules() and evaluate_rules_from_csv().
    
    Arguments:
        rules: DataFrame with association rules
        test_baskets: DataFrame [order_id, product_name, (groupby_column)]
        groupby_column: Optional column for grouped rule filtering
        k: Number of recommendations
        sample_size: Number of baskets to sample
        min_basket_size: Minimum basket size
    
    Returns:
        Dictionary with metrics or None
    """

    # Step 2: Filter baskets by minimum size
    # Need at least 4 products to split 50/50 (2 antecedents, 2 targets)
    test_baskets = test_baskets[test_baskets['product_name'].apply(len) >= min_basket_size]
    
    # Step 3: Sample if too many baskets
    if len(test_baskets) > sample_size:
        test_baskets = test_baskets.sample(sample_size, random_state=42)
    
    # Step 4: Evaluation loop (iterate through each test basket)
    results = []
    baskets_with_recs = 0
    
    for _, row in test_baskets.iterrows():
        basket = row['product_name']
        
        # Split basket 50/50
        split_point = len(basket) // 2
        antecedents_basket = set(basket[:split_point])     # Known items (input)
        actual_consequents = set(basket[split_point:])     # Target items (ground truth)
        
        # Filter rules by group if specified
        # Example: if basket is from 'produce', only use produce rules
        if groupby_column:
            group_value = row[groupby_column]
            applicable_rules = rules[rules[groupby_column] == group_value]
        else:
            applicable_rules = rules
        
        # Apply rules to generate recommendations
        recommendations = set()
        for _, rule in applicable_rules.iterrows():
            # Parse rule antecedents (IF part)
            rule_antecedents = set(rule['antecedent'].split(', '))
            
            # Check if rule applies: all rule antecedents must be in basket
            # Example: Rule {Banana, Milk} → {Yogurt} applies only if basket has both
            if rule_antecedents.issubset(antecedents_basket):
                # Add rule consequents to recommendations
                rule_consequents = set(rule['consequent'].split(', '))
                recommendations.update(rule_consequents)
        
        # Remove products already in basket (cannot recommend what user already has)
        recommendations = recommendations - antecedents_basket
        
        # Take top K recommendations (limit to k items)
        recommendations = list(recommendations)[:k]
        
        # Calculate metrics for this basket
        if len(recommendations) > 0:
            baskets_with_recs += 1
            
            # Calculate true positives (tp)
            # tp = products that are both recommended AND actually purchased
            hits = len(set(recommendations) & actual_consequents)
            
            # Calculate precision and recall
            # Precision = tp / (tp + fp) = hits / total_recommendations
            precision = hits / len(recommendations) if len(recommendations) > 0 else 0
            
            # Recall = tp / (tp + fn) = hits / total_target_items
            recall = hits / len(actual_consequents) if len(actual_consequents) > 0 else 0
            
            # Store results for this basket
            results.append({
                'precision': precision,
                'recall': recall,
                'hits': hits
            })
    
    # Step 5: Aggregate metrics across all baskets
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate average metrics
        metrics = {
            'precision@K': results_df['precision'].mean(),
            'recall@K': results_df['recall'].mean(),
            'coverage': baskets_with_recs / len(test_baskets),
            'avg_hits': results_df['hits'].mean(),
            'n_baskets': len(test_baskets),
            'n_baskets_with_recs': baskets_with_recs
        }
        
        return metrics
    else:
        # No recommendations generated for any basket
        return None


def print_evaluation_results(metrics):
    # Check if metrics exist
    if metrics is None:
        print("No recommendations generated")
        return
    
    # Print metrics
    print(f"  Precision@10: {metrics['precision@K']:.2%}")
    print(f"  Recall@10: {metrics['recall@K']:.2%}")
    print(f"  Coverage: {metrics['coverage']:.2%}")
    print(f"  Average hits: {metrics['avg_hits']:.2f}")
    print(f"  Baskets evaluated: {metrics['n_baskets']:,}")
    print(f"  Baskets with recommendations: {metrics['n_baskets_with_recs']:,}")