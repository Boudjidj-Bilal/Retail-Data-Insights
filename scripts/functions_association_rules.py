"""
Association Rules Generation and Evaluation Functions

This module provides utilities for:
1. Generating association rules using FP-Growth algorithm
2. Preparing transaction data from CSV files (chunk-based, memory-efficient)
3. Evaluating rules using offline hold-out validation
4. Comparing multiple recommendation approaches

All data processing functions read CSV files chunk by chunk to avoid loading the full dataset into memory.

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


# CHUNK-BASED Functions - Read CSV without loading everything in memory

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

    Makes a single pass through the CSV to count product frequencies, without loading the entire file into memory.

    Arguments:
        filepath: Path to the CSV file
        top_n: Number of top products to keep
        filter_column: Optional column to filter on
        filter_values: Value(s) to keep
        chunksize: Number of rows per chunk

    Returns:
        Set of top N product names
    """
    print(f"    Computing top {top_n} products...")
    product_counts = {}

    for chunk in csv_chunk_generator(filepath, chunksize, filter_column, filter_values):
        for product, count in chunk['product_name'].value_counts().items():
            product_counts[product] = product_counts.get(product, 0) + count

    top_products = set(
        sorted(product_counts, key=product_counts.get, reverse=True)[:top_n]
    )
    return top_products


# TRANSACTION PREPARATION

def prepare_transactions_from_csv(filepath, filter_column=None, filter_values=None,
                                   top_n_products=None, chunksize=100_000):
    """
    Prepare transaction list from a CSV file using chunks (memory-efficient).

    Converts long-format CSV (one row per product per order) into a list of transactions (one list of products per order), without loading the entire file into memory.

    Makes up to 2 passes through the CSV:
    - Pass 1 (if top_n_products specified): count product frequencies to identify top N
    - Pass 2: build the transaction dict {order_id -> [products]}

    Arguments:
        filepath: Path to the CSV file (must have 'order_id' and 'product_name' columns)
        filter_column: Optional column to filter on ('department', 'segment')
        filter_values: Value(s) to keep
        top_n_products: Keep only top N most frequent products
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

    # Pass 2: build transactions dict chunk by chunk
    print(f"    Building transactions...")
    transactions_dict = {}

    for chunk in csv_chunk_generator(filepath, chunksize, filter_column, filter_values):
        # Keep only top N products if specified
        if top_products:
            chunk = chunk[chunk['product_name'].isin(top_products)]

        # Group by order and accumulate products
        for order_id, group in chunk.groupby('order_id'):
            if order_id not in transactions_dict:
                transactions_dict[order_id] = []
            transactions_dict[order_id].extend(group['product_name'].tolist())

    return list(transactions_dict.values())


# RULE GENERATION - FP-Growth algorithm

def generate_association_rules(transactions, min_support=0.005, min_confidence=0.15,
                                min_lift=1.3, max_transactions=None):
    """
    Generate association rules from a transaction list using FP-Growth algorithm.

    Process:
    1. Encode transactions to binary matrix (1 if product in order, 0 otherwise)
    2. Find frequent itemsets using FP-Growth
    3. Generate association rules from frequent itemsets
    4. Filter rules by confidence and lift thresholds

    Arguments:
        transactions: List of lists, each containing product names
                     Example: [['Banana', 'Milk'], ['Banana', 'Bread', 'Eggs']]

        min_support: Minimum support threshold
                    Support = P(A inter B) = proportion of transactions containing the itemset

        min_confidence: Minimum confidence threshold
                       Confidence = P(B|A) = probability of B given A

        min_lift: Minimum lift threshold
                 Lift = P(A inter B) / (P(A) x P(B))
                 Lift = 1.3 means 30% more likely to co-occur than by chance

        max_transactions: Maximum number of transactions to use.
                         If exceeded, a random sample is taken.
                         Used to limit computation time and memory usage.

    Returns:
        DataFrame with columns [antecedent, consequent, support, confidence, lift]
        Returns None if no rules are found.
    """
    # Sample transactions if dataset too large
    if max_transactions and len(transactions) > max_transactions:
        transactions = random.sample(transactions, max_transactions)

    # Step 1: Encode transactions to binary matrix
    # Each row = one transaction, each column = one product (1 if present, 0 otherwise)
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # Step 2: Find frequent itemsets using FP-Growth
    # Returns all combinations of products appearing together at least min_support times
    frequent_itemsets = fpgrowth(df_encoded, min_support=min_support, use_colnames=True)

    if len(frequent_itemsets) == 0:
        return None

    # Step 3: Generate association rules from frequent itemsets
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)

    if len(rules) == 0:
        return None

    # Step 4: Clean and filter rules
    # Convert frozensets to readable strings
    # e.g. frozenset({'Banana', 'Milk'}) -> 'Banana, Milk'
    rules['antecedent'] = rules['antecedents'].apply(lambda x: ', '.join(sorted(x)))
    rules['consequent'] = rules['consequents'].apply(lambda x: ', '.join(sorted(x)))

    # Filter by lift threshold
    rules_filtered = rules[rules['lift'] > min_lift].copy()

    # Keep only essential columns
    rules_filtered = rules_filtered[['antecedent', 'consequent', 'support', 'confidence', 'lift']]

    # Free memory
    del df_encoded, frequent_itemsets, rules
    gc.collect()

    return rules_filtered



# RULE EVALUATION


def evaluate_rules_from_csv(rules, filepath, groupby_column=None, k=10,
                             sample_size=10_000, min_basket_size=4, chunksize=100_000):
    """
    Evaluate association rules using offline hold-out validation methodology.

    Source: Shani & Gunawardana (2011) - Evaluating Recommendation Systems

    Reads test data chunk by chunk from a CSV file (memory-efficient).

    EVALUATION METHODOLOGY

    1. BASKET AGGREGATION
       Build one basket per order_id from the CSV.
       If groupby_column is specified, keep the first value encountered per basket (for example the department or segment assigned to that order) :
            - for segment it's relevant because one order = one customer = one segment
            - for department it's less relevant because one order can contain multiple departments, but we keep it for simplicity

    2. BASKET SPLIT (50/50)
       Each basket is split into:
       - Known items (first 50%): Input to the recommendation system
       - Target items (last 50%): Actual purchases used for evaluation

    3. RULE APPLICATION
       Apply association rules to known items:
       - For each rule: IF rule_antecedent is subset of known_items THEN recommend rule_consequent
       - If groupby_column: only apply rules matching the basket's group value
       - Aggregate all recommendations (up to K items)

    4. METRICS CALCULATION
       Compare recommendations against target items:

                          | Recommended | Not Recommended |
       --------------------------------------------------------
       Purchased          |     tp      |       fn        |
       Not Purchased      |     fp      |       tn        |

       where:
       - tp (true positives)  = products recommended AND purchased
       - fp (false positives) = products recommended but NOT purchased
       - fn (false negatives) = products purchased but NOT recommended

       In our implementation:
       tp = intersection of recommendations and target items

    METRICS COMPUTED
    ================
    K = number of recommendations generated (here K=10)

    Precision@K = tp / K = |recommended & purchased| / K
    - Measures: What proportion of recommendations are correct?
    - Range: [0, 1], higher is better

    Recall@K = tp / |target| = |recommended & purchased| / |target|
    - Measures: What proportion of purchases were recommended?
    - Range: [0, 1], higher is better

    Coverage = baskets_with_recs / total_baskets
    - Measures: What proportion of baskets receive recommendations?
    - Range: [0, 1], higher is better

    Arguments:
        rules: DataFrame with columns [antecedent, consequent] and optional groupby_column
               Each row represents one rule: IF antecedent THEN consequent

        filepath: Path to the test CSV file
                 Must have columns [order_id, product_name] and optional groupby_column

        groupby_column: Optional column for grouped evaluation ('department', 'segment')
                       If specified, only rules matching the basket's group are applied
                       Example: 'department' -> apply only department-specific rules
                       If None, all rules are applied to all baskets

        k: Number of recommendations to generate per basket (default: 10)

        sample_size: Number of test baskets to evaluate (default: 10_000)
                    Balances evaluation speed vs statistical significance

        min_basket_size: Minimum basket size for evaluation (default: 4)
                        Baskets with fewer products cannot be split 50/50 meaningfully

        chunksize: Number of rows per chunk for CSV reading

    Returns:
        Dictionary with metrics:
        {
            'precision@K': float,
            'recall@K': float,
            'coverage': float,
            'avg_hits': float,
            'n_baskets': int,
            'n_baskets_with_recs': int
        }
        Returns None if no recommendations were generated.
    """
    print("    Building test baskets from CSV...")

    # Step 1: Build baskets dict from CSV chunks
    # Same aggregation logic as a groupby:
    # - baskets_dict accumulates products per order (equivalent to agg list)
    # - group_dict keeps the FIRST value seen per order (equivalent to agg 'first') :

    baskets_dict = {}   # order_id -> list of products
    group_dict   = {}   # order_id -> first value of groupby_column 

    for chunk in csv_chunk_generator(filepath, chunksize):
        for order_id, group in chunk.groupby('order_id'):
            if order_id not in baskets_dict:
                baskets_dict[order_id] = []
                if groupby_column and order_id not in group_dict:
                    group_dict[order_id] = group[groupby_column].iloc[0]
            baskets_dict[order_id].extend(group['product_name'].tolist())

    # Reconstruct DataFrame with same structure as a groupby result
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


def _evaluate_baskets(rules, test_baskets, groupby_column, k, sample_size, min_basket_size):
    """
    Core evaluation logic — see evaluate_rules_from_csv() for full methodology.

    Arguments:
        rules: DataFrame with association rules
        test_baskets: DataFrame [order_id, product_name, (groupby_column)]
                     product_name column must contain lists of products
        groupby_column: Optional column for grouped rule filtering
        k: Number of recommendations per basket
        sample_size: Number of baskets to evaluate 
        min_basket_size: Minimum basket size for evaluation

    Returns:
        Dictionary with metrics or None if no recommendations generated
    """

    # Step 2: Filter baskets by minimum size
    # Need at least min_basket_size products to split 50/50 meaningfully
    test_baskets = test_baskets[test_baskets['product_name'].apply(len) >= min_basket_size]

    if len(test_baskets) == 0:
        return None

    # Step 3: Sample if too many baskets
    if len(test_baskets) > sample_size:
        test_baskets = test_baskets.sample(sample_size, random_state=42)

    # Step 4: Evaluation loop — iterate through each test basket
    results = []
    baskets_with_recs = 0

    for _, row in test_baskets.iterrows():
        basket = row['product_name']

        # Split basket 50/50
        split_point = len(basket) // 2
        antecedents_basket = set(basket[:split_point])   # Known items (input)
        actual_consequents = set(basket[split_point:])   # Target items

        # Filter rules by group if specified
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

            # Check if rule applies: all rule antecedents must be in known basket
            if rule_antecedents.issubset(antecedents_basket):
                # Add rule consequents to recommendations
                rule_consequents = set(rule['consequent'].split(', '))
                recommendations.update(rule_consequents)

        # Remove products already in the known basket
        recommendations = recommendations - antecedents_basket

        # Limit to top K recommendations
        recommendations = list(recommendations)[:k]

        # Compute metrics for this basket
        if len(recommendations) > 0:
            baskets_with_recs += 1

            # True positives: recommended AND actually purchased
            hits = len(set(recommendations) & actual_consequents)

            # Precision@K = tp / K = hits / total recommendations
            precision = hits / len(recommendations)

            # Recall@K = tp / |target| = hits / total target items
            recall = hits / len(actual_consequents) if len(actual_consequents) > 0 else 0

            results.append({'precision': precision, 'recall': recall, 'hits': hits})

    # Step 5: Aggregate metrics across all evaluated baskets
    if results:
        results_df = pd.DataFrame(results)
        return {
            'precision@K':         results_df['precision'].mean(),
            'recall@K':            results_df['recall'].mean(),
            'coverage':            baskets_with_recs / len(test_baskets),
            'avg_hits':            results_df['hits'].mean(),
            'n_baskets':           len(test_baskets),
            'n_baskets_with_recs': baskets_with_recs
        }
    else:
        return None



# DISPLAY


def print_evaluation_results(metrics):

    if metrics is None:
        print("  No recommendations generated")
        return

    print(f"  Precision@10:              {metrics['precision@K']:.2%}")
    print(f"  Recall@10:                 {metrics['recall@K']:.2%}")
    print(f"  Coverage:                  {metrics['coverage']:.2%}")
    print(f"  Average hits:              {metrics['avg_hits']:.2f}")
    print(f"  Baskets evaluated:         {metrics['n_baskets']:,}")
    print(f"  Baskets with recs:         {metrics['n_baskets_with_recs']:,}")