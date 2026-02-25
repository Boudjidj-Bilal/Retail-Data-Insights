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
