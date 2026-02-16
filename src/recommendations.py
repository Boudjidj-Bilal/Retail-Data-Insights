import pandas as pd
import os
from pathlib import Path

class ProductRecommender:
    
    def __init__(self, rules_path):
        if not os.path.isabs(rules_path):
            # Find the absolute path
            current_file = Path(__file__).resolve()
            project_root = current_file.parent.parent  # src/ -> Retail-Data-Insights/
            rules_path = project_root / rules_path
        
        # Check file existing
        if not os.path.exists(rules_path):
            raise FileNotFoundError(f"File not found: {rules_path}")
        
        self.rules = pd.read_csv(rules_path) 
        print(f"{len(self.rules)} rules loaded")
        
        self.available_products = self._extract_all_products()
        print(f"{len(self.available_products)} products available for recommendation")
    
    def _extract_all_products(self): # products are string separated by ", "
        products = set()
        for _, row in self.rules.iterrows():
            products.update(row['antecedent'].split(', '))
            products.update(row['consequent'].split(', '))
        return products
    
    def recommend(self, products, top_n=10):
        if not products:
            return pd.DataFrame(columns=['product_name', 'score'])
        
        input_products = set(products)
        scores = {}
        
        for _, rule in self.rules.iterrows():
            antecedents = set(rule['antecedent'].split(', '))
            consequents = set(rule['consequent'].split(', '))
            
            if antecedents & input_products:
                for product in consequents:
                    if product not in input_products:
                        if product in scores:
                            scores[product] = max(scores[product], rule['lift'])
                        else:
                            scores[product] = rule['lift']
        
        if not scores:
            return pd.DataFrame(columns=['product_name', 'score'])
        
        recommendations = pd.DataFrame(
            list(scores.items()), 
            columns=['product_name', 'score']
        ).sort_values('score', ascending=False).head(top_n)
        
        return recommendations
    
    def get_available_products(self):
        return sorted(list(self.available_products))