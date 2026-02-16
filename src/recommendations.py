import pandas as pd
from typing import List

class ProductRecommender:

    
    def __init__(self, rules_path: str):
        self.rules = pd.read_csv(rules_path)
        print(f"✅ {len(self.rules)} règles chargées")
        
        self.available_products = self._extract_all_products()
        print(f"✅ {len(self.available_products)} produits disponibles")
    
    def _extract_all_products(self) -> set:
        """Extrait tous les produits uniques des règles"""
        products = set()
        for _, row in self.rules.iterrows():
            products.update(row['antecedent'].split(', '))
            products.update(row['consequent'].split(', '))
        return products
    
    def recommend(self, products: List[str], top_n: int = 10) -> pd.DataFrame:

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
        
        # If no recommendations, return empty DataFrame
        if not scores:
            return pd.DataFrame(columns=['product_name', 'score'])
        
        # Transform in DataFrame and sort
        recommendations = pd.DataFrame(
            list(scores.items()), 
            columns=['product_name', 'score']
        ).sort_values('score', ascending=False).head(top_n)
        
        return recommendations
    
    def get_available_products(self) -> List[str]:
        return sorted(list(self.available_products))


# ============= EXEMPLE D'USAGE =============
if __name__ == "__main__":
    import os
    
    # Remonte à la racine du projet
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.dirname(current_dir)                # Retail-Data-Insights/
    rules_path = os.path.join(project_root, 'data', 'processed', 'rules_clean.csv')
    
    # Initialise le recommender
    recommender = ProductRecommender(rules_path)
    
    # Test
    basket = ['Banana', 'Organic Hass Avocado', 'Organic Strawberries']
    print(f"\n🛒 Panier : {basket}")
    
    recs = recommender.recommend(basket, top_n=10)
    print("\n🎯 Recommandations :")
    print(recs.to_string(index=False))