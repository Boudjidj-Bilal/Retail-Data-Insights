# recommendations.py

Product recommendation system based on association rules.

# Usage
```python
from recommendations import ProductRecommender

recommender = ProductRecommender('data/processed/rules_clean.csv')
recs = recommender.recommend(['Banana', 'Milk'], top_n=10)
```

# Methods

- `recommend(products, top_n=10)`: Generate recommendations
- `get_available_products()`: List all available products

# Input Format

CSV file with columns: antecedent, consequent, lift, support, confidence

# Installation des dépendances

```python

pip install streamlit pandas plotly
```

# Lancement du dashboard

```python

streamlit run app.py
```