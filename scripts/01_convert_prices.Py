# ============================================================================
# SCRIPT 1: CONVERT PRICES TO EUR (RUN ONCE)
# ============================================================================
# 
# This script converts product_with_price(in).csv to EUR
# Run once when new pricing data arrives
# Output: products_priced_eur.csv (used by main pipeline)
#
# Project Alignment: Project2_PythonML_A25.pdf Section 4
# "Data engineers could/may enrich the dataset by crawling real-world 
# product prices... This would allow us to calculate the actual monetary 
# value of associations."
# ============================================================================

import pandas as pd
from forex_python.converter import CurrencyRates
from datetime import datetime
import os

print("="*70)
print("CURRENCY CONVERSION SCRIPT")
print("="*70)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# File paths
INPUT_FILE = './data/product_with_price(in).csv'
OUTPUT_FILE = './data/products_priced_eur.csv'

# Check if input file exists
if not os.path.exists(INPUT_FILE):
    print(f"\n Error: {INPUT_FILE} not found!")
    print("Please ensure product_with_price(in).csv is in ./data/")
    exit()

# Load pricing data
print(f"\n[1/4] Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"  ✓ Loaded {len(df):,} products")

# Initialize currency converter
print("\n[2/4] Initializing currency converter...")
c = CurrencyRates()
print("Ready")

# Convert prices to EUR
print("\n[3/4] Converting prices to EUR...")

def convert_to_eur(row):
    currency = str(row.get('currency', 'USD')).upper().strip()
    price = row.get('price', 0)
    
    if pd.isna(price):
        return 0
    
    if currency == 'EUR':
        return round(price, 2)
    
    try:
        rate = c.get_rate(currency, 'EUR')
        return round(price * rate, 2)
    except:
        return round(price, 2)

df['price_eur'] = df.apply(convert_to_eur, axis=1)

# Save converted data
print("\n[4/4] Saving converted data...")
df.to_csv(OUTPUT_FILE, index=False)
print(f"Saved to: {OUTPUT_FILE}")

# Summary
print("\n" + "="*70)
print("CONVERSION COMPLETE")
print("="*70)
print(f"Products converted: {len(df):,}")
print(f"Price range (EUR): €{df['price_eur'].min():.2f} - €{df['price_eur'].max():.2f}")
print(f"Average price (EUR): €{df['price_eur'].mean():.2f}")
print(f"\nNext step: Run main pipeline (scripts/02_run_pipeline.py)")