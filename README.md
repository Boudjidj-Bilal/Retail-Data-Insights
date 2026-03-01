# 🛒 Retail Data Insights
A data-driven product recommendation system built on the Instacart Market Basket dataset, combining customer segmentation, association rules mining, and price integration into a unified Streamlit web application.

> **Live app:** https://app-retail-insights-qhlmfeqvell293ibta37k6.streamlit.app/

---

### 1. Clone the repository
```bash
git clone https://github.com/Boudjidj-Bilal/Retail-Data-Insights.git
cd Retail-Data-Insights
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```
Or manually:
```bash
pip install streamlit pandas plotly mlxtend scikit-learn numpy
```

### 3. Download the dataset
Download the Instacart dataset from [Kaggle](https://www.kaggle.com/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset) and place the CSV files in `data/raw/`.

### 4. Run the notebooks in order
1. `EDA_instacart.ipynb`
2. `03_RFM_Customer_Segmentation.ipynb`
3. `04_Discount_Strategy.ipynb`
4. `association_rules.ipynb`
5. `Marketing_pipeline_FINAL_V5.ipynb`

### 5. Launch the app
```bash
streamlit run app/app.py
```

---

## Team contributions

| Author | Contribution |
|--------|-------------|
| Kali Volle | RFM segmentation, discount strategy, financial pipeline |
| Antoine Guibert | EDA, association rules mining |
| Guillaume Lopes Da Silva | Department analysis |
| Bilal Boudjidj | Price scraping, fuzzy matching |
| Benoit Gianni | APP

---

## References
- Kobets & Yashyna (2025). DOI: [10.15276/mdt.9.3.2025.3](https://doi.org/10.15276/mdt.9.3.2025.3)
- Wamsler et al. (2024). DOI: [10.1007/s00291-022-00685-w](https://doi.org/10.1007/s00291-022-00685-w)
- Gupta & Zeithaml (2006). DOI: [10.1287/mksc.1060.0221](https://doi.org/10.1287/mksc.1060.0221)
