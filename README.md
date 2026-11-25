# Market Basket Analysis – Customer Purchase Pattern Analysis & Product Recommendation System

An interactive Python & Streamlit data analytics web dashboard that analyzes customer transaction data using **Exploratory Data Analysis (EDA)**, the **Apriori Algorithm**, and **Association Rule Mining** (`mlxtend`) to uncover co-purchase patterns and generate actionable business recommendations.

## Project Structure

```text
Market_Basket_Analysis/
│
├── app.py                        # Multi-page Streamlit interactive dashboard
│
├── data/
│   └── groceries.csv             # Sample grocery transaction dataset (750 transactions)
│
├── modules/
│   ├── __init__.py
│   ├── preprocessing.py          # CSV loading, column detection, cleaning, basket matrix
│   ├── eda.py                    # Transaction & product frequency statistics
│   ├── apriori_analysis.py       # Mlxtend Apriori frequent itemset generation
│   ├── association_rules.py      # Support, Confidence, Lift, Leverage, Conviction rules
│   └── business_insights.py      # Dynamic cross-sell, bundling, placement & recommender
│
├── visualizations/
│   ├── __init__.py
│   └── charts.py                 # Interactive Plotly, NetworkX, and Seaborn charts
│
├── requirements.txt              # Python dependencies
└── README.md                     # Documentation & execution guide
```

## Features

1. **Home Dashboard**: Executive KPIs (Total Transactions, Total Products, Product Combinations, Association Rules, Average Basket Size), top charts, and key takeaways.
2. **Dataset Explorer**: Upload custom CSV datasets, map Transaction ID & Product columns, audit data cleaning (missing, invalid, and duplicate rows), inspect the binary `1`/`0` basket matrix, and look up individual transactions.
3. **Product Analysis**: Top 10 & least purchased products, purchase share donut chart, basket size frequency bar chart & histogram, and searchable product frequencies.
4. **Frequent Itemsets**: Interactive Minimum Support slider, Max Itemset Length selector, product count filters, comparison bar chart, and tabular results.
5. **Association Rules**: Interactive Confidence and Lift filters, Support vs Confidence scatter plot, Lift comparison chart, interactive Product Relationship Network Graph, and Seaborn Lift Heatmap.
6. **Business Insights**: Interactive **Product Recommendation Simulator** plus dynamic recommendations for Cross-Selling, Product Bundling, Store Placement, Promotions, and Inventory Planning.
7. **Download Reports**: Export Frequent Itemsets, Association Rules, Cleaned Transactions, Basket Matrix (`CSV`), and an Executive Summary (`Markdown`).

## Setup & Run in VS Code

1. Open the `Market_Basket_Analysis` folder in VS Code.
2. Create or activate your Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

3. Launch the Streamlit web dashboard:

```bash
streamlit run app.py
```

4. Open the local URL shown in the terminal (typically `http://localhost:8501`).
