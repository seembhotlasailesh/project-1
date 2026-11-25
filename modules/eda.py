"""Exploratory Data Analysis calculations for transaction baskets."""

from __future__ import annotations

from typing import Any

import pandas as pd


def compute_eda_summary(
    cleaned_df: pd.DataFrame,
    basket_matrix: pd.DataFrame,
    txn_col: str,
    item_col: str,
) -> dict[str, Any]:
    """Compute core EDA metrics, product frequencies, and basket size distributions."""
    if cleaned_df.empty or basket_matrix.empty:
        empty_freq = pd.DataFrame(columns=["Product", "Count", "Support (%)", "Share of Purchases (%)"])
        empty_dist = pd.DataFrame(columns=["Items in Basket", "Transaction Count", "Percentage (%)"])
        return {
            "total_transactions": 0,
            "total_products": 0,
            "total_items_purchased": 0,
            "avg_items_per_transaction": 0.0,
            "median_items_per_transaction": 0.0,
            "max_items_per_transaction": 0,
            "min_items_per_transaction": 0,
            "product_frequency": empty_freq,
            "top_products": empty_freq,
            "least_products": empty_freq,
            "items_per_transaction": pd.Series(dtype=int),
            "transaction_size_distribution": empty_dist,
        }

    total_transactions = int(basket_matrix.shape[0])
    total_products = int(basket_matrix.shape[1])
    total_items_purchased = int(len(cleaned_df))

    items_per_txn = basket_matrix.sum(axis=1).astype(int)
    avg_items = float(items_per_txn.mean())
    median_items = float(items_per_txn.median())
    max_items = int(items_per_txn.max())
    min_items = int(items_per_txn.min())

    counts = basket_matrix.sum(axis=0).sort_values(ascending=False)
    product_freq = pd.DataFrame(
        {
            "Product": counts.index.astype(str),
            "Count": counts.values.astype(int),
        }
    )
    product_freq["Support (%)"] = (product_freq["Count"] / total_transactions * 100).round(2)
    product_freq["Share of Purchases (%)"] = (
        product_freq["Count"] / max(total_items_purchased, 1) * 100
    ).round(2)

    size_counts = items_per_txn.value_counts().sort_index()
    txn_size_dist = pd.DataFrame(
        {
            "Items in Basket": size_counts.index.astype(int),
            "Transaction Count": size_counts.values.astype(int),
        }
    )
    txn_size_dist["Percentage (%)"] = (
        txn_size_dist["Transaction Count"] / total_transactions * 100
    ).round(2)

    return {
        "total_transactions": total_transactions,
        "total_products": total_products,
        "total_items_purchased": total_items_purchased,
        "avg_items_per_transaction": round(avg_items, 2),
        "median_items_per_transaction": round(median_items, 2),
        "max_items_per_transaction": max_items,
        "min_items_per_transaction": min_items,
        "product_frequency": product_freq,
        "top_products": product_freq.head(10).reset_index(drop=True),
        "least_products": product_freq.sort_values(by=["Count", "Product"], ascending=[True, True])
        .head(10)
        .reset_index(drop=True),
        "items_per_transaction": items_per_txn,
        "transaction_size_distribution": txn_size_dist,
    }
