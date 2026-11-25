"""Apriori frequent itemset mining wrapper around mlxtend."""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import apriori


def format_itemset(itemset: frozenset[str] | set[str] | list[str]) -> str:
    """Format an itemset collection into a deterministic comma-separated label."""
    return ", ".join(sorted(str(item) for item in itemset))


def run_apriori(
    basket_matrix: pd.DataFrame,
    min_support: float = 0.05,
    max_len: int | None = None,
) -> pd.DataFrame:
    """Run Apriori on a binary basket matrix and return frequent itemsets with support metrics."""
    columns = ["itemsets", "itemsets_str", "itemset_length", "support", "support_pct"]
    if basket_matrix.empty:
        return pd.DataFrame(columns=columns)

    if not 0 < min_support <= 1:
        raise ValueError("min_support must be between 0 (exclusive) and 1 (inclusive).")

    basket_bool = basket_matrix.astype(bool)
    frequent = apriori(
        basket_bool,
        min_support=min_support,
        use_colnames=True,
        max_len=max_len,
    )

    if frequent.empty:
        return pd.DataFrame(columns=columns)

    frequent = frequent.copy()
    frequent["itemset_length"] = frequent["itemsets"].apply(len).astype(int)
    frequent["itemsets_str"] = frequent["itemsets"].apply(format_itemset)
    frequent["support"] = frequent["support"].astype(float).round(4)
    frequent["support_pct"] = (frequent["support"] * 100).round(2)

    frequent = frequent.sort_values(
        by=["support", "itemset_length", "itemsets_str"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    return frequent[columns]
