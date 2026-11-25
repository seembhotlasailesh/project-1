"""Association rule mining and filtering using mlxtend."""

from __future__ import annotations

import pandas as pd
from mlxtend.frequent_patterns import association_rules

from .apriori_analysis import format_itemset


_RULE_COLUMNS = [
    "antecedents",
    "consequents",
    "antecedents_str",
    "consequents_str",
    "rule",
    "antecedent_len",
    "consequent_len",
    "antecedent support",
    "consequent support",
    "support",
    "confidence",
    "lift",
    "leverage",
    "conviction",
]


def generate_rules(
    frequent_itemsets: pd.DataFrame,
    min_confidence: float = 0.3,
    min_lift: float = 1.0,
) -> pd.DataFrame:
    """Generate and filter association rules from frequent itemsets."""
    if frequent_itemsets.empty or (frequent_itemsets["itemset_length"] < 2).all():
        return pd.DataFrame(columns=_RULE_COLUMNS)

    if not 0 < min_confidence <= 1:
        raise ValueError("min_confidence must be between 0 (exclusive) and 1 (inclusive).")

    rules = association_rules(
        frequent_itemsets[["itemsets", "support"]],
        metric="confidence",
        min_threshold=min_confidence,
    )

    if rules.empty:
        return pd.DataFrame(columns=_RULE_COLUMNS)

    rules = rules[rules["lift"] >= min_lift].copy()
    if rules.empty:
        return pd.DataFrame(columns=_RULE_COLUMNS)

    rules["antecedents_str"] = rules["antecedents"].apply(format_itemset)
    rules["consequents_str"] = rules["consequents"].apply(format_itemset)
    rules["rule"] = rules["antecedents_str"] + " -> " + rules["consequents_str"]
    rules["antecedent_len"] = rules["antecedents"].apply(len).astype(int)
    rules["consequent_len"] = rules["consequents"].apply(len).astype(int)

    numeric_cols = [
        "antecedent support",
        "consequent support",
        "support",
        "confidence",
        "lift",
        "leverage",
        "conviction",
    ]
    for col in numeric_cols:
        if col in rules.columns:
            rules[col] = pd.to_numeric(rules[col], errors="coerce").round(4)

    rules = rules.sort_values(
        by=["lift", "confidence", "support"],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    return rules[_RULE_COLUMNS]
