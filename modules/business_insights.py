"""Dynamic business insights and recommendation engine derived from association rules."""

from __future__ import annotations

from typing import Any

import pandas as pd


def _format_rule_entry(row: pd.Series, category_note: str) -> dict[str, Any]:
    return {
        "antecedent": row["antecedents_str"],
        "consequent": row["consequents_str"],
        "rule": row["rule"],
        "support": float(row["support"]),
        "confidence": float(row["confidence"]),
        "lift": float(row["lift"]),
        "summary": category_note,
    }


def generate_business_insights(
    rules_df: pd.DataFrame,
    eda_summary: dict[str, Any],
    frequent_itemsets_df: pd.DataFrame,
) -> dict[str, Any]:
    """Generate dynamic cross-sell, bundling, placement, promo, and inventory insights."""
    top_products = eda_summary.get("top_products", pd.DataFrame())
    positive_rules = (
        rules_df[rules_df["lift"] > 1.0].copy() if not rules_df.empty else pd.DataFrame()
    )

    cross_sell: list[dict[str, Any]] = []
    bundling: list[dict[str, Any]] = []
    placement: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []

    if not positive_rules.empty:
        # Cross-selling: highest confidence rules with meaningful lift
        cs_candidates = positive_rules.sort_values(
            by=["confidence", "lift"], ascending=[False, False]
        ).head(5)
        for _, row in cs_candidates.iterrows():
            note = (
                f"{row['confidence'] * 100:.1f}% of baskets with '{row['antecedents_str']}' "
                f"also include '{row['consequents_str']}' (Lift {row['lift']:.2f}x). "
                f"Recommend '{row['consequents_str']}' at checkout or on product pages."
            )
            cross_sell.append(_format_rule_entry(row, note))

        # Bundling: strong support + lift, deduplicated by symmetric itemsets
        seen_pairs: set[frozenset[str]] = set()
        bundle_candidates = positive_rules[positive_rules["lift"] >= 1.15].sort_values(
            by=["support", "lift"], ascending=[False, False]
        )
        for _, row in bundle_candidates.iterrows():
            pair_key = frozenset(row["antecedents"]) | frozenset(row["consequents"])
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            combo_items = ", ".join(sorted(pair_key))
            note = (
                f"'{combo_items}' appear together in {row['support'] * 100:.1f}% of all baskets "
                f"with {row['lift']:.2f}x lift. Package as a value combo bundle."
            )
            bundling.append(_format_rule_entry(row, note))
            if len(bundling) >= 5:
                break

        # Store / category placement: highest lift pairs
        seen_placement: set[frozenset[str]] = set()
        place_candidates = positive_rules.sort_values(
            by=["lift", "support"], ascending=[False, False]
        )
        for _, row in place_candidates.iterrows():
            pair_key = frozenset(row["antecedents"]) | frozenset(row["consequents"])
            if pair_key in seen_placement:
                continue
            seen_placement.add(pair_key)
            note = (
                f"Strong co-purchase affinity ({row['lift']:.2f}x lift): co-locate "
                f"'{row['antecedents_str']}' near '{row['consequents_str']}' or create end-cap displays."
            )
            placement.append(_format_rule_entry(row, note))
            if len(placement) >= 5:
                break

        # Promotional offers: moderate support & high confidence (ideal for targeted coupons)
        promo_candidates = positive_rules.sort_values(
            by=["leverage", "confidence"], ascending=[False, False]
        ).head(5)
        for _, row in promo_candidates.iterrows():
            note = (
                f"Offer a targeted discount on '{row['consequents_str']}' when customers buy "
                f"'{row['antecedents_str']}' ({row['confidence'] * 100:.1f}% conversion affinity)."
            )
            promotions.append(_format_rule_entry(row, note))

    # Inventory planning: top high-velocity products + their strongest companion items
    if not top_products.empty:
        for _, prod_row in top_products.head(5).iterrows():
            product = str(prod_row["Product"])
            count = int(prod_row["Count"])
            support_pct = float(prod_row["Support (%)"])
            companion = None
            if not positive_rules.empty:
                matches = positive_rules[
                    positive_rules["antecedents"].apply(lambda s: product in s)
                ]
                if not matches.empty:
                    top_match = matches.iloc[0]
                    companion = (
                        f"Pair stock alerts with '{top_match['consequents_str']}' "
                        f"({top_match['confidence'] * 100:.1f}% co-purchase confidence)."
                    )
            inventory.append(
                {
                    "product": product,
                    "count": count,
                    "support_pct": support_pct,
                    "summary": (
                        f"Anchor item appearing in {support_pct:.1f}% of transactions ({count} orders). "
                        + (companion or "Maintain high safety stock to prevent lost basket trips.")
                    ),
                }
            )

    multi_itemsets_count = (
        int((frequent_itemsets_df["itemset_length"] >= 2).sum())
        if not frequent_itemsets_df.empty
        else 0
    )

    return {
        "cross_sell": cross_sell,
        "bundling": bundling,
        "placement": placement,
        "promotions": promotions,
        "inventory": inventory,
        "stats": {
            "positive_rules": int(len(positive_rules)),
            "multi_itemsets": multi_itemsets_count,
            "avg_lift": round(float(positive_rules["lift"].mean()), 2)
            if not positive_rules.empty
            else 0.0,
            "max_lift": round(float(positive_rules["lift"].max()), 2)
            if not positive_rules.empty
            else 0.0,
        },
    }


def recommend_for_basket(
    rules_df: pd.DataFrame,
    selected_products: list[str],
    top_n: int = 8,
) -> pd.DataFrame:
    """Return ranked product recommendations given one or more products in a customer's basket."""
    cols = ["Recommended Product", "Triggered By", "Confidence (%)", "Lift", "Support (%)"]
    if rules_df.empty or not selected_products:
        return pd.DataFrame(columns=cols)

    basket_set = set(selected_products)
    matched = rules_df[
        rules_df["antecedents"].apply(lambda ant: ant.issubset(basket_set))
        & rules_df["consequents"].apply(lambda con: len(con - basket_set) > 0)
    ].copy()

    if matched.empty:
        return pd.DataFrame(columns=cols)

    rows = []
    for _, row in matched.iterrows():
        new_items = sorted(row["consequents"] - basket_set)
        for item in new_items:
            rows.append(
                {
                    "Recommended Product": item,
                    "Triggered By": row["antecedents_str"],
                    "Confidence (%)": round(float(row["confidence"]) * 100, 2),
                    "Lift": round(float(row["lift"]), 3),
                    "Support (%)": round(float(row["support"]) * 100, 2),
                }
            )

    rec_df = pd.DataFrame(rows)
    if rec_df.empty:
        return pd.DataFrame(columns=cols)

    rec_df = (
        rec_df.sort_values(by=["Lift", "Confidence (%)"], ascending=[False, False])
        .drop_duplicates(subset=["Recommended Product"], keep="first")
        .head(top_n)
        .reset_index(drop=True)
    )
    return rec_df
