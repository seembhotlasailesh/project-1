"""
Market Basket Analysis - Customer Purchase Pattern Analysis and Product Recommendation System.

Interactive Streamlit dashboard built with Pandas, Mlxtend (Apriori & Association Rules),
Plotly, Matplotlib, Seaborn, and NetworkX.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from modules import (
    clean_transaction_data,
    compute_eda_summary,
    create_basket_matrix,
    detect_columns,
    generate_business_insights,
    generate_rules,
    load_csv_data,
    recommend_for_basket,
    run_apriori,
)
from visualizations import (
    plot_frequent_itemsets_bar,
    plot_least_products_bar,
    plot_lift_comparison_bar,
    plot_product_distribution_pie,
    plot_product_network_graph,
    plot_seaborn_lift_heatmap,
    plot_support_vs_confidence_scatter,
    plot_top_products_bar,
    plot_transaction_frequency_bar,
    plot_transaction_size_histogram,
)

APP_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET_PATH = APP_DIR / "data" / "groceries.csv"

PAGES = [
    "Home Dashboard",
    "Dataset Explorer",
    "Product Analysis",
    "Frequent Itemsets",
    "Association Rules",
    "Business Insights",
    "Download Reports",
]


def _inject_custom_styles() -> None:
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2rem;
        }
        div[data-testid="stMetric"] {
            background-color: rgba(148, 163, 184, 0.08);
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 12px;
            padding: 14px 16px;
        }
        .insight-card {
            background-color: rgba(37, 99, 235, 0.06);
            border-left: 4px solid #2563EB;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .insight-card-green {
            background-color: rgba(16, 185, 129, 0.07);
            border-left: 4px solid #10B981;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .insight-card-amber {
            background-color: rgba(245, 158, 11, 0.08);
            border-left: 4px solid #F59E0B;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
        }
        .section-caption {
            color: #64748B;
            font-size: 0.92rem;
            margin-top: -0.4rem;
            margin-bottom: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "min_support": 0.03,
        "max_itemset_len": 3,
        "min_confidence": 0.35,
        "min_lift": 1.05,
        "uploaded_bytes": None,
        "uploaded_name": None,
        "dataset_key": "default",
        "txn_col": None,
        "item_col": None,
        "active_page": "Home Dashboard",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


@st.cache_data(show_spinner=False)
def _cached_load_default() -> pd.DataFrame:
    return load_csv_data(DEFAULT_DATASET_PATH)


@st.cache_data(show_spinner=False)
def _cached_load_uploaded(raw_bytes: bytes) -> pd.DataFrame:
    return load_csv_data(raw_bytes)


@st.cache_data(show_spinner=False)
def _cached_preprocess(
    df: pd.DataFrame, txn_col: str, item_col: str
) -> tuple[pd.DataFrame, dict[str, Any], pd.DataFrame, dict[str, Any]]:
    cleaned_df, clean_stats = clean_transaction_data(df, txn_col, item_col)
    basket_matrix = create_basket_matrix(cleaned_df, txn_col, item_col)
    eda_summary = compute_eda_summary(cleaned_df, basket_matrix, txn_col, item_col)
    return cleaned_df, clean_stats, basket_matrix, eda_summary


@st.cache_data(show_spinner=False)
def _cached_apriori(
    basket_matrix: pd.DataFrame, min_support: float, max_len: int | None
) -> pd.DataFrame:
    return run_apriori(basket_matrix, min_support=min_support, max_len=max_len)


@st.cache_data(show_spinner=False)
def _cached_rules(
    freq_itemsets: pd.DataFrame, min_confidence: float, min_lift: float
) -> pd.DataFrame:
    return generate_rules(freq_itemsets, min_confidence=min_confidence, min_lift=min_lift)


def _load_active_raw_df() -> tuple[pd.DataFrame, str]:
    if st.session_state["uploaded_bytes"] is not None:
        df = _cached_load_uploaded(st.session_state["uploaded_bytes"])
        return df, st.session_state["uploaded_name"] or "Uploaded CSV"
    if not DEFAULT_DATASET_PATH.exists():
        st.error(f"Default dataset not found at `{DEFAULT_DATASET_PATH}`.")
        st.stop()
    return _cached_load_default(), "groceries.csv (Default Sample Dataset)"


def _sync_column_selection(raw_df: pd.DataFrame) -> tuple[str, str]:
    cols = [str(c) for c in raw_df.columns]
    if len(cols) < 2:
        st.error("Dataset must contain at least two columns (Transaction ID and Product).")
        st.stop()

    detected_txn, detected_item = detect_columns(raw_df)
    if st.session_state["txn_col"] not in cols:
        st.session_state["txn_col"] = detected_txn
    if st.session_state["item_col"] not in cols or st.session_state["item_col"] == st.session_state["txn_col"]:
        st.session_state["item_col"] = (
            detected_item if detected_item != st.session_state["txn_col"] else cols[1]
        )
    return st.session_state["txn_col"], st.session_state["item_col"]


def _build_markdown_report(
    dataset_label: str,
    clean_stats: dict[str, Any],
    eda_summary: dict[str, Any],
    freq_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    insights: dict[str, Any],
) -> str:
    multi_combos = int((freq_df["itemset_length"] >= 2).sum()) if not freq_df.empty else 0
    lines = [
        "# Market Basket Analysis Report",
        "",
        f"**Dataset:** `{dataset_label}`  ",
        f"**Minimum Support:** `{st.session_state['min_support']:.3f}` | "
        f"**Minimum Confidence:** `{st.session_state['min_confidence']:.2f}` | "
        f"**Minimum Lift:** `{st.session_state['min_lift']:.2f}`",
        "",
        "## 1. Executive Summary",
        f"- **Total Transactions:** {eda_summary['total_transactions']:,}",
        f"- **Unique Products:** {eda_summary['total_products']:,}",
        f"- **Average Items per Transaction:** {eda_summary['avg_items_per_transaction']:.2f}",
        f"- **Frequent Product Combinations (2+ items):** {multi_combos:,}",
        f"- **Association Rules Generated:** {len(rules_df):,}",
        f"- **Rows Removed During Cleaning:** {clean_stats['total_removed']:,} "
        f"(Missing: {clean_stats['missing_rows_removed']}, "
        f"Invalid: {clean_stats['invalid_rows_removed']}, "
        f"Duplicates: {clean_stats['duplicate_rows_removed']})",
        "",
        "## 2. Top 10 Most Purchased Products",
    ]
    for idx, row in eda_summary["top_products"].iterrows():
        lines.append(
            f"{idx + 1}. **{row['Product']}** — {row['Count']} transactions ({row['Support (%)']}%)"
        )

    lines.append("")
    lines.append("## 3. Top 10 Association Rules by Lift")
    if rules_df.empty:
        lines.append("No rules matched the current thresholds.")
    else:
        for idx, row in rules_df.head(10).iterrows():
            lines.append(
                f"{idx + 1}. **{row['rule']}** — Support: {row['support']*100:.2f}%, "
                f"Confidence: {row['confidence']*100:.1f}%, Lift: {row['lift']:.2f}x"
            )

    lines.append("")
    lines.append("## 4. Actionable Business Insights")
    for title, key in [
        ("Cross-Selling Opportunities", "cross_sell"),
        ("Product Bundling Strategies", "bundling"),
        ("Product Placement Recommendations", "placement"),
        ("Promotional Offers", "promotions"),
    ]:
        lines.append(f"### {title}")
        items = insights.get(key, [])
        if not items:
            lines.append("- No qualifying rules at current threshold.")
        else:
            for item in items:
                lines.append(f"- **{item['rule']}**: {item['summary']}")
        lines.append("")

    return "\n".join(lines)


def _render_home_dashboard(
    dataset_label: str,
    eda_summary: dict[str, Any],
    freq_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    insights: dict[str, Any],
) -> None:
    st.title("Market Basket Analysis Dashboard")
    st.markdown(
        '<p class="section-caption">Customer Purchase Pattern Analysis & Product Recommendation System</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Active Dataset: **{dataset_label}** | "
        f"Min Support: `{st.session_state['min_support']:.3f}` | "
        f"Min Confidence: `{st.session_state['min_confidence']:.2f}` | "
        f"Min Lift: `{st.session_state['min_lift']:.2f}`"
    )

    multi_itemsets = int((freq_df["itemset_length"] >= 2).sum()) if not freq_df.empty else 0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Transactions", f"{eda_summary['total_transactions']:,}")
    col2.metric("Total Products", f"{eda_summary['total_products']:,}")
    col3.metric("Product Combinations", f"{multi_itemsets:,}", help="Frequent itemsets with 2 or more products")
    col4.metric("Association Rules", f"{len(rules_df):,}")
    col5.metric("Avg Items / Basket", f"{eda_summary['avg_items_per_transaction']:.2f}")

    st.divider()

    left_col, right_col = st.columns([1.05, 1.0])
    with left_col:
        st.plotly_chart(
            plot_top_products_bar(eda_summary["product_frequency"], top_n=10),
            width="stretch",
        )
    with right_col:
        if rules_df.empty:
            st.info(
                "No association rules meet the current Support/Confidence/Lift thresholds. "
                "Lower the thresholds in the sidebar or on the Frequent Itemsets / Association Rules pages."
            )
        else:
            st.plotly_chart(
                plot_support_vs_confidence_scatter(rules_df),
                width="stretch",
            )

    st.subheader("Top Discovered Association Rules")
    if rules_df.empty:
        st.warning("Adjust thresholds to discover association rules.")
    else:
        preview_rules = rules_df[["antecedents_str", "consequents_str", "support", "confidence", "lift"]].head(8).copy()
        preview_rules.columns = ["Antecedent (If Bought)", "Consequent (Also Bought)", "Support", "Confidence", "Lift"]
        st.dataframe(preview_rules, width="stretch", hide_index=True)

    st.subheader("Key Business Takeaways")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.markdown("**Top Cross-Sell**")
        if insights["cross_sell"]:
            top_cs = insights["cross_sell"][0]
            st.markdown(f'<div class="insight-card"><b>{top_cs["rule"]}</b><br>{top_cs["summary"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No cross-sell rules available at current threshold.")
    with b2:
        st.markdown("**Top Bundle Opportunity**")
        if insights["bundling"]:
            top_bd = insights["bundling"][0]
            st.markdown(f'<div class="insight-card-green"><b>{top_bd["rule"]}</b><br>{top_bd["summary"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No bundling rules available at current threshold.")
    with b3:
        st.markdown("**Top Placement Pair**")
        if insights["placement"]:
            top_pl = insights["placement"][0]
            st.markdown(f'<div class="insight-card-amber"><b>{top_pl["rule"]}</b><br>{top_pl["summary"]}</div>', unsafe_allow_html=True)
        else:
            st.info("No placement rules available at current threshold.")


def _render_dataset_explorer(
    raw_df: pd.DataFrame,
    dataset_label: str,
    cleaned_df: pd.DataFrame,
    clean_stats: dict[str, Any],
    basket_matrix: pd.DataFrame,
    txn_col: str,
    item_col: str,
) -> None:
    st.title("Dataset Explorer & Preprocessing")
    st.markdown(
        '<p class="section-caption">Upload CSV transaction logs, inspect data quality, and verify the one-hot encoded basket matrix.</p>',
        unsafe_allow_html=True,
    )

    up_col, action_col = st.columns([2.2, 1.0])
    with up_col:
        uploaded_file = st.file_uploader(
            "Upload Transaction CSV (columns such as Transaction ID & Product)",
            type=["csv"],
            help="Each row should contain a Transaction/Order ID and a Product/Item name.",
        )
        if uploaded_file is not None:
            file_bytes = uploaded_file.getvalue()
            if st.session_state["uploaded_bytes"] != file_bytes:
                st.session_state["uploaded_bytes"] = file_bytes
                st.session_state["uploaded_name"] = uploaded_file.name
                st.session_state["txn_col"] = None
                st.session_state["item_col"] = None
                st.rerun()

    with action_col:
        st.write("")
        st.write("")
        if st.session_state["uploaded_bytes"] is not None:
            if st.button("Reset to Default Groceries Dataset", width="stretch"):
                st.session_state["uploaded_bytes"] = None
                st.session_state["uploaded_name"] = None
                st.session_state["txn_col"] = None
                st.session_state["item_col"] = None
                st.rerun()
        sample_csv = (
            "Transaction ID,Product\n"
            "T001,Bread\nT001,Butter\nT001,Milk\n"
            "T002,Bread\nT002,Butter\n"
            "T003,Milk\nT003,Eggs\n"
        )
        st.download_button(
            "Download CSV Template",
            data=sample_csv,
            file_name="sample_transactions_template.csv",
            mime="text/csv",
            width="stretch",
        )

    st.info(f"Currently inspecting: **{dataset_label}** ({len(raw_df):,} raw rows, {len(raw_df.columns)} columns)")

    cols = [str(c) for c in raw_df.columns]
    map_c1, map_c2 = st.columns(2)
    with map_c1:
        selected_txn = st.selectbox(
            "Transaction ID Column",
            options=cols,
            index=cols.index(txn_col),
        )
    with map_c2:
        other_cols = [c for c in cols if c != selected_txn]
        default_item_idx = other_cols.index(item_col) if item_col in other_cols else 0
        selected_item = st.selectbox(
            "Product / Item Column",
            options=other_cols,
            index=default_item_idx,
        )

    if selected_txn != txn_col or selected_item != item_col:
        st.session_state["txn_col"] = selected_txn
        st.session_state["item_col"] = selected_item
        st.rerun()

    st.subheader("Preprocessing Audit Summary")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Raw Records", f"{clean_stats['raw_rows']:,}")
    s2.metric("Missing Removed", f"{clean_stats['missing_rows_removed']:,}")
    s3.metric("Invalid Removed", f"{clean_stats['invalid_rows_removed']:,}")
    s4.metric("Duplicates Removed", f"{clean_stats['duplicate_rows_removed']:,}")
    s5.metric("Clean Records", f"{clean_stats['clean_rows']:,}")

    tab_raw, tab_clean, tab_missing, tab_matrix, tab_lookup = st.tabs(
        ["Raw Data Preview", "Cleaned Data Preview", "Missing Values", "Basket Matrix (1/0)", "Transaction Lookup"]
    )

    with tab_raw:
        st.dataframe(raw_df.head(200), width="stretch")

    with tab_clean:
        st.dataframe(cleaned_df.head(200), width="stretch")

    with tab_missing:
        miss_df = pd.DataFrame(
            [
                {
                    "Column": col,
                    "Missing Count": cnt,
                    "Missing (%)": round(cnt / max(clean_stats["raw_rows"], 1) * 100, 2),
                    "Data Type": str(raw_df[col].dtype),
                }
                for col, cnt in clean_stats["missing_by_col"].items()
            ]
        )
        st.dataframe(miss_df, width="stretch", hide_index=True)

    with tab_matrix:
        st.caption(
            f"Basket Matrix Dimensions: **{basket_matrix.shape[0]:,} transactions x {basket_matrix.shape[1]:,} unique products** "
            "(1 = purchased, 0 = not purchased)"
        )
        col_limit = min(20, basket_matrix.shape[1])
        row_limit = min(50, basket_matrix.shape[0])
        selected_cols = st.multiselect(
            "Filter product columns to preview",
            options=list(basket_matrix.columns),
            default=list(basket_matrix.columns[:col_limit]),
        )
        if selected_cols:
            st.dataframe(basket_matrix[selected_cols].head(row_limit), width="stretch")

    with tab_lookup:
        txn_ids = list(basket_matrix.index.astype(str))
        chosen_txn = st.selectbox("Select Transaction ID", options=txn_ids[:500])
        if chosen_txn:
            items = cleaned_df[cleaned_df[txn_col].astype(str) == chosen_txn][item_col].tolist()
            st.write(f"**Transaction `{chosen_txn}` contains {len(items)} unique products:**")
            st.write(", ".join(items))


def _render_product_analysis(eda_summary: dict[str, Any]) -> None:
    st.title("Exploratory Product Analysis")
    st.markdown(
        '<p class="section-caption">Understand item popularity, long-tail products, and customer basket size distribution.</p>',
        unsafe_allow_html=True,
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Purchases (Line Items)", f"{eda_summary['total_items_purchased']:,}")
    m2.metric("Average Basket Size", f"{eda_summary['avg_items_per_transaction']:.2f}")
    m3.metric("Median Basket Size", f"{eda_summary['median_items_per_transaction']:.1f}")
    m4.metric("Max Items in One Basket", f"{eda_summary['max_items_per_transaction']}")

    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(
            plot_top_products_bar(eda_summary["product_frequency"], top_n=10),
            width="stretch",
        )
    with c2:
        st.plotly_chart(
            plot_least_products_bar(eda_summary["product_frequency"], bottom_n=10),
            width="stretch",
        )

    c3, c4 = st.columns(2)
    with c3:
        st.plotly_chart(
            plot_product_distribution_pie(eda_summary["product_frequency"], top_n=10),
            width="stretch",
        )
    with c4:
        st.plotly_chart(
            plot_transaction_frequency_bar(eda_summary["transaction_size_distribution"]),
            width="stretch",
        )

    st.plotly_chart(
        plot_transaction_size_histogram(eda_summary["items_per_transaction"]),
        width="stretch",
    )

    st.subheader("Full Product Purchase Frequency Table")
    search_term = st.text_input("Search product name", placeholder="e.g. Milk, Bread, Coffee...")
    freq_table = eda_summary["product_frequency"]
    if search_term.strip():
        freq_table = freq_table[
            freq_table["Product"].str.contains(search_term.strip(), case=False, na=False)
        ]
    st.dataframe(freq_table, width="stretch", hide_index=True)


def _render_frequent_itemsets(basket_matrix: pd.DataFrame) -> pd.DataFrame:
    st.title("Frequent Itemsets (Apriori Algorithm)")
    st.markdown(
        '<p class="section-caption">Configure the minimum support threshold to mine frequent single products and multi-item combinations.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("How Support Works", expanded=False):
        st.latex(r"\text{Support}(A) = \frac{\text{Transactions containing } A}{\text{Total Transactions}}")
        st.write(
            "Lower support uncovers rarer combinations; higher support focuses only on the most common baskets."
        )

    ctrl1, ctrl2, ctrl3 = st.columns([1.4, 1.0, 1.0])
    with ctrl1:
        st.session_state["min_support"] = st.slider(
            "Minimum Support Threshold",
            min_value=0.005,
            max_value=0.30,
            value=float(st.session_state["min_support"]),
            step=0.005,
            format="%.3f",
        )
    with ctrl2:
        max_len_choice = st.selectbox(
            "Max Itemset Length",
            options=[2, 3, 4, "No Limit"],
            index=[2, 3, 4, "No Limit"].index(
                st.session_state["max_itemset_len"]
                if st.session_state["max_itemset_len"] in (2, 3, 4)
                else "No Limit"
            ),
        )
    with ctrl3:
        length_filter = st.selectbox(
            "Filter by Product Count",
            options=["All Itemsets", "1-Item Only", "2-Item Combinations", "3+ Item Combinations"],
        )

    st.session_state["max_itemset_len"] = None if max_len_choice == "No Limit" else int(max_len_choice)

    with st.spinner("Running Apriori algorithm..."):
        freq_df = _cached_apriori(
            basket_matrix,
            st.session_state["min_support"],
            st.session_state["max_itemset_len"],
        )

    if freq_df.empty:
        st.warning("No frequent itemsets found at this support threshold. Try lowering Minimum Support.")
        return freq_df

    filtered_df = freq_df.copy()
    if length_filter == "1-Item Only":
        filtered_df = filtered_df[filtered_df["itemset_length"] == 1]
    elif length_filter == "2-Item Combinations":
        filtered_df = filtered_df[filtered_df["itemset_length"] == 2]
    elif length_filter == "3+ Item Combinations":
        filtered_df = filtered_df[filtered_df["itemset_length"] >= 3]

    product_query = st.text_input("Filter itemsets containing product", placeholder="Type a product name...")
    if product_query.strip():
        filtered_df = filtered_df[
            filtered_df["itemsets_str"].str.contains(product_query.strip(), case=False, na=False)
        ]

    k1, k2, k3 = st.columns(3)
    k1.metric("Matching Frequent Itemsets", f"{len(filtered_df):,}")
    k2.metric(
        "Multi-Item Combinations (2+)",
        f"{int((filtered_df['itemset_length'] >= 2).sum()):,}",
    )
    k3.metric(
        "Highest Support",
        f"{filtered_df['support_pct'].max():.2f}%" if not filtered_df.empty else "0.00%",
    )

    if filtered_df.empty:
        st.info("No itemsets match the active length/search filters.")
        return freq_df

    display_df = filtered_df[["itemsets_str", "itemset_length", "support", "support_pct"]].copy()
    display_df.columns = ["Frequent Itemset", "Product Count", "Support", "Support (%)"]

    tab_chart, tab_table = st.tabs(["Comparison Chart", "Frequent Itemsets Table"])
    with tab_chart:
        top_n = st.slider("Number of itemsets to plot", 5, min(30, max(5, len(filtered_df))), min(15, len(filtered_df)))
        st.plotly_chart(plot_frequent_itemsets_bar(filtered_df, top_n=top_n), width="stretch")
    with tab_table:
        st.dataframe(display_df, width="stretch", hide_index=True)

    return freq_df


def _render_association_rules(freq_df: pd.DataFrame) -> pd.DataFrame:
    st.title("Association Rule Mining")
    st.markdown(
        '<p class="section-caption">Discover directional product rules (Antecedent -> Consequent) measured by Support, Confidence, and Lift.</p>',
        unsafe_allow_html=True,
    )

    with st.expander("Metric Formulas & Lift Interpretation", expanded=False):
        st.latex(r"\text{Confidence}(A \rightarrow B) = \frac{\text{Support}(A \cup B)}{\text{Support}(A)}")
        st.latex(r"\text{Lift}(A \rightarrow B) = \frac{\text{Confidence}(A \rightarrow B)}{\text{Support}(B)}")
        st.markdown(
            "- **Lift > 1**: Positive association (products are bought together more often than chance).\n"
            "- **Lift = 1**: Independent (no meaningful relationship).\n"
            "- **Lift < 1**: Negative association (substitute products)."
        )

    c1, c2, c3 = st.columns([1.2, 1.2, 1.2])
    with c1:
        st.session_state["min_confidence"] = st.slider(
            "Minimum Confidence",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state["min_confidence"]),
            step=0.05,
        )
    with c2:
        st.session_state["min_lift"] = st.slider(
            "Minimum Lift Filter",
            min_value=0.50,
            max_value=4.00,
            value=float(st.session_state["min_lift"]),
            step=0.05,
        )
    with c3:
        rule_search = st.text_input("Filter rules by product", placeholder="e.g. Whole Milk, Bread...")

    rules_df = _cached_rules(freq_df, min_confidence=st.session_state["min_confidence"], min_lift=st.session_state["min_lift"])

    if rules_df.empty:
        st.warning(
            "No association rules matched the current Support, Confidence, and Lift filters. "
            "Try lowering Minimum Support (Frequent Itemsets page) or Minimum Confidence / Lift."
        )
        return rules_df

    filtered_rules = rules_df.copy()
    if rule_search.strip():
        q = rule_search.strip()
        filtered_rules = filtered_rules[
            filtered_rules["antecedents_str"].str.contains(q, case=False, na=False)
            | filtered_rules["consequents_str"].str.contains(q, case=False, na=False)
        ]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rules Found", f"{len(filtered_rules):,}")
    m2.metric("Highest Lift", f"{filtered_rules['lift'].max():.2f}x" if not filtered_rules.empty else "0.00x")
    m3.metric(
        "Highest Confidence",
        f"{filtered_rules['confidence'].max() * 100:.1f}%" if not filtered_rules.empty else "0.0%",
    )
    m4.metric(
        "Positive Rules (Lift > 1)",
        f"{int((filtered_rules['lift'] > 1.0).sum()):,}",
    )

    if filtered_rules.empty:
        st.info("No rules matched your search filter.")
        return rules_df

    table_tab, scatter_tab, lift_tab, network_tab, heatmap_tab = st.tabs(
        [
            "Rules Table",
            "Support vs Confidence",
            "Lift Comparison",
            "Network Graph",
            "Lift Heatmap (Seaborn)",
        ]
    )

    with table_tab:
        display_rules = filtered_rules[
            [
                "antecedents_str",
                "consequents_str",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        ].copy()
        display_rules.columns = [
            "Antecedents (If)",
            "Consequents (Then)",
            "Support",
            "Confidence",
            "Lift",
            "Leverage",
            "Conviction",
        ]
        st.dataframe(display_rules, width="stretch", hide_index=True)

    with scatter_tab:
        st.plotly_chart(plot_support_vs_confidence_scatter(filtered_rules), width="stretch")

    with lift_tab:
        top_n_lift = st.slider("Rules to compare", 5, min(30, max(5, len(filtered_rules))), min(15, len(filtered_rules)))
        st.plotly_chart(plot_lift_comparison_bar(filtered_rules, top_n=top_n_lift), width="stretch")

    with network_tab:
        max_edges = st.slider(
            "Maximum rule connections in graph",
            min_value=5,
            max_value=min(50, max(5, len(filtered_rules))),
            value=min(25, len(filtered_rules)),
        )
        st.plotly_chart(plot_product_network_graph(filtered_rules, max_rules=max_edges), width="stretch")

    with heatmap_tab:
        fig = plot_seaborn_lift_heatmap(filtered_rules, top_n=12)
        st.pyplot(fig, width="stretch")

    return rules_df


def _render_business_insights(
    eda_summary: dict[str, Any],
    rules_df: pd.DataFrame,
    insights: dict[str, Any],
) -> None:
    st.title("Business Insights & Recommendation Engine")
    st.markdown(
        '<p class="section-caption">Actionable merchandising, cross-selling, bundling, and inventory strategies dynamically derived from your dataset.</p>',
        unsafe_allow_html=True,
    )

    st.subheader("Interactive Product Recommendation Simulator")
    all_products = eda_summary["product_frequency"]["Product"].tolist()
    selected_basket = st.multiselect(
        "Simulate a customer's current shopping basket:",
        options=all_products,
        default=all_products[:1] if all_products else [],
    )
    if selected_basket:
        recs = recommend_for_basket(rules_df, selected_basket, top_n=8)
        if recs.empty:
            st.info(
                "No direct recommendations found for this combination at the current thresholds. "
                "Try adding another item or lowering Minimum Support / Confidence."
            )
        else:
            st.success(f"Top {len(recs)} recommended add-on products for `{', '.join(selected_basket)}`:")
            st.dataframe(recs, width="stretch", hide_index=True)

    st.divider()
    st.subheader("Data-Driven Strategy Recommendations")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 1. Cross-Selling Opportunities")
        if not insights["cross_sell"]:
            st.info("No qualifying rules found at current thresholds.")
        else:
            for item in insights["cross_sell"]:
                st.markdown(
                    f'<div class="insight-card"><b>{item["rule"]}</b><br>{item["summary"]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### 3. Store & Category Placement")
        if not insights["placement"]:
            st.info("No qualifying placement pairs found.")
        else:
            for item in insights["placement"]:
                st.markdown(
                    f'<div class="insight-card-amber"><b>{item["rule"]}</b><br>{item["summary"]}</div>',
                    unsafe_allow_html=True,
                )

    with col_b:
        st.markdown("### 2. Product Bundling Strategies")
        if not insights["bundling"]:
            st.info("No qualifying bundle combinations found.")
        else:
            for item in insights["bundling"]:
                st.markdown(
                    f'<div class="insight-card-green"><b>{item["rule"]}</b><br>{item["summary"]}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("### 4. Promotional & Discount Campaigns")
        if not insights["promotions"]:
            st.info("No qualifying promotional rules found.")
        else:
            for item in insights["promotions"]:
                st.markdown(
                    f'<div class="insight-card"><b>{item["rule"]}</b><br>{item["summary"]}</div>',
                    unsafe_allow_html=True,
                )

    st.markdown("### 5. Inventory & Safety Stock Planning")
    if insights["inventory"]:
        inv_df = pd.DataFrame(insights["inventory"])[["product", "count", "support_pct", "summary"]]
        inv_df.columns = ["Anchor Product", "Order Count", "Basket Penetration (%)", "Inventory Action"]
        st.dataframe(inv_df, width="stretch", hide_index=True)


def _render_reports_page(
    dataset_label: str,
    cleaned_df: pd.DataFrame,
    clean_stats: dict[str, Any],
    basket_matrix: pd.DataFrame,
    eda_summary: dict[str, Any],
    freq_df: pd.DataFrame,
    rules_df: pd.DataFrame,
    insights: dict[str, Any],
) -> None:
    st.title("Export & Download Reports")
    st.markdown(
        '<p class="section-caption">Download mined frequent itemsets, association rules, cleaned datasets, and an executive summary report.</p>',
        unsafe_allow_html=True,
    )

    freq_export = (
        freq_df[["itemsets_str", "itemset_length", "support", "support_pct"]].rename(
            columns={
                "itemsets_str": "Frequent Itemset",
                "itemset_length": "Item Count",
                "support": "Support",
                "support_pct": "Support (%)",
            }
        )
        if not freq_df.empty
        else pd.DataFrame(columns=["Frequent Itemset", "Item Count", "Support", "Support (%)"])
    )

    rules_export = (
        rules_df[
            [
                "antecedents_str",
                "consequents_str",
                "support",
                "confidence",
                "lift",
                "leverage",
                "conviction",
            ]
        ].rename(
            columns={
                "antecedents_str": "Antecedents",
                "consequents_str": "Consequents",
                "support": "Support",
                "confidence": "Confidence",
                "lift": "Lift",
                "leverage": "Leverage",
                "conviction": "Conviction",
            }
        )
        if not rules_df.empty
        else pd.DataFrame(
            columns=["Antecedents", "Consequents", "Support", "Confidence", "Lift", "Leverage", "Conviction"]
        )
    )

    summary_md = _build_markdown_report(
        dataset_label, clean_stats, eda_summary, freq_df, rules_df, insights
    )

    d1, d2 = st.columns(2)
    with d1:
        st.subheader("Analytical Outputs")
        st.download_button(
            "Download Frequent Itemsets (CSV)",
            data=freq_export.to_csv(index=False).encode("utf-8"),
            file_name="frequent_itemsets.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "Download Association Rules (CSV)",
            data=rules_export.to_csv(index=False).encode("utf-8"),
            file_name="association_rules.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "Download Executive Summary Report (Markdown)",
            data=summary_md.encode("utf-8"),
            file_name="market_basket_executive_report.md",
            mime="text/markdown",
            width="stretch",
        )

    with d2:
        st.subheader("Processed Datasets")
        st.download_button(
            "Download Cleaned Transactions (CSV)",
            data=cleaned_df.to_csv(index=False).encode("utf-8"),
            file_name="cleaned_transactions.csv",
            mime="text/csv",
            width="stretch",
        )
        st.download_button(
            "Download One-Hot Basket Matrix (CSV)",
            data=basket_matrix.to_csv().encode("utf-8"),
            file_name="basket_matrix.csv",
            mime="text/csv",
            width="stretch",
        )

    st.divider()
    st.subheader("Executive Summary Preview")
    st.markdown(summary_md)


def main() -> None:
    st.set_page_config(
        page_title="Market Basket Analysis Dashboard",
        page_icon="🛒",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_custom_styles()
    _init_session_state()

    with st.sidebar:
        st.markdown("## Market Basket Analytics")
        st.caption("Apriori & Association Rule Mining")

        page = st.radio("Navigation", options=PAGES, key="active_page")
        st.divider()

        st.markdown("### Mining Parameters")
        st.session_state["min_support"] = st.slider(
            "Global Min Support",
            min_value=0.005,
            max_value=0.30,
            value=float(st.session_state["min_support"]),
            step=0.005,
            format="%.3f",
        )
        st.session_state["min_confidence"] = st.slider(
            "Global Min Confidence",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state["min_confidence"]),
            step=0.05,
        )
        st.session_state["min_lift"] = st.slider(
            "Global Min Lift",
            min_value=0.50,
            max_value=4.00,
            value=float(st.session_state["min_lift"]),
            step=0.05,
        )

    try:
        raw_df, dataset_label = _load_active_raw_df()
        txn_col, item_col = _sync_column_selection(raw_df)
        cleaned_df, clean_stats, basket_matrix, eda_summary = _cached_preprocess(
            raw_df, txn_col, item_col
        )
    except Exception as exc:
        st.error(f"Failed to load or preprocess dataset: {exc}")
        st.stop()

    freq_df = _cached_apriori(
        basket_matrix,
        st.session_state["min_support"],
        st.session_state["max_itemset_len"],
    )
    rules_df = _cached_rules(
        freq_df,
        min_confidence=st.session_state["min_confidence"],
        min_lift=st.session_state["min_lift"],
    )
    insights = generate_business_insights(rules_df, eda_summary, freq_df)

    if page == "Home Dashboard":
        _render_home_dashboard(dataset_label, eda_summary, freq_df, rules_df, insights)
    elif page == "Dataset Explorer":
        _render_dataset_explorer(
            raw_df, dataset_label, cleaned_df, clean_stats, basket_matrix, txn_col, item_col
        )
    elif page == "Product Analysis":
        _render_product_analysis(eda_summary)
    elif page == "Frequent Itemsets":
        _render_frequent_itemsets(basket_matrix)
    elif page == "Association Rules":
        _render_association_rules(freq_df)
    elif page == "Business Insights":
        _render_business_insights(eda_summary, rules_df, insights)
    elif page == "Download Reports":
        _render_reports_page(
            dataset_label, cleaned_df, clean_stats, basket_matrix, eda_summary, freq_df, rules_df, insights
        )


if __name__ == "__main__":
    main()
