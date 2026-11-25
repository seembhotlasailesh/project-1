"""Interactive Plotly and statistical Seaborn visualizations for Market Basket Analysis."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

PALETTE = [
    "#2563EB",
    "#0EA5E9",
    "#10B981",
    "#F59E0B",
    "#8B5CF6",
    "#EC4899",
    "#14B8A6",
    "#F97316",
    "#6366F1",
    "#84CC16",
]


def _apply_base_layout(fig: go.Figure, title: str, height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        height=height,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12),
    )
    return fig


def plot_top_products_bar(product_freq_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Horizontal bar chart of the top N most frequently purchased products."""
    subset = product_freq_df.head(top_n).sort_values("Count", ascending=True)
    fig = px.bar(
        subset,
        x="Count",
        y="Product",
        orientation="h",
        text="Support (%)",
        color="Count",
        color_continuous_scale="Blues",
        hover_data={"Count": True, "Support (%)": True, "Share of Purchases (%)": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title_text="Number of Transactions", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title_text="")
    return _apply_base_layout(fig, f"Top {len(subset)} Most Purchased Products", height=430)


def plot_least_products_bar(product_freq_df: pd.DataFrame, bottom_n: int = 10) -> go.Figure:
    """Horizontal bar chart of the least frequently purchased products."""
    subset = (
        product_freq_df.sort_values(["Count", "Product"], ascending=[True, True])
        .head(bottom_n)
        .sort_values("Count", ascending=False)
    )
    fig = px.bar(
        subset,
        x="Count",
        y="Product",
        orientation="h",
        text="Support (%)",
        color="Count",
        color_continuous_scale="OrRd_r",
        hover_data={"Count": True, "Support (%)": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title_text="Number of Transactions", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title_text="")
    return _apply_base_layout(fig, f"Bottom {len(subset)} Least Purchased Products", height=430)


def plot_product_distribution_pie(product_freq_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    """Donut chart showing purchase share across top products and remaining items."""
    if product_freq_df.empty:
        return _apply_base_layout(go.Figure(), "Product Purchase Distribution")

    top_df = product_freq_df.head(top_n)[["Product", "Count"]].copy()
    other_count = int(product_freq_df.iloc[top_n:]["Count"].sum()) if len(product_freq_df) > top_n else 0
    if other_count > 0:
        top_df = pd.concat(
            [top_df, pd.DataFrame([{"Product": "Other Products", "Count": other_count}])],
            ignore_index=True,
        )

    fig = px.pie(
        top_df,
        names="Product",
        values="Count",
        hole=0.42,
        color_discrete_sequence=PALETTE,
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    return _apply_base_layout(fig, f"Purchase Share Distribution (Top {min(top_n, len(product_freq_df))} + Others)", height=440)


def plot_transaction_size_histogram(items_per_txn: pd.Series) -> go.Figure:
    """Histogram of the number of items purchased per transaction."""
    df_hist = pd.DataFrame({"Items in Basket": items_per_txn.values})
    fig = px.histogram(
        df_hist,
        x="Items in Basket",
        nbins=max(int(df_hist["Items in Basket"].max()) if not df_hist.empty else 5, 5),
        color_discrete_sequence=["#0EA5E9"],
    )
    fig.update_traces(marker_line_color="rgba(255,255,255,0.5)", marker_line_width=1)
    fig.update_xaxes(title_text="Number of Unique Items in Transaction", dtick=1)
    fig.update_yaxes(title_text="Transaction Count", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    return _apply_base_layout(fig, "Items per Transaction (Histogram)", height=400)


def plot_transaction_frequency_bar(txn_size_dist: pd.DataFrame) -> go.Figure:
    """Bar chart showing transaction count for each basket size."""
    fig = px.bar(
        txn_size_dist,
        x="Items in Basket",
        y="Transaction Count",
        text="Percentage (%)",
        color="Transaction Count",
        color_continuous_scale="Teal",
        hover_data={"Percentage (%)": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    fig.update_xaxes(title_text="Basket Size (Items)", dtick=1)
    fig.update_yaxes(title_text="Number of Transactions", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    return _apply_base_layout(fig, "Transaction Frequency by Basket Size", height=400)


def plot_frequent_itemsets_bar(freq_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Bar chart comparing top frequent itemsets colored by itemset length."""
    subset = freq_df.head(top_n).copy()
    subset["Length Label"] = subset["itemset_length"].astype(str) + "-item"
    subset = subset.sort_values("support", ascending=True)

    fig = px.bar(
        subset,
        x="support_pct",
        y="itemsets_str",
        color="Length Label",
        orientation="h",
        text="support_pct",
        color_discrete_sequence=PALETTE,
        hover_data={"support": True, "support_pct": True, "itemset_length": True},
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_xaxes(title_text="Support (%)", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title_text="")
    return _apply_base_layout(fig, f"Top {len(subset)} Frequent Itemsets Comparison", height=max(420, len(subset) * 28 + 100))


def plot_support_vs_confidence_scatter(rules_df: pd.DataFrame) -> go.Figure:
    """Interactive scatter plot of Support vs Confidence, sized and colored by Lift."""
    plot_df = rules_df.copy()
    plot_df["Support (%)"] = (plot_df["support"] * 100).round(2)
    plot_df["Confidence (%)"] = (plot_df["confidence"] * 100).round(2)

    fig = px.scatter(
        plot_df,
        x="Support (%)",
        y="Confidence (%)",
        size="lift",
        color="lift",
        hover_name="rule",
        hover_data={
            "Support (%)": True,
            "Confidence (%)": True,
            "lift": ":.3f",
            "leverage": ":.4f",
        },
        color_continuous_scale="Viridis",
        size_max=22,
    )
    fig.update_xaxes(title_text="Support (%)", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title_text="Confidence (%)", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    return _apply_base_layout(fig, "Association Rules: Support vs Confidence (Sized & Colored by Lift)", height=460)


def plot_lift_comparison_bar(rules_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    """Horizontal bar chart comparing Lift values for top association rules."""
    subset = rules_df.sort_values("lift", ascending=False).head(top_n).sort_values("lift", ascending=True)
    fig = px.bar(
        subset,
        x="lift",
        y="rule",
        orientation="h",
        color="confidence",
        text="lift",
        color_continuous_scale="Emrld",
        hover_data={"support": ":.4f", "confidence": ":.4f", "lift": ":.3f"},
    )
    fig.update_traces(texttemplate="%{text:.2f}x", textposition="outside")
    fig.add_vline(x=1.0, line_dash="dash", line_color="#EF4444", annotation_text="Lift = 1.0 (Independent)")
    fig.update_xaxes(title_text="Lift", showgrid=True, gridcolor="rgba(128,128,128,0.18)")
    fig.update_yaxes(title_text="")
    return _apply_base_layout(fig, f"Top {len(subset)} Association Rules by Lift", height=max(420, len(subset) * 30 + 100))


def plot_product_network_graph(rules_df: pd.DataFrame, max_rules: int = 25) -> go.Figure:
    """Interactive Plotly network graph illustrating directed product associations."""
    subset = rules_df.sort_values(["lift", "confidence"], ascending=[False, False]).head(max_rules)
    graph = nx.DiGraph()

    for _, row in subset.iterrows():
        ant = row["antecedents_str"]
        con = row["consequents_str"]
        graph.add_edge(
            ant,
            con,
            lift=float(row["lift"]),
            confidence=float(row["confidence"]),
            support=float(row["support"]),
        )

    if graph.number_of_nodes() == 0:
        return _apply_base_layout(go.Figure(), "Product Relationship Network Graph")

    pos = nx.spring_layout(graph, k=1.4 / np.sqrt(max(graph.number_of_nodes(), 1)), seed=42)

    edge_traces = []
    annotation_arrows = []
    for src, dst, data in graph.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        lift_val = data["lift"]
        width = min(max(lift_val * 1.4, 1.2), 6.0)
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=width, color="rgba(148, 163, 184, 0.65)"),
                hoverinfo="text",
                text=(
                    f"{src} -> {dst}<br>"
                    f"Lift: {lift_val:.2f} | Confidence: {data['confidence']*100:.1f}% | "
                    f"Support: {data['support']*100:.1f}%"
                ),
                showlegend=False,
            )
        )
        annotation_arrows.append(
            dict(
                ax=x0,
                ay=y0,
                x=x1,
                y=y1,
                xref="x",
                yref="y",
                axref="x",
                ayref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1.1,
                arrowwidth=1.2,
                arrowcolor="rgba(59, 130, 246, 0.65)",
                standoff=14,
                startstandoff=10,
            )
        )

    node_x, node_y, node_text, node_hover, node_degree = [], [], [], [], []
    for node in graph.nodes():
        x, y = pos[node]
        deg = graph.degree(node)
        in_edges = [f"{u} -> {node} (Lift {d['lift']:.2f})" for u, _, d in graph.in_edges(node, data=True)]
        out_edges = [f"{node} -> {v} (Lift {d['lift']:.2f})" for _, v, d in graph.out_edges(node, data=True)]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)
        node_degree.append(deg)
        details = "<br>".join((out_edges + in_edges)[:6])
        node_hover.append(f"<b>{node}</b><br>Connections: {deg}<br>{details}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        hovertext=node_hover,
        marker=dict(
            size=[14 + d * 4 for d in node_degree],
            color=node_degree,
            colorscale="Blues",
            showscale=True,
            colorbar=dict(title="Links"),
            line=dict(width=1.5, color="#FFFFFF"),
        ),
        showlegend=False,
    )

    fig = go.Figure(data=[*edge_traces, node_trace])
    fig.update_layout(
        annotations=annotation_arrows,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    )
    return _apply_base_layout(fig, f"Product Relationship Network Graph (Top {len(subset)} Rules)", height=540)


def plot_seaborn_lift_heatmap(rules_df: pd.DataFrame, top_n: int = 12) -> plt.Figure:
    """Seaborn heatmap of Lift between top antecedents and consequents."""
    subset = rules_df.head(max(top_n * 2, 20)).copy()
    pivot = subset.pivot_table(
        index="antecedents_str",
        columns="consequents_str",
        values="lift",
        aggfunc="max",
    )
    pivot = pivot.iloc[:top_n, :top_n]

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 0.9), max(4.5, len(pivot.index) * 0.55)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        cbar_kws={"label": "Lift"},
        ax=ax,
    )
    ax.set_title("Antecedent vs Consequent Lift Heatmap", fontsize=12, pad=12)
    ax.set_xlabel("Consequent (Recommended Item)")
    ax.set_ylabel("Antecedent (Basket Item)")
    plt.xticks(rotation=35, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()
    return fig
