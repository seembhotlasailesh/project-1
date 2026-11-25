"""Data loading, cleaning, and basket matrix transformation utilities."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd

_TXN_CANDIDATES = (
    "transaction id",
    "transaction_id",
    "transaction",
    "txn_id",
    "invoiceno",
    "invoice_no",
    "order_id",
    "order id",
    "member_number",
    "basket_id",
    "receipt_id",
)

_ITEM_CANDIDATES = (
    "product",
    "product_name",
    "item",
    "itemdescription",
    "item_description",
    "description",
    "sku_name",
    "item_name",
)

_INVALID_TOKENS = {"", "nan", "none", "null", "na", "n/a", "unknown", "-"}


def load_csv_data(source: Any) -> pd.DataFrame:
    """Load a CSV file path, file-like buffer, or uploaded bytes into a DataFrame."""
    if isinstance(source, (bytes, bytearray)):
        raw_bytes = bytes(source)
        for encoding in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                return pd.read_csv(BytesIO(raw_bytes), encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError("Unable to decode the uploaded CSV file with UTF-8 or Latin-1.")

    if hasattr(source, "getvalue"):
        return load_csv_data(source.getvalue())

    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(source, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("Unable to decode the CSV file. Please save it as UTF-8.")


def detect_columns(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Infer likely Transaction ID and Product columns from DataFrame headers."""
    if df.empty or len(df.columns) < 2:
        cols = list(df.columns)
        return (cols[0] if len(cols) >= 1 else None, cols[1] if len(cols) >= 2 else None)

    normalized = {str(col).strip().lower(): str(col) for col in df.columns}
    txn_col = next((normalized[c] for c in _TXN_CANDIDATES if c in normalized), None)
    item_col = next((normalized[c] for c in _ITEM_CANDIDATES if c in normalized), None)

    if txn_col is None:
        txn_col = next(
            (str(c) for c in df.columns if any(k in str(c).lower() for k in ("trans", "invoice", "order", "basket", "member", "id"))),
            str(df.columns[0]),
        )

    if item_col is None or item_col == txn_col:
        item_col = next(
            (
                str(c)
                for c in df.columns
                if str(c) != txn_col and any(k in str(c).lower() for k in ("prod", "item", "desc", "name", "sku", "goods"))
            ),
            next((str(c) for c in df.columns if str(c) != txn_col), str(df.columns[0])),
        )

    return txn_col, item_col


def _is_valid_series(series: pd.Series) -> pd.Series:
    """Return boolean mask of non-null, non-placeholder string entries."""
    not_na = series.notna()
    cleaned = series.astype(str).str.strip()
    valid_token = ~cleaned.str.lower().isin(_INVALID_TOKENS)
    return not_na & valid_token


def clean_transaction_data(
    df: pd.DataFrame,
    txn_col: str,
    item_col: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean raw transaction records and return cleaned DataFrame plus audit metrics."""
    if txn_col not in df.columns or item_col not in df.columns:
        raise ValueError(f"Selected columns '{txn_col}' and '{item_col}' must exist in the dataset.")
    if txn_col == item_col:
        raise ValueError("Transaction ID column and Product column must be different.")

    raw_rows = int(len(df))
    missing_by_col = {str(col): int(df[col].isna().sum()) for col in df.columns}

    working = df[[txn_col, item_col]].copy()

    # Step 1: Drop rows with missing values in key columns
    non_missing_mask = working[txn_col].notna() & working[item_col].notna()
    missing_rows_removed = int((~non_missing_mask).sum())
    working = working.loc[non_missing_mask].copy()

    # Step 2: Strip whitespace and remove invalid placeholder or cancelled entries
    working[txn_col] = working[txn_col].astype(str).str.strip()
    working[item_col] = working[item_col].astype(str).str.strip()

    valid_mask = _is_valid_series(working[txn_col]) & _is_valid_series(working[item_col])
    # Exclude negative/cancelled invoice numbers commonly starting with 'C' when numeric IDs exist
    not_cancelled = ~working[txn_col].str.match(r"^C\d+$", case=False, na=False)
    valid_mask = valid_mask & not_cancelled

    invalid_rows_removed = int((~valid_mask).sum())
    working = working.loc[valid_mask].copy()

    # Step 3: Remove duplicate (Transaction, Product) pairs so basket encoding stays binary
    before_dedup = len(working)
    working = working.drop_duplicates(subset=[txn_col, item_col]).reset_index(drop=True)
    duplicate_rows_removed = int(before_dedup - len(working))

    stats = {
        "raw_rows": raw_rows,
        "missing_rows_removed": missing_rows_removed,
        "invalid_rows_removed": invalid_rows_removed,
        "duplicate_rows_removed": duplicate_rows_removed,
        "total_removed": missing_rows_removed + invalid_rows_removed + duplicate_rows_removed,
        "clean_rows": int(len(working)),
        "unique_transactions": int(working[txn_col].nunique()),
        "unique_products": int(working[item_col].nunique()),
        "missing_by_col": missing_by_col,
    }
    return working, stats


def create_basket_matrix(
    cleaned_df: pd.DataFrame,
    txn_col: str,
    item_col: str,
) -> pd.DataFrame:
    """Transform cleaned transaction rows into a binary (1/0) one-hot encoded basket matrix."""
    if cleaned_df.empty:
        return pd.DataFrame()

    basket = (
        cleaned_df.assign(_present=1)
        .pivot_table(
            index=txn_col,
            columns=item_col,
            values="_present",
            aggfunc="max",
            fill_value=0,
        )
    )
    basket = (basket > 0).astype("int8")
    basket.index.name = txn_col
    basket.columns.name = None
    return basket
