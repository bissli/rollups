"""Row readers shared by the DataFrame-native tests.

Every test in this package compares a `pandas.DataFrame` against either
a hand-built expectation or the `DataSet` method it stands in for, so
each needs the same two readings of a result: rows as plain tuples, and
rows as dtype-blind tokens.
"""
from collections.abc import Iterable, Sequence
from typing import Any

import pandas as pd
from rollups import DataSet


def frame_rows(df: pd.DataFrame, cols: Sequence[str]) -> list[tuple]:
    """Rows of `df` as tuples in `cols` order, reading any null back as None.
    """
    # A container is never null, and pd.isna would answer elementwise
    # for one rather than with the single bool the guard needs.
    return [tuple(None if not isinstance(v, list | tuple | set | dict)
                  and pd.isna(v) else v for v in row)
            for row in df.loc[:, list(cols)].itertuples(index=False, name=None)]


def dataset_rows(ds: DataSet) -> list[tuple]:
    """Rows of `ds` as tuples in column order.
    """
    return [tuple(row[col] for col in ds.cols) for row in ds]


def tokens(rows: Iterable[tuple]) -> list[tuple[str, ...]]:
    """Rows as sorted, dtype-blind tokens: 1 and 1.0 alike, nulls alike.
    """
    def token(val: Any) -> str:
        if isinstance(val, list | tuple | set | dict):
            return f'{type(val).__name__}:{val!r}'
        if pd.isna(val):
            return 'null'
        if pd.api.types.is_bool(val):
            return f'bool:{bool(val)}'
        if pd.api.types.is_number(val):
            return f'num:{float(val)!r}'
        return f'{type(val).__name__}:{val!r}'
    return sorted(tuple(token(v) for v in row) for row in rows)
