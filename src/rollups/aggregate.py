"""Operations that reshape one dataset.

Each function takes a dataset and answers a new one, built through the
input's own class. None of them mutates its input beyond the type
conversion a read triggers; the `inplace` flag lives on the method,
where the receiver is the thing being replaced.

Notes
-----
- See docs/aggregation.md for the aggregation formats and worked
  examples.
"""


import itertools
import logging
import operator
from collections.abc import Callable, Hashable
from functools import wraps
from typing import Any

from opendate import Date, DateTime, Time

from libb import lazydict as attrdict

from .types import smart_type

logger = logging.getLogger(__name__)


def bucket_dataset(dataset: 'DataSet', keycols: str | list[str],
                   aggregations: list[tuple[str, Callable, Callable] | str]
                   ) -> 'DataSet':
    """Group rows by key columns and apply aggregation functions.

    The SQL GROUP BY: one result row per distinct key combination.

    Parameters
    ----------
    dataset : DataSet
        Rows to group.
    keycols : str or list of str
        Column(s) to group by. An empty list totals every row into one.
    aggregations : list
        What to aggregate and how. Each entry is a column name, or a
        tuple of 1 to 4 items: (col), (col, op), (col, op, alias),
        (col, op, filter), or (col, op, filter, alias). A three-item
        tuple reads its third item by type -- a string is an alias, a
        callable is a filter.

    Returns
    -------
    DataSet
        One row per key combination, with the aggregated values.

    Notes
    -----
    - A bare column name sums it, skipping None.
    - A filter receives the group's rows and returns the values to
      aggregate; return a one-item fallback list rather than an empty
      one, or the operation has nothing to work on.
    - Aggregating one column twice needs an alias on each, or the
      second overwrites the first.
    - The result column keeps the source type unless the operation
      changed it; an unknown source type is inferred from the first
      non-None result.
    - See docs/aggregation.md for the format table and examples.
    """
    dataset.ensure_types()

    def non_none(iterdict, col):
        """For passing-in unwound rows."""
        result = [_.get(col) for _ in iterdict if _.get(col) is not None]
        return result or [None]

    def infer_type_from_rows(rows: list, col: str) -> type:
        """Infer column type from first non-None value in rows."""
        for row in rows:
            val = row.get(col)
            if val is not None:
                return smart_type(val)
        return type(None)

    def infer_aggregation_type(buckets: list, alias: str, original_type: type) -> type:
        """Infer type for aggregation column from bucket results.
        """
        if original_type is not None and original_type is not type(None):
            return original_type
        for bucket in buckets:
            val = bucket.get(alias)
            if val is not None:
                return smart_type(val)
        return str

    def get_type(col, original_type, result_value):
        """Infer type preserving original unless operation changed it.
        """
        if result_value is None:
            return original_type
        actual_type = result_value.__class__
        if original_type is float and actual_type is int:
            return float
        if actual_type in {Date, DateTime, Time}:
            return actual_type
        return actual_type

    def safe(func):
        """Wrapper to handle None in sum, max, etc in aggregation."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except (ValueError, TypeError):
                if args and isinstance(args[0], list):
                    items = args[0]
                    if items and items != [None] and len(items) > 0:
                        first_type = type(items[0])
                        if all(isinstance(x, first_type) for x in items):
                            if first_type in {set, frozenset}:
                                return first_type().union(*items)
                            if first_type in {list, tuple}:
                                merged = itertools.chain.from_iterable(items)
                                return first_type(merged)
                return None
        return wrapper

    def parse_aggregation(agg: Any) -> tuple[str, Callable, Callable, str]:
        """Parse an aggregation into (col, op, filter, alias).
        """
        default_filter = lambda col: lambda x: non_none(x, col)

        if not isinstance(agg, tuple | list):
            col = agg
            return col, sum, default_filter(col), col
        if len(agg) == 1:
            col = agg[0]
            return col, sum, default_filter(col), col
        elif len(agg) == 2:
            col, op = agg
            return col, op, default_filter(col), col
        elif len(agg) == 3:
            col, op, third = agg
            if callable(third):
                return col, op, third, col
            else:
                return col, op, default_filter(col), third
        elif len(agg) == 4:
            return agg
        else:
            raise ValueError(f'Aggregation tuple must be length 1-4, got {len(agg)}: {agg}')

    aggcols = list(map(parse_aggregation, aggregations))

    # Prepare data and keys
    keycols = list(keycols) if isinstance(keycols, list | tuple) else [keycols]
    data = dataset.container[:]

    # Sort for grouping with a functional key
    sort_key = lambda row: tuple((row[col] is None, row[col])
                                 if isinstance(row[col], Hashable)
                                 else (False, None)
                                 for col in keycols)
    data.sort(key=sort_key)

    # Group function
    keyfn = lambda row: tuple(row[col] for col in keycols)

    # Process each group and apply aggregations
    buckets = []
    for key, grouped in itertools.groupby(data, keyfn):
        rows = list(grouped)
        bucket = attrdict(zip(keycols, key))
        for col, op, filt, alias in aggcols:
            bucket[alias] = safe(op)(filt(rows))
        buckets.append(bucket)

    result = dataset.__class__(buckets)

    # Add key columns with types from original dataset
    for col in keycols:
        if col in dataset.colmap:
            result.add_column(col, dataset.colmap[col])
        else:
            inferred_type = infer_type_from_rows(result.container, col)
            result.add_column(col, inferred_type)

    # Add aggregation columns with inferred or original types
    for col, *_, alias in aggcols:
        original_type = dataset.colmap.get(col, None)
        original_type = infer_aggregation_type(buckets, alias, original_type)
        if buckets:
            result_value = buckets[0].get(alias)
            typ = get_type(alias, original_type, result_value)
        else:
            typ = original_type
        result.add_column(alias, typ)

    return result


def flatten_dataset(dataset: 'DataSet', kept, flattened,
                    key: str = 'key', val: str = 'val') -> 'DataSet':
    """Reverse-pivot, moving each flattened column into its own row.

    Parameters
    ----------
    dataset : DataSet
        Rows to reverse-pivot.
    kept : str or list of str
        Columns carried through unchanged.
    flattened : list of str
        Columns folded into rows, one row per column per input row.
    key : str, default 'key'
        Name of the column holding the flattened column's name.
    val : str, default 'val'
        Name of the column holding its value.

    Returns
    -------
    DataSet
        The kept columns plus `key` and `val`.

    Notes
    -----
    - Flattening n columns over m rows gives n * m rows.
    - The `val` column is typed float.
    """
    if not isinstance(kept, list | tuple):
        kept = [kept]
    # flatten puts values of different types into one val column,
    # which a later read would coerce to the column's float type,
    # so turn type checking off on the result.
    ds = dataset.__class__(
        columns=[(c, t) for c, t in dataset.columns if c in kept],
        check_types=False)
    ds.add_column(key, str)
    ds.add_column(val, float)
    if not kept:
        for row in dataset.container:
            for k in flattened:
                ds.append(attrdict({key: k, val: row[k]}))
        return ds
    for keeps, grouped in itertools.groupby(dataset.container, operator.attrgetter(*kept)):
        if not isinstance(keeps, list | tuple):
            keeps = [keeps]
        for row in grouped:
            for k in flattened:
                ds.append(attrdict(list(zip(kept, keeps)), **{key: k, val: row[k]}))
    return ds
