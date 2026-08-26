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
from typing import TYPE_CHECKING, Any

import libb
from libb import lazydict

from .types import smart_type

if TYPE_CHECKING:
    from .core import DataSet

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
        """For passing-in unwound rows.

        Notes
        -----
        - Reads through the dict method rather than the row's own
          `get`, which is written in Python and answers a missing key
          with the attribute of that name, so a column named after one
          would aggregate a bound method.
        """
        values = (dict.get(row, col) for row in iterdict)
        return [val for val in values if val is not None] or [None]

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

    # Wrap each operation once. Wrapping inside the group loop instead
    # runs functools.wraps per group, which on a high-cardinality key
    # costs more than the aggregation it guards.
    aggcols = [(col, safe(op), filt, alias)
               for col, op, filt, alias in map(parse_aggregation, aggregations)]

    # Prepare data and keys
    keycols = list(keycols) if isinstance(keycols, list | tuple) else [keycols]
    data = dataset.container[:]

    # Sort for grouping with a functional key
    def sort_key(row):
        key = []
        for col in keycols:
            val = row[col]
            key.append((val is None, val) if isinstance(val, Hashable)
                       else (False, None))
        return tuple(key)

    data.sort(key=sort_key)

    # Group function
    keyfn = lambda row: tuple(row[col] for col in keycols)

    # Process each group and apply aggregations
    buckets = []
    for key, grouped in itertools.groupby(data, keyfn):
        rows = list(grouped)
        bucket = lazydict(zip(keycols, key))
        for col, op, filt, alias in aggcols:
            bucket[alias] = op(filt(rows))
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
    - The `val` column is typed `object`: it gathers columns of several
      types into one, so no single type describes what it holds.
    """
    if not isinstance(kept, list | tuple):
        kept = [kept]
    ds = dataset.__class__(
        columns=[(c, t) for c, t in dataset.columns if c in kept])
    ds.add_column(key, str)
    # `object` is the honest declaration here: the flattened columns
    # carry different types, so declaring float would either coerce
    # every value to float or reject the ones that are not.
    ds.add_column(val, object)
    if not kept:
        for row in dataset.container:
            for k in flattened:
                ds.append(lazydict({key: k, val: row[k]}))
        return ds
    for keeps, grouped in itertools.groupby(dataset.container, operator.attrgetter(*kept)):
        if not isinstance(keeps, list | tuple):
            keeps = [keeps]
        for row in grouped:
            for k in flattened:
                ds.append(lazydict(list(zip(kept, keeps)), **{key: k, val: row[k]}))
    return ds


def pivot_dataset(dataset: 'DataSet', index_col, data_cols, pivot_col,
                  aggr=sum, alias=lambda x, _: x) -> 'DataSet':
    """Pivot dataset to turn row values into columns.

    Each distinct value of `pivot_col` becomes a column, holding the
    aggregated `data_cols` for that `index_col` group.

    Parameters
    ----------
    dataset : DataSet
        Rows to pivot.
    index_col : str
        Column to use as the row index.
    data_cols : str or list of str
        Column(s) to aggregate.
    pivot_col : str
        Column whose values become the new column names.
    aggr : Callable, default sum
        Aggregation applied to each cell.
    alias : Callable, default lambda x, _: x
        Builds a column name from (pivot_value, data_col).

    Returns
    -------
    DataSet
        Pivoted dataset.

    Notes
    -----
    - Pivoting several data columns needs an `alias` that varies with
      the data column, or they collide on one generated name.
    - See docs/aggregation.md for a worked example.
    - Groups through `dataset.bucket`, the method, so a subclass that
      overrides it is honored.
    """
    dataset.ensure_types()
    if isinstance(data_cols, str):
        data_cols = [data_cols]
    new_cols = libb.unique(row[pivot_col] for row in dataset)
    filterby = \
        lambda col, data_col: \
        lambda rows: \
        [row[data_col] for row in rows if row[pivot_col] == col] or [0.0]
    pivoted = [(c, aggr, filterby(c, d), alias(c, d))
               for c in new_cols for d in data_cols]
    grouped = dataset.bucket(index_col, pivoted)

    return grouped


def transpose_dataset(dataset: 'DataSet', new_index_name: str,
                      pivot_index: int = 0) -> 'DataSet':
    """Transpose so one column's values become the column names.

    Parameters
    ----------
    new_index_name : str
        Name for the column holding the old column names.
    pivot_index : int, default 0
        Position of the column whose values become column names.

    Returns
    -------
    DataSet
        Transposed dataset.

    Notes
    -----
    - Assumes the rows are already in the order wanted, and uses
      every column.
    """
    pivot_col, _ = dataset.columns[pivot_index]
    cols = [c for c, _ in dataset.columns]
    new_index = cols[:pivot_index] + cols[(pivot_index + 1) :]
    new_rows = []
    new_cols = [new_index_name] + [old_row[pivot_col] for old_row in dataset]
    for idx in new_index:
        new_row = {new_index_name: idx}
        new_row.update({old_row[pivot_col]: old_row[idx] for old_row in dataset})
        new_rows.append(new_row)
    return dataset.__class__(new_rows, cols=new_cols)
