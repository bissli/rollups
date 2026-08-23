import copy
import datetime
import json
import logging
import math
import operator
import random
from collections import defaultdict
from collections.abc import Callable
from functools import wraps
from pydoc import locate
from typing import Any, Self

import numpy as np
from opendate import Date, DateTime, Time

import libb
from libb import lazydict as attrdict

from . import aggregate, io, join
# Re-exported so a caller importing from this module keeps working, and
# so the package root can name them. noqa: the formatter strips an
# import it sees as unused, and these have no local caller.
from .io import log_excel_errors  # noqa: F401
from .io import on_error_randomize  # noqa: F401
from .types import _cached_date_parse  # noqa: F401
from .types import _cached_datetime_parse  # noqa: F401
from .types import _convert_value  # noqa: F401
from .types import force_type  # noqa: F401
from .types import infer_numeric_type  # noqa: F401
from .types import is_dynamic_date_code  # noqa: F401
from .types import islistoftuples  # noqa: F401
from .types import smart_type  # noqa: F401

logger = logging.getLogger(__name__)


def ensure_types_converted(method):
    """Decorator to ensure types are converted before accessing data.
    """
    @wraps(method)
    def wrapper(self, *args, **kwargs):
        self._ensure_types_converted()
        return method(self, *args, **kwargs)
    return wrapper


class DataSet:
    """A list of dictionary rows, each column carrying a Python type.

    Behaves as a list over its rows, which are reached by index, by slice,
    and by iteration. On top of that sit filtering, grouping, joining,
    pivoting, and aggregation. A column type validates and converts the
    values under it.

    Notes
    -----
    - Reading `.summary` calls `add_summary_row()` when no summary was asked
      for, so a caller never has to prepare one.
    """

    def __init__(
        self,
        container: list[attrdict] = None,
        columns: list[tuple[str, type]] = None,
        cols: list[str] | None = None,
        typs: list[type] | None = None,
        page: int | None = None,
        per_page: int | None = None,
        total: int | None = None,
        exemplar: int | None = 0,
        check_types: bool = True,
        infer_numeric_strings: bool = False,
    ) -> Self:
        """DataSet constructor from an iterable of dicts or another DataSet.

        Parameters
        ----------
        container : list of attrdict or DataSet, default None
            Row dictionaries, or another DataSet to copy.
        columns : list of tuple, default None
            (column_name, column_type) pairs.
        cols : list of str or None, default None
            Column names, an alternative to `columns`.
        typs : list of type or None, default None
            Column types, an alternative to `columns`.
        page : int or None, default None
            Current page number for pagination.
        per_page : int or None, default None
            Rows per page.
        total : int or None, default None
            Rows across all pages. None counts the container.
        exemplar : int or None, default 0
            Row index that supplies the column names and starts type
            inference.
        check_types : bool, default True
            If True, validate and convert types on first access.
        infer_numeric_strings : bool, default False
            If True, a numeric string yields int or float rather than str.

        Notes
        -----
        - Set `cols` and `typs` yourself, or let them come from
          item[exemplar].
        - Type conversion is lazy: it runs on the first read, not here.

        """
        if columns is None:
            columns = []
        if container is None:
            container = []
        if isinstance(container, DataSet):
            self.container = list(container.container)
            self.columns = list(container.columns)
            self._summary_args = container._summary_args[:]
            self._types_converted = container._types_converted
        else:
            self.container = [row if isinstance(row, attrdict) else attrdict(row) for row in container]
            self.columns = list(columns) \
                if columns \
                else DataSet.guess_columns(self.container, cols=cols,
                                           typs=typs, exemplar=exemplar,
                                           infer_numeric_strings=infer_numeric_strings)
            self._summary_args = ()
            self._types_converted = False

        self._check_types = check_types

        self.page = page
        self.per_page = per_page
        self.total = total or len(self.container)
        self.pageable = None

    def __repr__(self):
        return f'{self.__class__.__name__}(cols={len(self.cols)}, rows={len(self.container)})'

    def __eq__(self, other):
        return self.container == other.container

    #
    # container-like methods
    #

    def append(self, obj: attrdict, validate: bool = False) -> None:
        """Append a dict row to the dataset.

        Parameters
        ----------
        obj : attrdict
            Row to append.
        validate : bool, default False
            If True, convert types now. If False, conversion waits for
            the first read, which is faster but defers any error.
        """
        if validate and self.columns and self._check_types:
            converted = attrdict()
            for name, typ in self.columns:
                val = obj.get(name)
                converted[name] = _convert_value(val, typ)
            self.container.append(converted)
        else:
            self.container.append(obj)
            self._types_converted = False

    @ensure_types_converted
    def __getitem__(self, key: int) -> attrdict:
        # if they ask for a slice, return a new Dataset
        if isinstance(key, slice):
            ds = self.copy()
            ds.container = ds.container[key]
            return ds
        return self.container[key]

    def __len__(self) -> int:
        return len(self.container)

    def __bool__(self) -> bool:
        return bool(self.__len__())

    def __nonzero__(self):
        return self.__bool__()

    @ensure_types_converted
    def __iter__(self):
        return self.container.__iter__()

    def extend(self, sequence: list[attrdict], validate: bool = False) -> None:
        """Extend the dataset by a sequence of rows or another DataSet.

        Parameters
        ----------
        sequence : list of attrdict or DataSet
            Rows to add.
        validate : bool, default False
            If True, convert types now. If False, conversion waits for
            the first read, which is faster but defers any error.
        """
        if validate and self.columns and self._check_types:
            for obj in sequence:
                converted = attrdict()
                for name, typ in self.columns:
                    val = obj.get(name)
                    converted[name] = _convert_value(val, typ)
                self.container.append(converted)
        else:
            self.container.extend(sequence)
            self._types_converted = False

    def sort(self, key=None, reverse=False) -> Self:
        """Sort rows the way list.sort does.

        Parameters
        ----------
        key : Callable or str or list or tuple of str or None, default None
            A callable sorts by its result; a name or list or tuple of
            names sorts by those attributes. None sorts by each column
            value, left to right.
        reverse : bool, default False
            If True, sort descending.

        Returns
        -------
        DataSet
            Self, sorted in place.

        Notes
        -----
        - For SQL-style `order by` with per-column direction, use
          `sort_data` instead.
        """
        if not key:
            self.container.sort(
                key=lambda row: tuple(row[c] for c in self.cols),
                reverse=reverse)
            return self
        if callable(key):
            self.container.sort(key=key, reverse=reverse)
            return self
        if not isinstance(key, list | tuple):
            key = (key,)
        self.container.sort(key=operator.attrgetter(*key), reverse=reverse)
        return self

    def reverse(self) -> Self:
        """Reverse the row order in place, returning self.
        """
        self.container.reverse()
        return self

    def order(self, col, *args):
        """Rearrange rows to the order of the values given.

        Parameters
        ----------
        col : str
            Column matched against each value.
        *args
            Values, in the order the rows should end up.

        Notes
        -----
        - Assumes the values are unique within the column.
        - Requires one value per row; a mismatch raises AssertionError.
        """
        assert len(self.container) == len(args), 'container length != args length'
        for arg in args:
            try:
                self.container += [self.container.pop(find(self.container, col, arg, raise_err=True))]
            except ValueError:
                pass

    def pop(self, col, val) -> attrdict | None:
        """Remove and return the first row whose `col` holds `val`.

        Parameters
        ----------
        col : str
            Column to read.
        val : Any
            Value to match.

        Returns
        -------
        attrdict or None
            The removed row, or None where nothing matched.
        """
        for i, row in enumerate(self.container):
            if row[col] == val:
                return self.container.pop(i)

    def __add__(self, other) -> Self:
        ds = self.copy()
        ds.extend(other)
        return ds

    def ensure_types(self) -> None:
        """Convert every value to its column's type, unless already done.

        Notes
        -----
        - Idempotent, and cheap to call again: it reads one flag and
          returns. A function outside this class that reads rows should
          call it first rather than assume a caller did.
        - Does nothing where the dataset was built with
          `check_types=False`.
        """
        if not self._types_converted and self._check_types:
            self.convert_container_types()
            self._types_converted = True

    def _ensure_types_converted(self) -> None:
        """Convert types lazily on first access if needed. Alias of
        `ensure_types`.
        """
        self.ensure_types()

    def _copy_structure(self) -> Self:
        """Copy helper for shared structure between copy/shallowcopy/deepcopy.
        """
        ds = self.__class__()
        ds.columns = list(self.columns) if self.columns else self.columns
        ds._summary_args = self._summary_args
        ds._check_types = getattr(self, '_check_types', True)
        ds.page = self.page
        ds.per_page = self.per_page
        ds.total = self.total
        ds.pageable = self.pageable
        return ds

    def copy(self, empty: bool = False) -> Self:
        """Create a new container holding the same row objects.

        Parameters
        ----------
        empty : bool, default False
            If True, return the structure with no rows.

        Returns
        -------
        DataSet
            New dataset sharing the original's row objects.

        Notes
        -----
        - The container is new but the rows are not: editing a row
          through either dataset is visible from the other.
        - Column metadata and pagination settings carry over.
        """
        ds = self._copy_structure()
        ds.container = [] if empty else list(self.container)
        ds._types_converted = getattr(self, '_types_converted', False)
        return ds

    def shallowcopy(self, empty: bool = False) -> Self:
        """Create a new container holding new rows over the same values.

        Parameters
        ----------
        empty : bool, default False
            If True, return the structure with no rows.

        Returns
        -------
        DataSet
            New dataset with its own row objects.

        Notes
        -----
        - Each row is a new attrdict, so adding or removing a column on
          one dataset leaves the other alone.
        - The values inside the rows are still shared, so mutating a
          list, dict or DataSet held in a row is visible from both.
        - Useful for aggregating with metadata columns without paying
          for a deep copy.
        """
        ds = self._copy_structure()
        ds.container = [] if empty else [attrdict(row) for row in self.container]
        ds._types_converted = getattr(self, '_types_converted', False)
        return ds

    def deepcopy(self) -> Self:
        """Create a fully independent copy sharing nothing.

        Returns
        -------
        DataSet
            New dataset with its own container, rows and values.

        Notes
        -----
        - Nested structures are copied too, so neither dataset can
          reach the other's data.
        - Type conversion runs on this dataset first where it has not
          already, so the copy holds converted values.
        """
        if not self._types_converted:
            self.convert_container_types()
            self._types_converted = True
        ds = self._copy_structure()
        ds.container = [attrdict(copy.deepcopy(dict(row))) for row in self.container]
        ds._types_converted = True
        return ds

    def dedupe(self, keys, filter_fn=None):
        """Remove duplicate rows by key.

        Parameters
        ----------
        keys : str or list of str
            Column(s) whose combined value identifies a duplicate.
        filter_fn : Callable or None, default None
            Chooses which row of a duplicate group to keep. None keeps
            the first occurrence.

        Returns
        -------
        DataSet
            New dataset holding one row per distinct key.

        Notes
        -----
        - Will NOT work where a key holds an unhashable value such as a
          list or a nested DataSet.
        - Where `filter_fn` matches no row in a group, the group's first
          row is kept.
        """
        keys = keys if isinstance(keys, list | tuple) else [keys]

        if filter_fn is None:
            d = {}
            for i, row in enumerate(self.unwind(*keys)):
                if row not in d:
                    d[row] = i
            ix = set(d.values())
            uq = [row for i, row in enumerate(self) if i in ix]
        else:
            groups = {}
            for row in self:
                key_vals = tuple(row[k] for k in keys)
                if key_vals not in groups:
                    groups[key_vals] = []
                groups[key_vals].append(row)

            uq = []
            for key_vals in sorted(groups.keys()):
                group_rows = groups[key_vals]
                matched = False
                for row in group_rows:
                    if filter_fn(row):
                        uq.append(row)
                        matched = True
                        break
                if not matched:
                    uq.append(group_rows[0])

        return self.__class__(uq, cols=self.cols, typs=self.typs)

    def itemize(self):
        """Split each row into a DataSet of its own.

        Returns
        -------
        list of DataSet
            One single-row dataset per row, carrying the same columns.
        """
        tods = lambda row: self.__class__([row], cols=self.cols, typs=self.typs)
        ncols = len(self.cols)
        return [tods(dict(zip(self.cols, x if ncols > 1 else (x,))))
                for x in self.unwind(*self.cols)]

    def sample(self, n):
        """Return a random sample of `n` rows.

        Parameters
        ----------
        n : int
            Rows wanted. More than the dataset holds returns them all,
            and a negative count returns none.

        Returns
        -------
        DataSet
            Deep copy holding the sampled rows.
        """
        sample = self.deepcopy()
        sample.container = random.sample(sample.container, min(max(n, 0), len(self.container)))
        return sample

    def shift(self, colname, periods=1, new_colname=None) -> None:
        """Shift a column's values forward or backward.

        Parameters
        ----------
        colname : str
            Column to shift.
        periods : int, default 1
            Positions to shift. Positive moves values forward, negative
            backward. Zero leaves the values alone and logs a warning.
        new_colname : str or None, default None
            Column to write to. None writes back over `colname`.

        Notes
        -----
        - Vacated positions hold None, and shifting by at least the row
          count leaves the column all None.
        """
        colval = list(self.unwind(colname))
        if periods > 0:
            if periods >= len(colval):
                colval = [None] * len(colval)
            else:
                colval = [None] * periods + colval[:-periods]
        elif periods < 0:
            if -periods >= len(colval):
                colval = [None] * len(colval)
            else:
                colval = colval[-periods:] + [None] * -periods
        else:
            logger.warning('Shifting by 0, results will be unshifted')
        colix = [i for i, x in enumerate(list(zip(*self.columns))[0]) if x == colname][0]
        coltyp = self.columns[colix][1]
        new_colname = new_colname or colname
        self.add_column(new_colname, coltyp, values=colval)

    def diff(self, colname: str, new_colname: str, index: int = 0) -> None:
        """Difference a column's consecutive values into a new column.

        Parameters
        ----------
        colname : str
            Source column, which must be int or float.
        new_colname : str
            Name for the new difference column.
        index : int, default 0
            Where the differencing pivots: 0 differences forward
            (current minus previous, first None), -1 differences
            backward (current minus next, last None), and any other
            index pivots at that row, which itself holds None.

        Raises
        ------
        ValueError
            If `index` falls outside the rows.
        """
        colix = self.cols.index(colname)
        coltyp = self.columns[colix][1]
        assert coltyp in {int, float}, 'only supports int and float types'

        values = list(self.unwind(colname))

        if not values:
            self.add_column(new_colname, coltyp, values=[])
            return

        last_idx = len(values) - 1

        if index < -1 or (index > last_idx and index != -1):
            raise ValueError(f'index {index} out of range for dataset with {len(values)} rows')

        diffs = []

        if index == 0:
            diffs.append(None)
            diffs.extend(libb.safe_diff(values[i], values[i-1]) for i in range(1, len(values)))
        elif index in {-1, last_idx}:
            diffs.extend(libb.safe_diff(values[i], values[i+1]) for i in range(len(values) - 1))
            diffs.append(None)
        else:
            for i in range(len(values)):
                if i < index:
                    diffs.append(libb.safe_diff(values[i], values[i+1]))
                elif i == index:
                    diffs.append(None)
                else:
                    diffs.append(libb.safe_diff(values[i], values[i-1]))
        self.add_column(new_colname, coltyp, values=diffs)

    def pct_change(self, colname, new_colname):
        """Calculate percent change between consecutive values.

        Parameters
        ----------
        colname : str
            Source column, which must be int or float.
        new_colname : str
            Name for the new percent change column.

        Notes
        -----
        - The first value is None, matching numpy.
        - Forward direction only, one period.
        """
        colix = [i for i, x in enumerate(list(zip(*self.columns))[0]) if x == colname][0]
        coltyp = self.columns[colix][1]
        assert coltyp in {int, float}, 'only supports int and float types'
        colval = list(self.unwind(colname))
        self.add_column(new_colname, float, values=libb.pct_change(colval))

    def backfill(self, colname: str, new_colname: str | None = None) -> None:
        """Fill None values using the nearest non-None value.

        Parameters
        ----------
        colname : str
            Column to read values from.
        new_colname : str or None, default None
            Column to write to. None writes back over `colname`.

        Notes
        -----
        - Each None takes the value that precedes it; a run of leading
          Nones takes the first value that follows. An all-None column
          comes back unchanged.
        - The filled column keeps the source column's position and type,
          so a caller indexing by column order stays aligned.
        """
        colix = self.cols.index(colname)
        coltyp = self.columns[colix][1]
        values = libb.backfill(list(self.unwind(colname)))
        new_colname = new_colname or colname
        self.add_column(new_colname, coltyp, index=colix, values=values)

    #
    # pagination
    #

    @property
    def pages(self) -> int:
        """Total number of pages, from `total` and `per_page`.
        """
        return int(math.ceil(float(self.total) / float(self.per_page)))

    @property
    def has_prev(self) -> bool:
        """Whether a page precedes the current one.
        """
        return self.page > 1

    @property
    def has_next(self) -> bool:
        """Whether a page follows the current one.
        """
        return self.page < self.pages

    def get_pages(self, start_max=2, left_this=2, right_this=5, end_max=2):
        """Generate the page numbers to render for a pager.

        Parameters
        ----------
        start_max : int, default 2
            Pages always shown at the start.
        left_this : int, default 2
            Pages shown before the current one.
        right_this : int, default 5
            Pages shown after the current one.
        end_max : int, default 2
            Pages always shown at the end.

        Returns
        -------
        Iterator
            Page numbers, with '...' standing in for each gap.
        """
        last_p = 0
        for this_p in range(1, self.pages + 1):
            to_show = (
                this_p <= start_max
                or (self.page - left_this - 1 < this_p < self.page + right_this)
                or self.pages - end_max < this_p
            )
            if to_show:
                end_of_ellipsis = this_p != last_p + 1
                if end_of_ellipsis:
                    yield '...'
                yield this_p
                last_p = this_p

    #
    # utility methods
    #

    def sort_data(self, *columns) -> Self:
        """Sort rows by column name, as SQL `order by` does.

        Parameters
        ----------
        *columns : str
            Column names, most significant first. Prefix a name with
            `-` to sort that column descending.

        Returns
        -------
        DataSet
            Self, sorted in place.

        Notes
        -----
        - For a sort driven by a function rather than column names,
          use `sort`.
        """
        libb.multikeysort(self.container, columns, inplace=True)
        return self

    def filter_data(self, pattern_or_predicate: str | Callable | None,
                    replace=None, logkey=None, inplace=True) -> Self | None:
        """Filter rows, in place unless told otherwise.

        Parameters
        ----------
        pattern_or_predicate : str or Callable or None
            A callable keeps the rows it returns true for; a string is
            matched against every str field, case-insensitive. None
            filters nothing.
        replace : Callable or None, default None
            Rewrites the matched text. String patterns only.
        logkey : str or None, default None
            Column named in a debug line for each row dropped.
        inplace : bool, default True
            If True, modify this dataset and return None; if False,
            return a filtered copy.

        Returns
        -------
        DataSet or None
            None where `inplace` is True, else the filtered copy.

        Notes
        -----
        - With `inplace` False and types already converted, the copy is
          a deep one; prefer `inplace` True where that cost matters.
        - `replace` saves each original value under `{col}~orig` so a
          formatter can still reach it.
        """
        if callable(pattern_or_predicate):
            predicate = pattern_or_predicate
        elif pattern_or_predicate:
            predicate = lambda row: self._match(row, pattern_or_predicate, replace)
        else:
            return None if inplace else self.deepcopy()

        if logkey:
            filtered_rows = []
            for row in self.container:
                if predicate(row):
                    filtered_rows.append(row)
                else:
                    logger.debug(f'Filtered {logkey} {row.get(logkey)} from dataset')
        else:
            filtered_rows = [row for row in self.container if predicate(row)]

        if inplace:
            self.container = filtered_rows
            return
        else:
            result = self.copy(empty=True)
            if self._types_converted:
                result.container = [attrdict(copy.deepcopy(dict(row))) for row in filtered_rows]
                result._types_converted = True
            else:
                result.container = [attrdict(row) for row in filtered_rows]
            return result

    def partition(self, partition_func) -> defaultdict[Any, Self]:
        """Split the dataset into new datasets by a partition function.

        Parameters
        ----------
        partition_func : Callable
            Called per row; its result keys the partition the row lands
            in.

        Returns
        -------
        defaultdict
            Partition key to DataSet. Reading an absent key creates an
            empty dataset carrying this one's columns.
        """
        partitions = defaultdict(lambda: self.copy(empty=True))
        for row in self:
            partitions[partition_func(row)].append(row)
        return partitions

    #
    # summary functions
    #

    def _summary_columns(self, label_idx: int) -> list[str]:
        """Numeric columns to total, excluding the label column.
        """
        label_col = self.columns[label_idx][0] if 0 <= label_idx < len(self.columns) else None
        return [col for col, typ in self.columns if libb.isnumeric(typ) and col != label_col]

    def add_summary_row(
        self,
        label_idx=0,
        label='Total',
        columns: list[str] | None = None,
        cols_funcs: list[tuple[str, Callable]] | None = None,
    ) -> None:
        """Declare a summary row, to be computed on first read.

        Parameters
        ----------
        label_idx : int, default 0
            Column index the label is written into.
        label : str, default 'Total'
            Text placed in the label column.
        columns : list of str or None, default None
            Columns to total. None takes every numeric column but the
            label column.
        cols_funcs : list of tuple or None, default None
            (column, function) pairs, used in place of `columns` to
            aggregate with something other than sum.

        Notes
        -----
        - Nothing is computed here, so filtering or editing the rows
          after this call still gives the right total.
        """
        if columns is None:
            columns = self._summary_columns(label_idx)
        self._summary_args = (label_idx, label, columns, cols_funcs)

    def calc_summary_row(
        self,
        label_idx=0,
        label='Total',
        columns: list[str] | None = None,
        cols_funcs: list[tuple[str, Callable]] | None = None,
    ) -> attrdict:
        """Compute the summary row now.

        Parameters
        ----------
        label_idx : int, default 0
            Column index the label is written into.
        label : str, default 'Total'
            Text placed in the label column.
        columns : list of str or None, default None
            Columns to total.
        cols_funcs : list of tuple or None, default None
            (column, function) pairs, used in place of `columns` to
            aggregate with something other than sum.

        Returns
        -------
        attrdict
            The summary row, carrying `__is_summary__` True and None in
            every column not aggregated.
        """
        if cols_funcs is None and columns is None:
            columns = self._summary_columns(label_idx)
        total = self.bucket([], cols_funcs) if cols_funcs else self.bucket([], columns)
        if total:
            summary = total[0].copy()
            summary['__is_summary__'] = True
            label_col = self.columns[label_idx][0] if 0 <= label_idx < len(self.columns) else None
            if label_col:
                summary[label_col] = label
            summary.update((c, None) for c in self.cols if c not in summary)  # fill empty vals
        else:
            summary = attrdict((c, None) for c in self.cols)
        return summary

    @property
    def summary(self) -> attrdict:
        """The summary row, recomputed on every read.

        Notes
        -----
        - A caller never has to declare one first; reading this totals
          every numeric column where nothing was declared.
        - Nothing is cached, so editing or filtering the rows and
          reading again totals the current rows.
        """
        if not getattr(self, '_summary_args', None):
            self.add_summary_row()
        return self.calc_summary_row(*self._summary_args)

    @classmethod
    def is_summary_row(cls, row):
        """Check whether a row is a summary row rather than data.
        """
        return row.get('__is_summary__', False)

    def add_summary_column(self, label, columns=None, row_func=sum) -> None:
        """Add a column holding a per-row total across other columns.

        Parameters
        ----------
        label : str
            Name for the new column.
        columns : list of str or None, default None
            Columns to combine. None uses every column.
        row_func : Callable, default sum
            Applied to each row's values.

        Notes
        -----
        - Falsy values are skipped, so a zero or None does not reach
          `row_func`.
        - The new column is typed float.
        """
        columns = columns or [_[0] for _ in self.columns]
        for row in self:
            row[label] = row_func(row[_] for _ in columns if row.get(_))
        self.add_column(label, float)

    #
    # columns property
    #

    @property
    def columns(self) -> list[tuple[str, type]]:
        """Return mutable list of column definitions.
        """
        return self._columns

    @columns.setter
    def columns(self, columns: list[tuple[str, type]]) -> None:
        """Set column definitions, keeping the last of any repeated name.
        """
        seen = {}
        for col, typ in columns:
            if col in seen:
                del seen[col]
            seen[col] = typ
        self._columns = list(seen.items())

    @property
    def colmap(self) -> dict[str, type]:
        """Column definitions as a name-to-type dict.
        """
        return dict(self._columns)

    @property
    def cols(self) -> list[str]:
        """Column names, in column order.
        """
        return [col for col, _ in self._columns]

    @cols.setter
    def cols(self, cols: list[str]) -> None:
        """Set column names, preserving types for existing columns.
        """
        current_types = dict(self._columns)
        new_columns = []
        for col in cols:
            if col not in current_types:
                logger.warning(f'Column {col} not in current schema, skipping')
                continue
            new_columns.append((col, current_types[col]))
        self._columns = libb.unique(new_columns)

    @property
    def typs(self) -> list[type]:
        """Column types, in column order.
        """
        return [typ for _, typ in self._columns]

    @ensure_types_converted
    def add_column(self, name, typ, index=None,
                   value: Callable | None = None, values=None) -> None:
        """Add a column, or update one already present.

        Parameters
        ----------
        name : str
            Column name.
        typ : type
            Column type.
        index : int or None, default None
            Position to insert at. None appends. An existing column is
            dropped from its old position either way, so it moves to the
            end unless an index puts it back.
        value : Callable or Any or None, default None
            A callable is called per row to produce the value; anything
            else is written to every row.
        values : list or None, default None
            One value per row, positionally matched. Takes precedence
            over `value`.

        Raises
        ------
        ValueError
            If `values` is given and its length is not the row count.

        Notes
        -----
        - With neither `value` nor `values`, an existing value survives
          and a missing one becomes None.
        - Changing an existing column's type re-arms type conversion, so
          the new type is applied on the next read.
        """
        existing_index = next((i for i, (c, _) in enumerate(self._columns) if c == name), None)
        old_type = self._columns[existing_index][1] if existing_index is not None else None
        if existing_index is not None:
            del self._columns[existing_index]
        if index is not None:
            self._columns.insert(index, (name, typ))
        else:
            self._columns.append((name, typ))
        self.columns = self._columns
        if values is not None:
            if len(values) != len(self.container):
                raise ValueError(
                    f'values length {len(values)} must match dataset length {len(self.container)} for column {name}')
            for row, val in zip(self.container, values):
                row[name] = val
        elif value is not None:
            value_fn = value if callable(value) else lambda _: value
            for row in self.container:
                row[name] = value_fn(row)
        else:
            for row in self.container:
                row[name] = row.get(name)
        if old_type is not None and old_type != typ:
            self._types_converted = False

    def remove_column(self, name: str) -> None:
        """Remove a column and drop it from every row.

        Parameters
        ----------
        name : str
            Column to remove. A name not present is a no-op, logged at
            debug. Matching is case-sensitive.
        """
        if name in self.colmap:
            columns = [(c, t) for c, t in self.columns if c != name]
            self.columns = columns
        else:
            logger.debug(f'{name} not in DataSet columns')
        for row in self.container:
            if name in row:
                del row[name]

    def rename_column(self, name: str, rename: str) -> None:
        """Rename a column in the dataset.

        Parameters
        ----------
        name : str
            Column to rename. A name not present is a no-op.
        rename : str
            New name. An existing column of this name is removed first.
        """
        if name == rename:
            logger.warning(f'Column {name} matches rename column {rename}')
            return
        if rename in self.colmap and name in self.colmap:
            self.remove_column(rename)
        if name in self.colmap:
            idx = self.cols.index(name)
            columns = list(self.columns)
            columns[idx] = (rename, columns[idx][1])
            self.columns = columns
        for row in self.container:
            if name in row:
                row[rename] = row.pop(name, None)

    #
    # constructors
    #

    @classmethod
    def from_empty(cls, columns: list[tuple[str, type]]) -> Self:
        """Create a one-row DataSet holding each column's empty value.

        Parameters
        ----------
        columns : list of tuple
            (name, type) pairs defining the columns.

        Returns
        -------
        DataSet
            One row: '' for str, 0 for int, 0.0 for float, None else.
        """
        emptyrow = attrdict()
        for col, typ in columns:
            if typ == str:
                emptyrow[col] = ''
            elif typ == int:
                emptyrow[col] = 0
            elif typ == float:
                emptyrow[col] = 0.
            else:
                emptyrow[col] = None
        return cls([emptyrow], columns=columns)

    @classmethod
    def from_list(cls, rows, cols, typs) -> Self:
        """Build a DataSet from rows of tuples.

        Parameters
        ----------
        rows : iterable of tuple
            One tuple per row, in `cols` order.
        cols : list of str
            Column names.
        typs : list of type
            Column types, positionally matched to `cols`.

        Returns
        -------
        DataSet
            Rows converted to dicts under the given names and types.

        Raises
        ------
        ValueError
            If `cols` and `typs` are of different length.
        """
        if len(cols) != len(typs):
            raise ValueError('cols and typs length mismatch')
        return cls([attrdict(list(zip(cols, row))) for row in rows],
                   columns=list(zip(cols, typs)))

    @classmethod
    def from_dataframe(cls, df, cols=None, columns=None) -> Self:
        """Build a DataSet from a pandas DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            Frame to convert.
        cols : list of str or None, default None
            Columns to keep. None keeps them all.
        columns : list of tuple or None, default None
            (name, type) pairs overriding the guessed type for those
            columns.

        Returns
        -------
        DataSet
            Rows of the frame, with NaN read as None.
        """
        _columns = dict(guess_dataframe_dataset_columns(df))
        if cols:
            _columns = {col: typ for col, typ in _columns.items() if col in cols}
            df = df[list(_columns.keys())]
        if columns:
            for col, typ in columns:
                if col in _columns:
                    _columns[col] = typ

        rows = df.to_dict('records')
        for row in rows:
            for key in list(row.keys()):
                val = row[key]
                if isinstance(val, float) and np.isnan(val):
                    row[key] = None

        rows = [attrdict(row) for row in rows]
        return cls(rows, columns=list(_columns.items()))

    def convert_container_types(self):
        """Convert every row's values to the column types.

        Notes
        -----
        - A row missing a column gains it, holding None.
        - A column with no declared type is left as it is.
        """
        cols_to_convert = [(name, typ) for name, typ in self.colmap.items()
                           if typ not in {None, None.__class__}]

        for row in self.container:
            for name, typ in self.columns:
                if name not in row:
                    row[name] = None
            for name, typ in cols_to_convert:
                val = row[name]
                if val is None:
                    continue
                if typ in {Date, datetime.date} and isinstance(val, datetime.date) and not isinstance(val, Date):
                    row[name] = Date.instance(val)
                    continue
                if typ in {DateTime, datetime.datetime} and isinstance(val, datetime.datetime) and not isinstance(val, DateTime):
                    row[name] = DateTime.instance(val)
                    continue
                if typ in {Time, datetime.time} and isinstance(val, datetime.datetime | datetime.time) and not isinstance(val, Time):
                    row[name] = Time.instance(val)
                    continue
                if isinstance(val, typ):
                    continue
                row[name] = _convert_value(val, typ)

        self._types_converted = True

    def to_array(self, columns=None, numpy_type=None):
        """Convert selected columns to a numpy array.

        Parameters
        ----------
        columns : list of str or None, default None
            Columns to take, in order. None takes them all.
        numpy_type : type or None, default None
            dtype for the array. None uses the first column's type.

        Returns
        -------
        np.ndarray
            Two-dimensional array, one row per dataset row.
        """
        columns = columns or self.cols
        numpy_type = numpy_type or self.colmap[columns[0]]
        return np.array([[row[col] for col in columns] for row in self], dtype=numpy_type)

    def to_list(self):
        """Convert the DataSet to a grid of column tuples.

        Returns
        -------
        list of tuple
            One tuple per column, holding that column's values in row
            order.
        """
        return list(zip(*[list(row.values()) for row in self.container]))

    @classmethod
    def from_excel_sheets(cls, *args, **kwargs) -> Self:
        """Generator of (sheetname, DataSet) items.

        Returns
        -------
        Iterator of tuple
            (sheetname, DataSet) for each parsed sheet.

        Notes
        -----
        - Pass infer_numeric_strings=True to type numeric strings as int
          or float rather than str.
        """
        opts = {
            'columns': kwargs.pop('columns', None),
            'cols': kwargs.pop('cols', None),
            'typs': kwargs.pop('typs', None),
            'exemplar': kwargs.pop('exemplar', 0),
            'infer_numeric_strings': kwargs.pop('infer_numeric_strings', False),
        }
        for k, v in list(io.parse_excel_sheets(*args, **kwargs).items()):
            yield k, cls(v, **opts)

    @classmethod
    @log_excel_errors
    def from_excel(cls, *args, **kwargs) -> Self:
        """Create DataSet object from an excel sheet, inferring column types.

        Returns
        -------
        DataSet or dict
            Rows of the single parsed sheet, or an empty dict where the
            workbook yielded no sheet.

        Notes
        -----
        - If stream, pass in stream as key in kwargs.
        - Pass infer_numeric_strings=True to type numeric strings as int
          or float rather than str.
        """
        opts = {
            'columns': kwargs.pop('columns', None),
            'cols': kwargs.pop('cols', None),
            'typs': kwargs.pop('typs', None),
            'exemplar': kwargs.pop('exemplar', 0),
            'infer_numeric_strings': kwargs.pop('infer_numeric_strings', False),
        }
        sheetnames = kwargs.get('sheetnames', [])
        if not sheetnames:
            kwargs['first'] = True
        else:
            assert len(sheetnames) == 1, 'Only one sheet allowed in `from_excel`'
        rows = io.parse_excel_sheets(*args, **kwargs)
        return cls(rows[list(rows.keys())[0]], **opts) if rows else {}

    @classmethod
    def from_json(cls, data, raw=True) -> Self | tuple[Self, dict]:
        """Build a DataSet from json text.

        Parameters
        ----------
        data : str
            Json text. ISO date strings are parsed back to dates.
        raw : bool, default True
            If True, read a bare array of row objects. If False, read
            an object carrying `data`, and optionally `order`, `types`
            and any further keys.

        Returns
        -------
        DataSet or tuple
            A DataSet where `raw` is True; otherwise (DataSet, other),
            where other holds every key but `data`, `order` and
            `types`.

        Notes
        -----
        - A declared type wins over the parsed one, so `types` of
          ['int'] reads 2.0 as 2 rather than as a float.
        """
        obj = json.loads(data, cls=libb.JSONDecoderISODate)
        if raw:
            return cls(obj)
        handle_date = {'date': datetime.date, 'datetime': datetime.datetime}
        types = [handle_date.get(typ, locate(typ)) for typ in (obj.get('types') or [])]
        ds = cls(obj['data'], cols=obj.get('order'), typs=types)
        other = {k: v for k, v in obj.items() if k not in {'data', 'order', 'types'}}
        return ds, other

    @classmethod
    def read(cls, file_or_name, **kw) -> Self:
        """Read a DataSet from a csv file or open handle.

        Returns
        -------
        DataSet
            Parsed rows, with `filename` set to the source.

        See Also
        --------
        rollups.io.read_csv_rows : the csv format, the type suffixes,
            and the rename_fields hook
        """
        rows, columns = io.read_csv_rows(file_or_name, **kw)
        ds = cls(rows, columns=columns)
        ds.filename = file_or_name
        return ds

    @classmethod
    def from_csv(cls, file_or_name, **kw) -> Self:
        """Read a DataSet from a csv file or open handle. Alias of `read`.
        """
        return cls.read(file_or_name, **kw)

    #
    # writers
    #

    def write_csv(self, path_or_buf, **kwargs):
        """Write the DataSet to a csv file or buffer.

        See Also
        --------
        rollups.io.write_csv_file : the contract, the format hooks, and
            the write-retry behavior
        """
        io.write_csv_file(self, path_or_buf, **kwargs)

    def write_excel(self, path_or_buf, **kwargs):
        """Write the DataSet to an Excel workbook.

        See Also
        --------
        rollups.io.write_excel_file : the contract, the cell conversion
            rules, and the write-retry behavior
        """
        io.write_excel_file(self, path_or_buf, **kwargs)

    def json(self, columns=None, raw=False,
             format_value=lambda x, y, _: x.get(y), **kw):
        """Serialize the DataSet to a json string.

        See Also
        --------
        rollups.io.to_json : the contract and the shape of the object
            it emits
        """
        return io.to_json(self, columns=columns, raw=raw,
                          format_value=format_value, **kw)

    #
    # DB-like methods
    #

    @classmethod
    def join(cls, adataset: Self, akey: tuple[str], bdataset: Self,
             bkey: tuple[str], jointype: str = 'inner',
             amod: str = '', bmod: str = '',
             acol: list[str] | None = None, bcol: list[str] | None = None,
             first: bool = False, bfirst: bool = False) -> Self:
        """Join two datasets on their key columns.

        See Also
        --------
        rollups.join.join_datasets : the contract, the join-type table,
            and the worked examples

        Notes
        -----
        - The result takes the class this method was reached through, so
          `Sub.join(a, ...)` answers a `Sub` and `DataSet.join(a, ...)`
          answers a `DataSet` for the same inputs.
        """
        return join.join_datasets(adataset, akey, bdataset, bkey,
                                  jointype=jointype, amod=amod, bmod=bmod,
                                  acol=acol, bcol=bcol, first=first,
                                  bfirst=bfirst, cls=cls)

    def bucket(self, keycols: str,
               aggregations: list[tuple[str, Callable, Callable] | str],
               inplace: bool = False) -> Self:
        """Group rows by key columns and apply aggregation functions.

        Parameters
        ----------
        inplace : bool, default False
            If True, replace this dataset's rows and columns with the
            result and answer self.

        See Also
        --------
        rollups.aggregate.bucket_dataset : the contract, the aggregation
            format table, and the typing rules
        """
        result = aggregate.bucket_dataset(self, keycols, aggregations)
        if inplace:
            self.container = result.container
            self.columns = result.columns
            return self
        return result

    def flatten(self, kept, flattened, key='key', val='val') -> Self:
        """Reverse-pivot, moving each flattened column into its own row.

        See Also
        --------
        rollups.aggregate.flatten_dataset : the contract and the row
            count it produces
        """
        return aggregate.flatten_dataset(self, kept, flattened,
                                         key=key, val=val)

    def pivot(self, index_col, data_cols, pivot_col, aggr=sum,
              alias=lambda x, _: x, inplace: bool = False) -> Self:
        """Pivot dataset to turn row values into columns.

        Parameters
        ----------
        inplace : bool, default False
            If True, replace this dataset's rows and columns with the
            result and answer self.

        See Also
        --------
        rollups.aggregate.pivot_dataset : the contract, the alias rule,
            and the worked example
        """
        grouped = aggregate.pivot_dataset(self, index_col, data_cols,
                                          pivot_col, aggr=aggr, alias=alias)
        if inplace:
            self.container = grouped.container
            self.columns = grouped.columns
            return self
        return grouped

    def transpose(self, new_index_name, pivot_index=0):
        """Transpose so one column's values become the column names.

        See Also
        --------
        rollups.aggregate.transpose_dataset : the contract and the
            assumptions it makes about row order
        """
        return aggregate.transpose_dataset(self, new_index_name, pivot_index)
