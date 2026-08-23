import copy
import logging
import operator
from functools import wraps
from typing import Self


from libb import lazydict as attrdict

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
