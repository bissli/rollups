import logging
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
