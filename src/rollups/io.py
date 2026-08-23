"""Reading and writing a dataset at the process boundary.

These functions cross to a file or a buffer. None of them constructs a
`DataSet`: the reader answers rows and columns, and the caller builds
the container. That is what keeps this module free of any dependency on
the class.

Notes
-----
- A csv header field may carry a type suffix: `name:s`, `age:i`,
  `score:f`, `on:b`, `when:d`. A field without one reads as str.
"""


import csv
import datetime
import itertools
import json
import logging
import os
import pathlib
import random
import time
from collections.abc import Callable
from functools import wraps

from opendate import Date

import libb
from libb import lazydict as attrdict

logger = logging.getLogger(__name__)


def _csv_reader_wrapper(reader):
    """Yield csv rows, logging and skipping any row that will not decode.

    Notes
    -----
    - Locates the row an encoding error came from, and does not choke
      on a null byte, so one bad line does not lose the file.
    """
    while True:
        try:
            yield next(reader)
        except csv.Error as err:
            logger.error(err)
        except StopIteration:
            break


def on_error_randomize(arg: int | None = None, kwarg: str | None = None) -> Callable:
    """Randomize specified kw/arg if there is an IO/EnvironmentError"""

    def wrapper(io_fn):
        @wraps(io_fn)
        def new_io_fn(*args, **kwargs):
            if arg is not None and len(args) > arg:
                filename = args[arg]
            elif kwarg and kwarg in kwargs:
                filename = kwargs[kwarg]
            else:
                filename = None  # StringIO, BytesIO
            try:
                return io_fn(*args, **kwargs)
            except OSError:
                if not filename:
                    raise
                name, ext = os.path.splitext(str(filename))
                new_filename = f'{name}{str(random.getrandbits(128))}{ext}'
                logger.warning(f'Could not write to file {filename}, retrying with {new_filename}')
                if arg is not None:
                    listargs = list(args)
                    listargs[arg] = new_filename
                    args = tuple(listargs)
                elif kwarg:
                    kwargs[kwarg] = new_filename
                return io_fn(*args, **kwargs)

        return new_io_fn

    return wrapper


def read_csv_rows(file_or_name, **kw) -> tuple[list, list]:
    """Read a DataSet from a csv file or open handle.

    Parameters
    ----------
    file_or_name : str or file-like
        Path to open, or a handle already open.
    **kw
        Passed to csv.reader, less `skips` and `rename_fields`.
        `skips` drops that many leading rows before the header.
        `rename_fields` is a callable taking the raw header row and
        returning the names to use.

    Returns
    -------
    tuple
        (rows, columns), where rows is a list of attrdict and columns
        is a list of (name, type) pairs. The caller builds the
        container, so this layer never names a container class.

    Notes
    -----
    - A header field may carry a type suffix: `name:s`, `age:i`,
      `score:f`, `on:b`, `when:d`. A field without one reads as str.
    - An empty field becomes None, and a blank row is skipped.
    - A row that will not decode is logged and skipped rather than
      raising, so one bad line does not lose the file.
    - `rename_fields` sees the header before the type suffix is
      split off, so a renamer must carry the suffix through or the
      column falls back to str.
    - Two headers sharing a name collapse into one column, the last
      winning. A source with repeated headers needs `rename_fields`
      to tell them apart.
    """

    def _parse(s, typ):
        if s == '':
            return None
        if typ is bool:
            s_lower = s.lower().strip()
            if s_lower in {'0', 'f', 'false', 'no'}:
                return False
            if s_lower in {'1', 't', 'true', 'yes'}:
                return True
            logger.warning(f'Unexpected bool value "{s}", treating as False')
            return False
        if typ in {datetime.date, Date}:
            return Date(*time.strptime(s[:10], '%Y-%m-%d')[:3])
        if typ is float:
            try:
                return float(s)
            except (ValueError, TypeError):
                pass
            try:
                parsed = libb.parse(s)
            except (ValueError, TypeError):
                return None
            return float(parsed) if parsed is not None else None
        if typ is int:
            # Notes:
            # - OverflowError joins the caught pair here and nowhere
            #   else: int(float('1e400')) sees inf, and only the int
            #   conversion has an infinity it cannot represent.
            # - Catching it keeps the promise that one bad cell costs
            #   its own value, not the rest of the file.
            try:
                return int(s)
            except (ValueError, TypeError):
                pass
            try:
                return int(float(s))
            except (ValueError, TypeError, OverflowError):
                pass
            try:
                parsed = libb.parse(s)
            except (ValueError, TypeError):
                return None
            try:
                return int(parsed) if parsed is not None else None
            except (ValueError, TypeError, OverflowError):
                return None
        if typ is str:
            return s.strip()
        raise ValueError(f'Unknown type: {typ}')

    # Close only a handle we opened. One the caller passed in is
    # theirs to close.
    opened = isinstance(file_or_name, str)
    f = pathlib.Path(file_or_name).open() if opened else file_or_name

    skips = kw.pop('skips', 0)
    rename_fields = kw.pop('rename_fields', None)
    reader = _csv_reader_wrapper(csv.reader(f, **kw))
    if skips:
        for _ in range(skips):
            _ = next(reader)
        logger.debug(f'Skipped {skips} rows in csv')

    fields = next(reader)
    if rename_fields is not None:
        fields = rename_fields(fields)

    TYPES = {
        'b': bool,
        'd': Date,
        'f': float,
        'i': int,
        's': str,
    }
    columns, types = [], []
    for fld in fields:
        if ':' in fld:
            col, typ = fld.rsplit(':', 1)
            types.append(TYPES[typ])
            columns.append(col)
        else:
            types.append(str)
            columns.append(fld)
    rows = []
    for row in reader:
        if not row:  # auto-skip empty rows
            continue
        pyvals = list(itertools.starmap(_parse, zip(row, types)))
        rows.append(attrdict(list(zip(columns, pyvals))))
    if opened:
        f.close()
    return rows, list(zip(columns, types))


def _emit(dataset, file_handle, header=True, **kwargs):
    """Render the dataset into an open csv writer."""
    def _format(row, col, typ):
        if row[col] is None:
            return ''
        return row[col]

    def _fmt_label(col):
        return col

    _format = kwargs.pop('format', _format)
    _format_label = kwargs.pop('format_label', _fmt_label)

    cols = [_format_label(c) for c, t in dataset.columns]

    writer = csv.writer(file_handle)
    if header:
        writer.writerow(cols)
    for row in dataset:
        try:
            writer.writerow([_format(row, c, t) for c, t in dataset.columns])
        except Exception as exc:
            logger.error(f'Problem with this row:\n{str(row)}')
            logger.exception(exc)
            raise


# Notes:
# - arg=1 is path_or_buf. It survived the move from a method
#   because (self, path) became (dataset, path); reordering
#   these two parameters would silently randomize the wrong one.
@on_error_randomize(arg=1)
def write_csv_file(dataset: 'DataSet', path_or_buf, **kwargs) -> None:
    """Write the DataSet to a csv file or buffer.

    Parameters
    ----------
    path_or_buf : str or file-like
        Path to write, or a buffer to write into.
    **kwargs
        `header` writes the column names, default True. `format`
        and `format_label` override how a value and a column name
        are rendered.

    Notes
    -----
    - Where the path cannot be written, a randomized name is tried
      once before giving up.
    - A None value is written as an empty field.
    """
    dataset.ensure_types()

    if isinstance(path_or_buf, str):
        with pathlib.Path(path_or_buf).open('w', newline='') as file_handle:
            _emit(dataset, file_handle, **kwargs)
    else:
        _emit(dataset, path_or_buf, **kwargs)


def to_json(dataset: 'DataSet', columns=None, raw=False,
            format_value=lambda x, y, _: x.get(y), **kw) -> str:
    """Serialize the DataSet to a json string.

    Parameters
    ----------
    dataset : DataSet
        Rows to serialize.
    columns : list of str or None, default None
        Columns to keep. None or empty keeps them all.
    raw : bool, default False
        If True, emit a bare array of row objects. If False, emit an
        object carrying `order`, `types` and `data`.
    format_value : Callable, default lambda x, y, _: x.get(y)
        Renders one value, taking (row, column, type).
    **kw
        Extra keys folded into the object. Ignored where `raw`.

    Returns
    -------
    str
        The json text. Dates are written in ISO form.
    """
    dataset.ensure_types()
    if columns is None:
        columns = []
    cols = [(c, t) for c, t in dataset.columns if c in columns] if columns else dataset.columns
    order, types = list(zip(*cols))
    types = [_.__name__ for _ in types]
    data = [{c: format_value(row, c, t) for c, t in cols} for row in dataset.container]
    if raw:
        return json.dumps(data, cls=libb.JSONEncoderISODate)
    else:
        data_d = dict(order=order, types=types, data=data, **kw)
        return json.dumps(data_d, cls=libb.JSONEncoderISODate)


def log_excel_errors(func: Callable) -> Callable:
    """Log the file, error type, and arguments when Excel parsing fails.

    Parameters
    ----------
    func : Callable
        Excel-parsing function to wrap.

    Returns
    -------
    Callable
        The wrapped function; it re-raises after logging.

    Notes
    -----
    - Meant to sit under `@classmethod`, and names the workbook either
      way.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            # The wrapper runs under @classmethod, so it is handed the
            # class before the file. Drop a leading class, or every
            # failure logs the same class repr instead of the workbook.
            file_args = args[1:] if args and isinstance(args[0], type) else args
            source = file_args[0] if file_args else kwargs.get(
                'file_or_name', 'unknown')
            error_details = [
                f'Excel parsing failed for file: {source}',
                f'Error type: {type(exc).__name__}',
                f'Error details: {exc}',
                f'Args: {args}',
                f'Kwargs: {kwargs}',
                ]

            if hasattr(exc, 'strerror'):
                error_details.append(f'System error: {exc.strerror}')

            if 'com_error' in str(type(exc)):
                error_details.append(
                    'This appears to be a COM/Excel automation error. '
                    'Check if Excel is properly installed and accessible.')

            logger.error('\n'.join(error_details))
            raise

    return wrapper
