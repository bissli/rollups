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
