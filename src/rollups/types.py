"""Column type inference and conversion.

Every column carries a Python type. These functions decide what that
type is when nobody declared one, and convert a value to it on read.

Notes
-----
- Date parsing is cached, because a date string repeats across rows.
  A dynamic date code is never cached, since it resolves against the
  current day.
"""
import contextlib
import datetime
import logging
import re
from functools import lru_cache

from opendate import Date, DateTime, Time

import libb

logger = logging.getLogger(__name__)


def infer_numeric_type(val: str) -> type | None:
    """Infer whether a string holds an int or a float.

    Parameters
    ----------
    val : str
        Candidate value. Anything that is not a str yields None.

    Returns
    -------
    type or None
        int or float when the string is confidently numeric, else None.

    Notes
    -----
    - A string that parses only as a float ('100.0', '1e6') never comes
      back as int.
    - A percentage is always float, never int.
    - A string holding an underscore is an identifier, not a number.
    - Digits behind a leading zero ('007', '0123') mark a code, not a
      number.
    """
    if not isinstance(val, str):
        return None

    if '_' in val:
        return None

    stripped = val.strip()
    check_str = stripped.lstrip('+-')

    if (len(check_str) >= 3 and check_str[0] == '0' and check_str[1].isdigit()
            and '.' not in check_str and 'e' not in check_str.lower()):
        return None

    if '%' in val:
        float_result = libb.numify(val, float)
        return float if float_result is not None else None

    # Ask for int before float: libb.numify applies the semantic checks
    # that reject a conversion the digits alone would allow.
    int_result = libb.numify(val, int)
    if int_result is not None:
        return int

    float_result = libb.numify(val, float)
    if float_result is not None:
        return float

    return None


def smart_type(val, infer_numeric_strings: bool = False):
    """Infer type from a value, preserving explicit DateTime/Date distinctions.

    Parameters
    ----------
    val : Any
        Value to infer the type from.
    infer_numeric_strings : bool, default False
        If True, a numeric string yields int or float rather than str.

    Returns
    -------
    type
        Inferred type for the value.
    """
    if val.__class__ == DateTime:
        return DateTime
    if val.__class__ == Date:
        return Date
    if val.__class__ == datetime.datetime \
            and val.hour == val.minute == val.second == val.microsecond == 0:
        logger.warning('Converting midnight datetime.datetime to Date type - incorrectly typing DateTime columns')
        return Date
    if infer_numeric_strings and isinstance(val, str):
        numeric_type = infer_numeric_type(val)
        if numeric_type is not None:
            return numeric_type
    return val.__class__


def is_dynamic_date_code(date_str: str) -> bool:
    """Check if a date string is a dynamic code that should not be cached.

    Parameters
    ----------
    date_str : str
        Candidate value. A non-str is never a dynamic code.

    Returns
    -------
    bool
        True for a dynamic code.

    Notes
    -----
    - The codes are T (today), Y (yesterday), P (previous business day),
      M (last month end) and N (now).
    - Each takes an optional offset, as in T-3 or P+2b.
    """
    if not isinstance(date_str, str):
        return False
    return bool(re.match(r'^[NTYPM]([-+]\d+)?b?$', date_str))


def islistoftuples(x):
    """Check that `x` is a sequence of (name, type) pairs, as `columns` wants.
    """
    return libb.issequence(x) and all(libb.issequence(y) and len(y) == 2 for y in x)


@lru_cache(maxsize=10000)
def _cached_date_parse(date_str: str):
    """Cache date string parsing - dates repeat often in datasets.
    """
    return Date.parse(date_str)


@lru_cache(maxsize=10000)
def _cached_datetime_parse(dt_str: str):
    """Cache datetime string parsing.
    """
    return DateTime.parse(dt_str)


def _convert_value(val, typ):
    """Convert a single value to the target type.

    Notes
    -----
    - A None value, an undeclared type, and a callable are all returned
      as they are.
    - A row is a `libb.lazydict`, so a callable value is the row's own
      computed column, resolved on attribute access. Converting it
      would freeze `str(function)` - its repr and address - into the
      data, and `str` is the one target type that accepts anything, so
      without this guard that column alone would be destroyed while
      every other declared type left it alone.
    """
    if val is None or typ is None or typ is None.__class__:
        return val
    if callable(val):
        return val
    if typ in {datetime.datetime, DateTime}:
        if isinstance(val, datetime.date):
            return DateTime.instance(val)
        elif isinstance(val, str):
            if is_dynamic_date_code(val):
                return DateTime.parse(val)
            return _cached_datetime_parse(val)
    if typ in {datetime.date, Date}:
        if isinstance(val, datetime.date):
            return Date.instance(val)
        elif isinstance(val, str):
            if is_dynamic_date_code(val):
                return Date.parse(val)
            return _cached_date_parse(val)
    if typ in {datetime.time, Time}:
        if isinstance(val, datetime.datetime | datetime.time):
            return Time.instance(val)
        elif isinstance(val, str):
            return Time.parse(val)
    with contextlib.suppress(Exception):
        if typ in {int, float}:
            result = libb.numify(val, typ)
            return result if result is not None else val
        return typ(val)
    return val


def force_type(somestr, date_fmt='%d%b%y'):
    """Force a string to float, else to DateTime, else leave it a string.

    Parameters
    ----------
    somestr : str
        Value to convert.
    date_fmt : str, default '%d%b%y'
        Format tried when the value is not a number.

    Returns
    -------
    float or DateTime or str
        The converted value, or the input unchanged.

    Notes
    -----
    - A parenthesized number is read as negative, so '(1.5)' is -1.5.
    """
    try:
        paren_re = r'^\(([0-9]*[.]?[0-9]*)\)$'
        neg_val = re.sub(paren_re, lambda x: f'-{x.group(1)}', somestr)
        return float(neg_val)
    except ValueError:
        try:
            return DateTime.strptime(somestr, date_fmt)
        except ValueError:
            return somestr
