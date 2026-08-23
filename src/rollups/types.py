"""Column type inference and conversion.

Every column carries a Python type. These functions decide what that
type is when nobody declared one, and convert a value to it on read.

Notes
-----
- Date parsing is cached, because a date string repeats across rows.
  A dynamic date code is never cached, since it resolves against the
  current day.
"""


import datetime
import logging

from opendate import Date, DateTime

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
