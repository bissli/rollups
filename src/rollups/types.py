"""Column type inference and conversion.

Every column carries a Python type. These functions decide what that
type is when nobody declared one, and convert a value to it on read.

Notes
-----
- Date parsing is cached, because a date string repeats across rows.
  A dynamic date code is never cached, since it resolves against the
  current day.
"""


import logging


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
