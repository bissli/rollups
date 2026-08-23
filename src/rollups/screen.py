"""A small query language for filtering a dataset by column.

A screen travels as text, so a saved filter can come from a config file
or a web form rather than from code. One query per column, and every
clause has to hold for a row to survive.

Notes
-----
- See docs/screening.md for the operators, the value rules, and the
  column-reference syntax.
"""


import logging
import operator
import re

import libb

logger = logging.getLogger(__name__)


def interpret_screen(screen):
    """Parse a screen query into its comparison conditions.

    Parameters
    ----------
    screen : str or None
        Comma-separated query, each clause an optional comparison token,
        a value, and an optional arithmetic suffix.

    Returns
    -------
    list of tuple or None
        (comparison, value, operator, operand) per clause, or None where
        `screen` is empty or not a str.

    Notes
    -----
    - A clause with no comparison token parses with a None comparison,
      which `matches` treats as a regex.
    - A value that parses as a number becomes one; one that does not
      stays a str, which is what preserves a grade such as 'B-'.
    - The arithmetic suffix is optional; without one the compared
      value is used unchanged.
    - See docs/screening.md for the query syntax.
    """
    if not screen or not isinstance(screen, str):
        return

    matchem = lambda x: re.match((
        r'(?P<cmp>[\<\>=!]{,2})'
        r'(?P<val>[+-]?[\w()%\.\| ]*(?:[+-]$)?)'
        r'(?P<op>[*\/\+\-]*)'
        r'(?P<op_val>[\d\.]*)'
        ), x.strip())

    def clean_val(cmp, val, op, op_val):
        op = op or '*'
        op_val = 1 if not op_val and op == '*' else libb.parse(op_val)
        COMPARE = {
            '<': operator.lt,
            '<=': operator.le,
            '=': operator.eq,
            '<>': operator.ne,
            '!=': operator.ne,
            '>=': operator.ge,
            '>': operator.gt,
            }
        OPERATOR = {
            '+': libb.safe_add,
            '-': libb.safe_diff,
            '*': libb.safe_mult,
            '/': libb.safe_divide,
            }
        cmp = COMPARE.get(cmp)
        op = OPERATOR.get(op)
        try:
            val = val.strip()
            if val.lower() in {'none', 'null'}:
                val = None
            elif val.endswith('%'):  # handle in output formatting
                val = float(val[:-1])
            else:
                val = float(val)
        except ValueError:
            val = str(val)
        return (cmp, val, op, op_val)

    cleaned = [clean_val(*m.groups()) for m in map(matchem, [s.strip() for s in screen.split(',')])]
    logger.debug(f'Parsed screen {screen} as {cleaned}')
    return cleaned


def get_or_val(cmp_val, row):
    """Resolve a screen value that may name another column.

    Parameters
    ----------
    cmp_val : Any
        Screen value. A leading underscore names a column of `row`.
    row : dict
        Row supplying the referenced value.

    Returns
    -------
    Any
        The referenced column's value, or `cmp_val` unchanged.
    """
    if str(cmp_val).startswith('_'):
        return row.get(cmp_val.lstrip('_'))
    return cmp_val
