"""An iterable of dictionaries, with typed columns.

`DataSet` is a list of dictionary rows. Each column declares a Python
type, and a value converts to it on first read rather than at
construction. The row is the primary object, and remains an ordinary
dict.

Notes
-----
- `attrdict`, `oset` and `emptydict` are libb's names, not this
  package's. Import them from libb: `from libb import lazydict as
  attrdict`, `OrderedSet as oset`, and `emptydict`. Note that a row is
  a `lazydict`, which is not the same class as `libb.attrdict`.
- See docs/README.md for the guides.
"""
from .core import DataSet, ensure_types_converted, find, force_type
from .core import guess_dataframe_dataset_columns, infer_numeric_type
from .core import is_dynamic_date_code, islistoftuples, log_excel_errors
from .core import on_error_randomize, smart_type
from .io import register_excel_backend
from .screen import apply_screen, get_or_val, interpret_screen, matches

__all__ = [
    'DataSet',
    'apply_screen',
    'ensure_types_converted',
    'find',
    'force_type',
    'get_or_val',
    'guess_dataframe_dataset_columns',
    'infer_numeric_type',
    'interpret_screen',
    'is_dynamic_date_code',
    'islistoftuples',
    'log_excel_errors',
    'matches',
    'on_error_randomize',
    'register_excel_backend',
    'smart_type',
    ]
