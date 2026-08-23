"""DataFrame-native join, group-by, and construction.

Four functions taking and returning `pandas.DataFrame`, none of them
depending on `DataSet`. `join_dataframes` and `bucket_dataframe` mirror
`DataSet.join` and `DataSet.bucket`. `empty_dataframe` and
`dataframe_from_list` mirror the constructors of the same shape.

Notes
-----
- Every gap these functions introduce reads as null under `isna`. Which
  null it is follows the column's dtype: NaN, NaT, or pd.NA, and None
  where the column stays object. Read a gap with `isna`, never
  `is None`.
- An int column holding a gap widens to float, as it does anywhere else
  in pandas.
- See docs/dataframe-native.md for the contract differences against
  `DataSet.join` and `DataSet.bucket`.
"""


import logging
from collections.abc import Sequence


logger = logging.getLogger(__name__)


JOIN_TYPES = ('inner', 'outer', 'left', 'right', 'cross')


# Stands in for a null while two key columns are matched, so that a None
# on one side matches the NaN on the other.
NULL_KEY = object()


# Notes:
# - Ops whose pandas form already answers NA on an all-null group, so
#   the vectorized path and the per-group path agree on an empty group.
# - sum needs min_count=1 or pandas totals an all-null group to 0.
# - Anything else - len, a lambda, a libb.stats helper - falls through
#   to the per-group path, which runs the callable itself.
FAST_OPS = {
    sum: ('sum', {'min_count': 1}),
    max: ('max', {}),
    min: ('min', {}),
    }


def _free_name(stem: str, taken: Sequence[str]) -> str:
    """Name built from `stem` that collides with nothing in `taken`.

    Parameters
    ----------
    stem : str
        Preferred name.
    taken : sequence of str
        Names already in use.

    Returns
    -------
    str
        `stem`, or `stem` wrapped in underscores until it is free.
    """
    name = stem
    while name in taken:
        name = f'_{name}_'
    return name
